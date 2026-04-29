"""
End-to-end proctor signals pipeline.

Coordinates: download → normalize → run SyncNet + pose analysis (in parallel) →
fuse verdicts → assemble flags → return structured response.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from app.core.metrics import StageTimer
from app.models.proctor import ProctorSignalsRequest
from app.services.fraud.proctor_service import ProctorThresholds, proctor_service
from app.services.lipsync.syncnet_service import syncnet_service
from app.utils.ffmpeg import normalize_container
from app.utils.file_manager import job_dir
from app.utils.video_download import download_to_file, guess_video_extension

logger = get_logger("services.orchestration.proctor")


class ProctorOrchestrator:
    """High-level entry point used by HTTP + Kafka paths."""

    def execute(self, payload: ProctorSignalsRequest, *, job_id: str) -> dict[str, Any]:
        reference = f"cand_{job_id}"
        work_dir = job_dir(job_id)
        ext = guess_video_extension(payload.videoUrl)
        video_path = work_dir / f"input{ext}"
        timer = StageTimer()
        try:
            with timer.track("download"):
                download_to_file(payload.videoUrl, video_path)
            with timer.track("normalize"):
                syncnet_path = normalize_container(video_path, work_dir, job_id=job_id)

            thresholds = self._thresholds(payload)
            skip_sn = settings.proctor.skip_syncnet or bool(payload.skipSyncNet)

            if skip_sn:
                with timer.track("pose"):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pose_result = pool.submit(proctor_service.analyze, video_path, thresholds).result()
                syncnet_out: dict[str, Any] = {"skipped": True, "passed": False, "scores": {}}
                fused_lipsync = syncnet_service.syncnet_only_fusion(syncnet_out, skipped=True)
            else:
                with timer.track("syncnet+pose"):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        fut_sn = pool.submit(
                            syncnet_service.analyze_windowed, syncnet_path, reference, work_dir
                        )
                        fut_pose = pool.submit(proctor_service.analyze, video_path, thresholds)
                        syncnet_result = fut_sn.result()
                        pose_result = fut_pose.result()
                syncnet_out = {k: v for k, v in syncnet_result.items() if k != "all_tracks"}
                fused_lipsync = syncnet_service.syncnet_only_fusion(syncnet_out, skipped=False)

            flags = self._compute_flags(
                fused_lipsync=fused_lipsync,
                syncnet_out=syncnet_out,
                pose_result=pose_result,
                thresholds=thresholds,
                skip_sn=skip_sn,
            )
            suspicious = len(flags) > 0
            return {
                "jobId": job_id,
                "questionId": payload.questionId,
                "candidateId": payload.candidateId,
                "videoUrl": payload.videoUrl,
                "lipSync": {
                    "passed": fused_lipsync.get("passed", False),
                    "verdict": fused_lipsync.get("verdict"),
                    "reason": fused_lipsync.get("reason"),
                    "fusion": {
                        "mode": "syncnet_only",
                        "positiveMethods": fused_lipsync.get("positive_methods", []),
                        "syncNetSkipped": skip_sn,
                        "flagSource": settings.proctor.lipsync_flag_source,
                        "mediapipeLipSyncSkipped": True,
                    },
                    "syncnet": syncnet_out,
                },
                "eyeMovement": pose_result["eyeMovement"],
                "headPose": pose_result["headPose"],
                "videoMeta": pose_result["videoMeta"],
                "integrityAnalysis": {
                    "verdict": "SUSPECT" if suspicious else "CLEAR",
                    "confidenceScore": 0.88 if suspicious else 0.2,
                    "flags": flags,
                },
                "summary": {
                    "suspicious": suspicious,
                    "signalCount": len(flags),
                    "rules": pose_result["summary"]["rules"],
                },
                "metrics": timer.snapshot(),
            }
        except Exception as exc:
            logger.exception("proctor job %s failed", job_id)
            raise RuntimeError(str(exc)[:4000]) from exc
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _thresholds(payload: ProctorSignalsRequest) -> ProctorThresholds:
        p = settings.proctor
        return ProctorThresholds(
            sample_fps=payload.sampleFps or p.sample_fps,
            offscreen_ratio_threshold=payload.offscreenRatioThreshold or p.offscreen_ratio_threshold,
            improper_head_ratio_threshold=payload.improperHeadRatioThreshold or p.improper_head_ratio_threshold,
            repetitive_pattern_threshold=payload.repetitivePatternThreshold or p.repetitive_pattern_threshold,
        )

    @staticmethod
    def _compute_flags(
        *,
        fused_lipsync: dict,
        syncnet_out: dict,
        pose_result: dict,
        thresholds: ProctorThresholds,
        skip_sn: bool,
    ) -> list[dict]:
        flag_source = settings.proctor.lipsync_flag_source
        if flag_source == "none" or skip_sn:
            lip_sync_mismatch = False
        else:
            syncnet_verdict = str(syncnet_out.get("verdict") or "").upper()
            if syncnet_verdict == "ERROR" or bool(syncnet_out.get("error")):
                lip_sync_mismatch = False
            else:
                lip_sync_mismatch = not bool(syncnet_out.get("passed", False))

        eye_tracking = pose_result["eyeMovement"].get("eyeTracking", {})
        eye_reliable = bool(eye_tracking.get("reliable", True))
        offscreen_ratio = pose_result["eyeMovement"]["offScreenRatio"]
        improper_head_ratio = pose_result["headPose"]["improperHeadRatio"]
        repetitive_pattern_count = int(
            pose_result["eyeMovement"].get("repetitivePatternCount", 0) or 0
        )
        reading_pattern_score = float(pose_result["eyeMovement"].get("readingPatternScore", 0.0) or 0.0)
        sustained = pose_result["eyeMovement"].get("sustainedGaze", {}) or {}
        sustained_detected = bool(sustained.get("detected", False))
        direction_counts = pose_result["eyeMovement"].get("directionCounts", {}) or {}
        up_count = int(direction_counts.get("UP", 0) or 0)
        down_count = int(direction_counts.get("DOWN", 0) or 0)
        non_center = (
            int(direction_counts.get("LEFT", 0) or 0)
            + int(direction_counts.get("RIGHT", 0) or 0)
            + up_count
            + down_count
        )
        valid_eye_frames = int(pose_result["eyeMovement"].get("validEyeFrames", 0) or 0)
        eye_movement_flag = valid_eye_frames > 0 and (non_center / valid_eye_frames) >= 0.4
        downward_evidence_min = max(2, int(valid_eye_frames * 0.06))
        has_downward_reading_evidence = down_count >= downward_evidence_min and down_count >= up_count
        reading_from_external_flag = eye_reliable and (
            sustained_detected
            or (
                has_downward_reading_evidence
                and (
                    (
                        offscreen_ratio >= thresholds.offscreen_ratio_threshold
                        and repetitive_pattern_count >= thresholds.repetitive_pattern_threshold
                    )
                    or reading_pattern_score >= 0.60
                )
            )
        )
        head_flag = improper_head_ratio >= thresholds.improper_head_ratio_threshold

        flags: list[dict] = []
        if lip_sync_mismatch:
            flags.append({"type": "LIP_SYNC_MISMATCH", "severity": "HIGH"})
        if reading_from_external_flag:
            flags.append({"type": "READING_FROM_EXTERNAL", "severity": "MEDIUM"})
        if eye_movement_flag:
            flags.append({"type": "EYE_MOVEMENT", "severity": "MEDIUM"})
        if head_flag:
            flags.append({"type": "IMPROPER_HEAD_POSE", "severity": "MEDIUM"})
        return flags


proctor_orchestrator = ProctorOrchestrator()


def _filter_empty_path(path: Path) -> Path:  # pragma: no cover - helper export
    return path
