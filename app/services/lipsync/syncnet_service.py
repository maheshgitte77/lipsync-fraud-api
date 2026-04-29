"""
SyncNet lip-sync analysis service.

Wraps the vendored `syncnet_python` scripts as subprocess calls. This keeps the
research code isolated and lets us switch to a native Python API later without
changing callers.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from app.core.config import settings
from app.core.device import device_manager
from app.core.logger import get_logger
from app.services.lipsync.window_builder import SyncNetWindow, prepare_windows

logger = get_logger("services.lipsync.syncnet")

_SYNCNET_BLOCK = re.compile(
    r"AV offset:\s*([-\d]+).*?Min dist:\s*([-\d.]+).*?Confidence:\s*([-\d.]+)",
    re.DOTALL,
)


class SyncNetError(RuntimeError):
    """Raised when the SyncNet subprocess fails."""


class SyncNetService:
    """Run run_pipeline.py + run_syncnet.py, parse scores, produce a verdict."""

    def __init__(self) -> None:
        self.syncnet_dir: Path = settings.paths.syncnet_dir

    # ---------- public API -------------------------------------------------

    def is_available(self) -> bool:
        return self.syncnet_dir.is_dir() and (self.syncnet_dir / "run_pipeline.py").is_file()

    def model_path(self) -> Path:
        p = self.syncnet_dir / "data" / "syncnet_v2.model"
        if not p.is_file():
            raise FileNotFoundError(
                f"SyncNet weights not found at {p}. Run download_model.sh inside syncnet_python."
            )
        return p

    def analyze_windowed(self, video_path: Path, reference: str, job_dir: Path) -> dict:
        """Run a single-window SyncNet pass and return the verdict dict."""
        try:
            windows, windowing = prepare_windows(video_path, job_dir)
        except Exception as exc:  # noqa: BLE001
            return {"passed": False, "verdict": "ERROR", "error": str(exc)[:8000], "scores": {}}

        if len(windows) == 1:
            result = self._analyze_single(windows[0].path, reference, job_dir / "w0")
            result["windowing"] = {
                **windowing,
                "windows": [
                    {
                        "index": 1,
                        "startSec": windows[0].start_sec,
                        "durationSec": windows[0].duration_sec,
                    }
                ],
            }
            return result
        return self._aggregate_windows(windows, reference, job_dir, windowing)

    def syncnet_only_fusion(self, syncnet_out: dict, *, skipped: bool) -> dict:
        if skipped:
            return {
                "verdict": "SKIPPED",
                "passed": True,
                "reason": "SyncNet was skipped by request/configuration.",
                "fusion_mode": "syncnet_only",
                "positive_methods": [],
            }
        passed = bool(syncnet_out.get("passed", False))
        return {
            "verdict": "PASS" if passed else "FAIL",
            "passed": passed,
            "reason": syncnet_out.get("reason", "SyncNet-only lip-sync result"),
            "fusion_mode": "syncnet_only",
            "positive_methods": ["syncnet"] if passed else [],
        }

    # ---------- internal helpers ------------------------------------------

    def _analyze_single(self, video_path: Path, reference: str, job_dir: Path) -> dict:
        try:
            scores = self._run(video_path, reference, job_dir)
            raw_tail = scores.pop("raw_log_tail", None)
            verdict = self._build_verdict(scores)
            out = {**verdict, "all_tracks": scores.get("all_tracks")}
            if settings.syncnet.debug_log and raw_tail:
                out["debug_log_tail"] = raw_tail
            return out
        except FileNotFoundError as exc:
            return {"passed": False, "verdict": "ERROR", "error": str(exc), "scores": {}}
        except (ValueError, RuntimeError) as exc:
            return {"passed": False, "verdict": "ERROR", "error": str(exc)[:8000], "scores": {}}

    def _aggregate_windows(
        self,
        windows: list[SyncNetWindow],
        reference: str,
        job_dir: Path,
        windowing: dict,
    ) -> dict:
        results: list[dict] = []
        for idx, w in enumerate(windows):
            res = self._analyze_single(w.path, f"{reference}_w{idx + 1}", job_dir / f"w{idx + 1}")
            results.append(
                {
                    "index": idx + 1,
                    "startSec": w.start_sec,
                    "durationSec": w.duration_sec,
                    "result": res,
                }
            )

        ok = [r for r in results if "error" not in r["result"]]
        if not ok:
            first_err = results[0]["result"].get("error", "No SyncNet window produced a score")
            return {
                "passed": False,
                "verdict": "ERROR",
                "error": first_err,
                "scores": {},
                "windowing": {**windowing, "windows": self._windows_for_meta(results)},
            }

        agg = {
            "av_offset_frames": 0,
            "min_dist": max(float(r["result"]["scores"].get("min_dist", 999.0)) for r in ok),
            "confidence": min(float(r["result"]["scores"].get("confidence", 0.0)) for r in ok),
            "tracks_evaluated": sum(int(r["result"]["scores"].get("tracks_evaluated", 0)) for r in ok),
            "mean_abs_offset_frames": round(
                sum(float(r["result"]["scores"].get("mean_abs_offset_frames", 0.0)) for r in ok)
                / max(len(ok), 1),
                3,
            ),
        }
        verdict = self._build_verdict(agg)
        if any(r["result"].get("passed") is False for r in ok):
            verdict["reason"] += " Dynamic windowing observed at least one failing window."
        verdict["windowing"] = {
            **windowing,
            "syncnetTrimEnabled": bool(settings.syncnet.trim_enabled),
            "syncnetTrimMaxSeconds": max(1.0, min(settings.syncnet.trim_max_seconds, 600.0)),
            "windows": self._windows_for_meta(results),
        }
        return verdict

    @staticmethod
    def _windows_for_meta(results: list[dict]) -> list[dict]:
        return [
            {
                "index": r["index"],
                "startSec": r["startSec"],
                "durationSec": r["durationSec"],
                "verdict": r["result"].get("verdict"),
                "passed": r["result"].get("passed"),
                "error": r["result"].get("error"),
                "scores": r["result"].get("scores", {}),
            }
            for r in results
        ]

    def _run(self, video_path: Path, reference: str, data_dir: Path) -> dict:
        video_path = video_path.resolve()
        data_dir = data_dir.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        model = self.model_path()

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        if not device_manager.is_gpu():
            env.setdefault("CUDA_VISIBLE_DEVICES", "")

        def run_script(name: str) -> subprocess.CompletedProcess:
            cmd = [
                sys.executable,
                str(self.syncnet_dir / name),
                "--videofile",
                str(video_path),
                "--reference",
                reference,
                "--data_dir",
                str(data_dir),
            ]
            if name == "run_pipeline.py":
                cmd.extend(self._pipeline_extras())
            if name == "run_syncnet.py":
                cmd.extend(["--initial_model", str(model)])
                cmd.extend(self._syncnet_extras())
            logger.debug("syncnet cmd: %s", " ".join(cmd))
            return subprocess.run(
                cmd,
                cwd=str(self.syncnet_dir),
                capture_output=True,
                text=True,
                env=env,
                encoding="utf-8",
                errors="replace",
            )

        p1 = run_script("run_pipeline.py")
        if p1.returncode != 0:
            err = (p1.stderr or "") + (p1.stdout or "")
            raise SyncNetError(f"run_pipeline.py failed (exit {p1.returncode}):\n{err[-8000:]}")

        p2 = run_script("run_syncnet.py")
        out = (p2.stdout or "") + "\n" + (p2.stderr or "")
        if p2.returncode != 0:
            raise SyncNetError(f"run_syncnet.py failed (exit {p2.returncode}):\n{out[-8000:]}")

        tracks = self._parse(out)
        if not tracks:
            raise ValueError("Could not parse SyncNet scores. Raw tail:\n" + out[-4000:])
        agg = self._aggregate(tracks)
        agg["raw_log_tail"] = out[-6000:].strip()
        return agg

    @staticmethod
    def _parse(stdout: str) -> list[dict]:
        tracks = []
        for m in _SYNCNET_BLOCK.finditer(stdout):
            tracks.append(
                {
                    "av_offset_frames": int(m.group(1)),
                    "min_dist": float(m.group(2)),
                    "confidence": float(m.group(3)),
                }
            )
        return tracks

    @staticmethod
    def _aggregate(tracks: list[dict]) -> dict:
        if not tracks:
            raise ValueError(
                "No SyncNet scores in subprocess output. "
                "Face may be missing, video too short, or pipeline failed silently."
            )
        worst = max(tracks, key=lambda t: t["min_dist"])
        min_conf = min(t["confidence"] for t in tracks)
        mean_offset = sum(abs(t["av_offset_frames"]) for t in tracks) / len(tracks)
        return {
            "av_offset_frames": worst["av_offset_frames"],
            "min_dist": worst["min_dist"],
            "confidence": min_conf,
            "tracks_evaluated": len(tracks),
            "all_tracks": tracks,
            "mean_abs_offset_frames": round(mean_offset, 3),
        }

    @staticmethod
    def _build_verdict(scores: dict) -> dict:
        min_dist = scores["min_dist"]
        confidence = scores["confidence"]
        sync_ok = min_dist <= settings.syncnet.min_dist_pass
        confident_ok = confidence >= settings.syncnet.confidence_pass
        passed = sync_ok and confident_ok

        if passed:
            reason = "Lip motion and audio appear in sync; candidate likely speaking in the recording."
        elif not sync_ok and not confident_ok:
            reason = "Poor audio-visual sync and low model confidence; possible dubbed or mismatched audio."
        elif not sync_ok:
            reason = "Audio-visual sync distance is high; possible dubbed or pre-recorded audio."
        else:
            reason = "Sync distance is acceptable but confidence is low; recommend manual review."

        return {
            "verdict": "PASS" if passed else "FAIL",
            "passed": passed,
            "reason": reason,
            "scores": {
                "av_offset_frames": scores["av_offset_frames"],
                "min_dist": round(min_dist, 4),
                "confidence": round(confidence, 4),
                "tracks_evaluated": scores["tracks_evaluated"],
                "mean_abs_offset_frames": scores["mean_abs_offset_frames"],
            },
            "thresholds": {
                "min_dist_max_pass": settings.syncnet.min_dist_pass,
                "confidence_min_pass": settings.syncnet.confidence_pass,
            },
        }

    @staticmethod
    def _pipeline_extras() -> list[str]:
        out: list[str] = []
        s = settings.syncnet
        if s.frame_rate:
            out.extend(["--frame_rate", s.frame_rate])
        if s.min_track:
            out.extend(["--min_track", s.min_track])
        if s.facedet_scale:
            out.extend(["--facedet_scale", s.facedet_scale])
        if s.crop_scale:
            out.extend(["--crop_scale", s.crop_scale])
        if s.num_failed_det:
            out.extend(["--num_failed_det", s.num_failed_det])
        if s.min_face_size:
            out.extend(["--min_face_size", s.min_face_size])
        return out

    @staticmethod
    def _syncnet_extras() -> list[str]:
        out: list[str] = []
        s = settings.syncnet
        if s.batch_size:
            out.extend(["--batch_size", s.batch_size])
        if s.vshift:
            out.extend(["--vshift", s.vshift])
        return out


syncnet_service = SyncNetService()
