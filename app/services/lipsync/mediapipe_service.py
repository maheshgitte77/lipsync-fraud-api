"""
MediaPipe lip-openness vs audio-energy correlation service.

Supports both the legacy `mediapipe.solutions.face_mesh` API and the newer
Tasks API. Exposes a single analyze() method returning a verdict dict.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import librosa
import numpy as np

from app.core.config import settings
from app.core.logger import get_logger
from app.utils.ffmpeg import extract_wav

logger = get_logger("services.lipsync.mediapipe")

_UPPER_LIP = 13
_LOWER_LIP = 14

_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _default_model_cache_dir() -> Path:
    env = os.environ.get("MEDIAPIPE_MODEL_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return settings.paths.project_root / ".cache"


def resolve_face_landmarker_model() -> Path:
    """Path to the FaceLandmarker .task weights. Downloads on first use."""
    explicit = os.environ.get("MEDIAPIPE_FACE_MODEL_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"MEDIAPIPE_FACE_MODEL_PATH not found: {p}")
        return p

    cache = _default_model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / "face_landmarker.task"
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest

    url = os.environ.get("MEDIAPIPE_FACE_MODEL_URL", _DEFAULT_MODEL_URL).strip()
    tmp = dest.with_suffix(".task.download")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"{settings.app_name}/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(dest)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return dest


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    rng = hi - lo
    if rng <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / rng


def _lip_gap(lm_list) -> float | None:
    if not lm_list or len(lm_list) <= max(_UPPER_LIP, _LOWER_LIP):
        return None
    return abs(float(lm_list[_LOWER_LIP].y - lm_list[_UPPER_LIP].y))


class MediaPipeCorrelationService:
    """Correlate lip-openness with audio RMS energy frame-by-frame."""

    def analyze(
        self,
        video_path: str | Path,
        *,
        corr_threshold: float = 0.4,
        corr_borderline: float = 0.25,
        audio_sample_rate: int = 16000,
        fps_fallback: float = 25.0,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_frames: int = 5,
    ) -> dict:
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            return {"error": f"Video not found: {video_path}", "passed": False}

        audio_fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(audio_fd)
        try:
            try:
                extract_wav(video_path, Path(audio_path), sample_rate=audio_sample_rate)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"ffmpeg failed or empty audio: {exc}", "passed": False}
            y, sr = librosa.load(audio_path, sr=int(audio_sample_rate), mono=True)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "OpenCV could not open video", "passed": False}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = float(fps_fallback)

        try:
            lip_openness, backend = self._collect(
                cap,
                fps=fps,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            cap.release()
            return {"passed": False, "verdict": "ERROR", "error": str(exc)[:2500], "scores": {}}
        cap.release()

        n_frames = len(lip_openness)
        if n_frames < int(min_frames):
            return {
                "error": "Too few video frames for correlation",
                "passed": False,
                "frames_processed": n_frames,
            }

        hop = max(1, int(sr / fps))
        audio_energy: list[float] = []
        for i in range(n_frames):
            start = i * hop
            end = start + hop
            chunk = y[start:end] if end <= len(y) else y[start:]
            if len(chunk) > 0:
                audio_energy.append(float(np.sqrt(np.mean(np.square(chunk)))))
            else:
                audio_energy.append(0.0)

        lip_norm = _normalize(np.array(lip_openness))
        audio_norm = _normalize(np.array(audio_energy))

        if np.std(lip_norm) < 1e-9 or np.std(audio_norm) < 1e-9:
            correlation = 0.0
        else:
            c = np.corrcoef(lip_norm, audio_norm)[0, 1]
            correlation = 0.0 if np.isnan(c) else float(c)

        passed = correlation >= corr_threshold
        if passed:
            reason = "Lip openness and audio energy are correlated; plausible co-speech activity."
        elif correlation >= float(corr_borderline):
            reason = "Weak lip–audio correlation; borderline or noisy conditions."
        else:
            reason = "Low lip–audio correlation; possible muted video, dubbed audio, or poor face visibility."

        return {
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
            "reason": reason,
            "scores": {
                "correlation": round(correlation, 4),
                "frames_processed": n_frames,
                "fps_used": round(fps, 3),
                "mediapipe_backend": backend,
            },
            "thresholds": {
                "correlation_min_pass": corr_threshold,
                "correlation_borderline": corr_borderline,
            },
        }

    def _collect(
        self,
        cap: "cv2.VideoCapture",
        *,
        fps: float,
        max_num_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> tuple[list[float], str]:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return self._collect_solutions(
                cap,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        return self._collect_tasks(
            cap,
            fps=fps,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _collect_solutions(
        cap: "cv2.VideoCapture",
        *,
        max_num_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> tuple[list[float], str]:
        import mediapipe as mp

        mp_face = mp.solutions.face_mesh
        mesh_kw = dict(
            static_image_mode=False,
            max_num_faces=max(1, int(max_num_faces)),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        try:
            face_mesh = mp_face.FaceMesh(refine_landmarks=True, **mesh_kw)
        except TypeError:
            face_mesh = mp_face.FaceMesh(**mesh_kw)

        lip_openness: list[float] = []
        with face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0].landmark
                    gap = _lip_gap(lm)
                    lip_openness.append(gap if gap is not None else 0.0)
                else:
                    lip_openness.append(0.0)
        return lip_openness, "mediapipe.solutions.face_mesh"

    @staticmethod
    def _collect_tasks(
        cap: "cv2.VideoCapture",
        *,
        fps: float,
        max_num_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> tuple[list[float], str]:
        import mediapipe as mp
        from mediapipe.tasks.python.core import base_options as base_options_lib
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        model_path = resolve_face_landmarker_model()
        base = base_options_lib.BaseOptions(model_asset_path=str(model_path))
        options = FaceLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.VIDEO,
            num_faces=max(1, int(max_num_faces)),
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

        lip_openness: list[float] = []
        landmarker = FaceLandmarker.create_from_options(options)
        try:
            frame_ms = max(1, int(round(1000.0 / max(fps, 1e-3))))
            timestamp_ms = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp_ms += frame_ms
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    gap = _lip_gap(lm)
                    lip_openness.append(gap if gap is not None else 0.0)
                else:
                    lip_openness.append(0.0)
        finally:
            landmarker.close()
        return lip_openness, "mediapipe.tasks.vision.FaceLandmarker"
