"""
Proctor signal analysis: eye gaze + head pose via MediaPipe FaceLandmarker.

Exposes:
    analyze_eye_head_pose(video_path, thresholds) -> dict
    ProctorThresholds
    proctor_service (singleton convenience wrapper)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from app.core.logger import get_logger
from app.services.lipsync.mediapipe_service import resolve_face_landmarker_model

logger = get_logger("services.fraud.proctor")


# ---- landmark indices (478-point FaceLandmarker topology) ----
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_IRIS_CENTER = 468
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_IRIS_CENTER = 473
LEFT_UPPER_EYELID = 159
LEFT_LOWER_EYELID = 145
RIGHT_UPPER_EYELID = 386
RIGHT_LOWER_EYELID = 374
NOSE_TIP = 1
CHIN = 152


@dataclass
class ProctorThresholds:
    sample_fps: float = 1.0
    offscreen_ratio_threshold: float = 0.35
    improper_head_ratio_threshold: float = 0.35
    repetitive_pattern_threshold: int = 6


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


class _OnlineStats:
    """Numerically stable running mean/std (Welford)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / float(self.n)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / float(self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _euler_from_transform_matrix(matrix_4x4: list[float]) -> tuple[float, float, float] | None:
    """
    Best-effort yaw/pitch/roll extraction from MediaPipe facial transformation matrix.
    Returns degrees (yaw, pitch, roll) or None if parsing fails.
    """
    if not matrix_4x4 or len(matrix_4x4) < 16:
        return None
    # MediaPipe matrix is typically row-major 4x4.
    r02 = matrix_4x4[2]
    r10, r11, r12 = matrix_4x4[4], matrix_4x4[5], matrix_4x4[6]
    r22 = matrix_4x4[10]

    # Standard yaw-pitch-roll extraction (Y-X-Z order approximation).
    pitch = math.asin(_clamp(-r12, -1.0, 1.0))
    yaw = math.atan2(r02, r22)
    roll = math.atan2(r10, r11)
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _extract_face_transform_matrix(res) -> list[float] | None:
    mats = getattr(res, "facial_transformation_matrixes", None)
    if not mats:
        return None
    first = mats[0]
    data = getattr(first, "data", None)
    if not data:
        return None
    try:
        flat = [float(x) for x in data]
    except Exception:
        return None
    if len(flat) < 16:
        return None
    return flat[:16]


def _classify_head_pose(
    landmarks,
    *,
    yaw_left_threshold: float = -0.18,
    yaw_right_threshold: float = 0.18,
    pitch_up_threshold: float = 0.12,
    pitch_down_threshold: float = 0.50,
) -> str:
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    eye_center_x = (left_eye.x + right_eye.x) / 2.0
    eye_center_y = (left_eye.y + right_eye.y) / 2.0
    eye_span_x = abs(right_eye.x - left_eye.x)
    yaw = _safe_ratio(nose.x - eye_center_x, eye_span_x, 0.0)
    pitch = _safe_ratio(nose.y - eye_center_y, abs(chin.y - eye_center_y), 0.0)
    if yaw < yaw_left_threshold:
        return "LEFT"
    if yaw > yaw_right_threshold:
        return "RIGHT"
    if pitch < pitch_up_threshold:
        return "UP"
    if pitch > pitch_down_threshold:
        return "DOWN"
    return "FRONTAL"


def _iris_metrics(landmarks) -> dict[str, float]:
    left_outer = landmarks[LEFT_EYE_OUTER]
    left_inner = landmarks[LEFT_EYE_INNER]
    left_iris = landmarks[LEFT_IRIS_CENTER]
    right_inner = landmarks[RIGHT_EYE_INNER]
    right_outer = landmarks[RIGHT_EYE_OUTER]
    right_iris = landmarks[RIGHT_IRIS_CENTER]

    left_x = _safe_ratio(left_iris.x - left_outer.x, max(abs(left_inner.x - left_outer.x), 1e-6), 0.5)
    right_x = _safe_ratio(right_iris.x - right_inner.x, max(abs(right_outer.x - right_inner.x), 1e-6), 0.5)
    x_norm = (left_x + right_x) / 2.0

    eye_center_y = (
        landmarks[LEFT_EYE_OUTER].y
        + landmarks[LEFT_EYE_INNER].y
        + landmarks[RIGHT_EYE_OUTER].y
        + landmarks[RIGHT_EYE_INNER].y
    ) / 4.0
    left_lid = abs(landmarks[LEFT_LOWER_EYELID].y - landmarks[LEFT_UPPER_EYELID].y)
    right_lid = abs(landmarks[RIGHT_LOWER_EYELID].y - landmarks[RIGHT_UPPER_EYELID].y)
    avg_lid = (left_lid + right_lid) / 2.0
    y_norm = _safe_ratio(((left_iris.y + right_iris.y) / 2.0) - eye_center_y, max(avg_lid, 1e-6), 0.0)
    return {"x_norm": x_norm, "y_norm": y_norm, "lid_span": avg_lid}


def _classify_black_dot_direction(
    x_norm: float,
    y_norm: float,
    *,
    horizontal_center_band: float = 0.12,
    horizontal_bias: float = 0.0,
    vertical_up_band: float = 0.24,
    vertical_down_band: float = 0.16,
    vertical_bias: float = 0.0,
) -> str:
    center_x = 0.5 + horizontal_bias
    if x_norm < (center_x - horizontal_center_band):
        return "LEFT"
    if x_norm > (center_x + horizontal_center_band):
        return "RIGHT"
    adjusted_y = y_norm - vertical_bias
    if adjusted_y < -vertical_up_band:
        return "UP"
    if adjusted_y > vertical_down_band:
        return "DOWN"
    return "CENTER"


def _seconds_to_mmss(seconds: float) -> str:
    sec = max(0, int(seconds))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _segment_from_indices(indices: list[int], fps: float) -> list[dict[str, Any]]:
    if not indices:
        return []
    segments: list[dict[str, Any]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx != prev + 1:
            segments.append(
                {
                    "startSec": round(start / fps, 2),
                    "endSec": round(prev / fps, 2),
                    "range": f"{_seconds_to_mmss(start / fps)}-{_seconds_to_mmss(prev / fps)}",
                    "frameCount": prev - start + 1,
                }
            )
            start = idx
        prev = idx
    segments.append(
        {
            "startSec": round(start / fps, 2),
            "endSec": round(prev / fps, 2),
            "range": f"{_seconds_to_mmss(start / fps)}-{_seconds_to_mmss(prev / fps)}",
            "frameCount": prev - start + 1,
        }
    )
    return segments


def _segments_with_direction(indices: list[int], fps: float, direction: str) -> list[dict[str, Any]]:
    segs = _segment_from_indices(indices, fps)
    for s in segs:
        s["direction"] = direction
    return segs


def _count_scan_cycles(seq: list[str], pattern: tuple[str, str, str]) -> int:
    if len(seq) < 3:
        return 0
    compressed: list[str] = []
    for d in seq:
        if not compressed or compressed[-1] != d:
            compressed.append(d)
    a, b, c = pattern
    cycles = 0
    i = 0
    while i <= len(compressed) - 3:
        if compressed[i] == a and compressed[i + 1] == b and compressed[i + 2] == c:
            cycles += 1
            i += 3
        else:
            i += 1
    return cycles


def _count_direction_to_center_cycles(seq: list[str], direction: str) -> int:
    if len(seq) < 2:
        return 0
    compressed: list[str] = []
    for d in seq:
        if not compressed or compressed[-1] != d:
            compressed.append(d)
    cycles = 0
    for i in range(len(compressed) - 1):
        if compressed[i] == direction and compressed[i + 1] == "CENTER":
            cycles += 1
    return cycles


def _count_repetitive_center_oscillation(seq: list[str], direction: str) -> int:
    if len(seq) < 4:
        return 0
    compressed: list[str] = []
    for d in seq:
        if not compressed or compressed[-1] != d:
            compressed.append(d)
    count = 0
    i = 0
    while i <= len(compressed) - 4:
        if (
            compressed[i] == direction
            and compressed[i + 1] == "CENTER"
            and compressed[i + 2] == direction
            and compressed[i + 3] == "CENTER"
        ):
            count += 1
            i += 4
        else:
            i += 1
    return count


def _create_face_landmarker_video():
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    model_path = resolve_face_landmarker_model()
    base = base_options_lib.BaseOptions(model_asset_path=str(model_path))
    options = FaceLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def _maybe_downscale_bgr(frame, max_width: int):
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (max_width, nh), interpolation=cv2.INTER_AREA)


def analyze_eye_head_pose(
    video_path: Path, thresholds: ProctorThresholds | None = None
) -> dict[str, Any]:
    """Full proctor pass: iris-based gaze + head-pose + reading-pattern score."""
    cfg = thresholds or ProctorThresholds()
    max_frame_w = int(os.environ.get("PROCTOR_MAX_FRAME_WIDTH", "480") or "480")
    fps_fallback = float(os.environ.get("PROCTOR_FPS_FALLBACK", "25.0") or "25.0")
    fps_min = float(os.environ.get("PROCTOR_FPS_MIN", "5.0") or "5.0")
    fps_max = float(os.environ.get("PROCTOR_FPS_MAX", "120.0") or "120.0")
    iris_blink_lid_span_min = float(os.environ.get("EYE_IRIS_BLINK_LID_SPAN_MIN", "0.0055") or "0.0055")
    iris_jitter_delta_min = float(os.environ.get("EYE_IRIS_JITTER_DELTA_MIN", "0.015") or "0.015")
    head_pose_use_transform = str(os.environ.get("HEAD_POSE_USE_TRANSFORM", "true") or "true").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    head_yaw_left_threshold = float(os.environ.get("HEAD_YAW_LEFT_THRESHOLD", "-0.18") or "-0.18")
    head_yaw_right_threshold = float(os.environ.get("HEAD_YAW_RIGHT_THRESHOLD", "0.18") or "0.18")
    head_pitch_up_threshold = float(os.environ.get("HEAD_PITCH_UP_THRESHOLD", "0.12") or "0.12")
    head_pitch_down_threshold = float(os.environ.get("HEAD_PITCH_DOWN_THRESHOLD", "0.50") or "0.50")
    head_yaw_left_deg = float(os.environ.get("HEAD_YAW_LEFT_DEG", "-18") or "-18")
    head_yaw_right_deg = float(os.environ.get("HEAD_YAW_RIGHT_DEG", "18") or "18")
    head_pitch_up_deg = float(os.environ.get("HEAD_PITCH_UP_DEG", "-12") or "-12")
    head_pitch_down_deg = float(os.environ.get("HEAD_PITCH_DOWN_DEG", "15") or "15")
    head_pose_hold_frames = int(os.environ.get("HEAD_POSE_HOLD_FRAMES", "2") or "2")
    eye_count_only_when_frontal = str(
        os.environ.get("EYE_COUNT_ONLY_WHEN_FRONTAL", "true") or "true"
    ).lower() in ("1", "true", "yes", "y")
    eye_horizontal_center_band = float(os.environ.get("EYE_HORIZONTAL_CENTER_BAND", "0.12") or "0.12")
    eye_vertical_up_band = float(os.environ.get("EYE_VERTICAL_UP_BAND", "0.24") or "0.24")
    eye_vertical_down_band = float(os.environ.get("EYE_VERTICAL_DOWN_BAND", "0.16") or "0.16")
    eye_vertical_baseline_alpha = float(os.environ.get("EYE_VERTICAL_BASELINE_ALPHA", "0.15") or "0.15")
    eye_vertical_baseline_clamp = float(os.environ.get("EYE_VERTICAL_BASELINE_CLAMP", "0.12") or "0.12")
    eye_adaptive_enabled = str(os.environ.get("EYE_ADAPTIVE_ENABLED", "true") or "true").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    eye_adaptive_min_samples = int(os.environ.get("EYE_ADAPTIVE_MIN_SAMPLES", "20") or "20")
    eye_adaptive_k_horizontal = float(os.environ.get("EYE_ADAPTIVE_K_HORIZONTAL", "2.8") or "2.8")
    eye_adaptive_k_up = float(os.environ.get("EYE_ADAPTIVE_K_UP", "3.2") or "3.2")
    eye_adaptive_k_down = float(os.environ.get("EYE_ADAPTIVE_K_DOWN", "2.6") or "2.6")
    eye_adaptive_band_min = float(os.environ.get("EYE_ADAPTIVE_BAND_MIN", "0.08") or "0.08")
    eye_adaptive_band_max = float(os.environ.get("EYE_ADAPTIVE_BAND_MAX", "0.30") or "0.30")
    eye_adaptive_std_min = float(os.environ.get("EYE_ADAPTIVE_STD_MIN", "0.015") or "0.015")
    eye_steady_direction_hold_frames = int(os.environ.get("EYE_STEADY_DIRECTION_HOLD_FRAMES", "3") or "3")
    proctor_warmup_sec = float(os.environ.get("PROCTOR_WARMUP_SEC", "2.0") or "2.0")
    proctor_cooldown_sec = float(os.environ.get("PROCTOR_COOLDOWN_SEC", "2.0") or "2.0")
    sustained_gaze_sec_threshold = float(os.environ.get("EYE_SUSTAINED_GAZE_SEC", "3.0") or "3.0")
    eye_reliable_min = float(os.environ.get("EYE_RELIABILITY_MIN", "0.4") or "0.4")
    eye_reliable_strong = float(os.environ.get("EYE_RELIABILITY_STRONG", "0.7") or "0.7")
    eye_scan_cycles_threshold = int(os.environ.get("EYE_SCAN_CYCLES_THRESHOLD", "6") or "6")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video for eye/head analysis")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    native_fps = raw_fps if fps_min <= raw_fps <= fps_max else fps_fallback
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    has_sane_frame_count = 0 < total_frames < 10_000_000
    duration = _safe_ratio(total_frames, native_fps, 0.0) if has_sane_frame_count else 0.0
    warmup_ms = int(round(max(0.0, proctor_warmup_sec) * 1000.0))
    analysis_end_ms: int | None = None
    if duration > 0 and proctor_cooldown_sec > 0 and duration > proctor_cooldown_sec:
        analysis_end_ms = int(round((duration - proctor_cooldown_sec) * 1000.0))

    frame_step = max(1, int(round(native_fps / max(cfg.sample_fps, 0.1))))
    sample_fps = _safe_ratio(native_fps, frame_step, 1.0)

    landmarker = _create_face_landmarker_video()
    use_seek = has_sane_frame_count and total_frames >= frame_step

    sampled = 0
    analyzed_frames = 0
    missing_face_frames = 0
    offscreen_frames = 0
    improper_head_frames = 0
    pose_counts = {"FRONTAL": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "NO_FACE": 0}
    offscreen_indices: list[int] = []
    eye_indices_by_direction: dict[str, list[int]] = {"LEFT": [], "RIGHT": [], "UP": [], "DOWN": []}
    improper_indices: list[int] = []
    head_indices_by_direction: dict[str, list[int]] = {"LEFT": [], "RIGHT": [], "UP": [], "DOWN": [], "NO_FACE": []}
    last_ts_ms = 0
    eye_direction_sequence: list[str] = []
    eye_direction_counts = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "CENTER": 0}
    valid_eye_frames = 0
    unknown_eye_frames = 0
    blink_eye_frames = 0
    previous_iris_x: float | None = None
    previous_iris_y: float | None = None
    previous_eye_dir = "CENTER"
    steady_hold_used = 0
    vertical_eye_bias = 0.0
    iris_x_stats = _OnlineStats()
    iris_y_stats = _OnlineStats()
    previous_head_pose = "FRONTAL"
    head_hold_used = 0

    # Continuous (consecutive) sustained gaze tracking
    run_dir: str | None = None
    run_start_idx: int | None = None
    run_len = 0
    last_counted_idx: int | None = None
    sustained_best = {"direction": None, "startIdx": None, "endIdx": None, "frameCount": 0}

    def process_one_frame(frame_bgr: Any, current_ts_ms: int) -> None:
        nonlocal sampled, missing_face_frames, offscreen_frames, improper_head_frames
        nonlocal last_ts_ms, valid_eye_frames, unknown_eye_frames, blink_eye_frames
        nonlocal previous_iris_x, previous_iris_y, vertical_eye_bias, previous_eye_dir, steady_hold_used
        nonlocal previous_head_pose, head_hold_used
        nonlocal analyzed_frames
        nonlocal run_dir, run_start_idx, run_len, last_counted_idx, sustained_best

        sampled += 1
        last_ts_ms = max(last_ts_ms, int(current_ts_ms))
        analysis_enabled = current_ts_ms >= warmup_ms and (
            analysis_end_ms is None or current_ts_ms <= analysis_end_ms
        )
        small = _maybe_downscale_bgr(frame_bgr, max_frame_w)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, current_ts_ms)

        if not res.face_landmarks:
            missing_face_frames += 1
            pose_counts["NO_FACE"] += 1
            offscreen_frames += 1
            improper_head_frames += 1
            offscreen_indices.append(sampled - 1)
            improper_indices.append(sampled - 1)
            head_indices_by_direction["NO_FACE"].append(sampled - 1)
            return

        lm = res.face_landmarks[0]
        if len(lm) <= CHIN:
            missing_face_frames += 1
            pose_counts["NO_FACE"] += 1
            offscreen_frames += 1
            improper_head_frames += 1
            offscreen_indices.append(sampled - 1)
            improper_indices.append(sampled - 1)
            head_indices_by_direction["NO_FACE"].append(sampled - 1)
            return

        if analysis_enabled:
            analyzed_frames += 1

        # ---- Head pose (prefer MediaPipe transform, fallback to landmark ratios) ----
        raw_head_pose = "FRONTAL"
        if head_pose_use_transform:
            matrix = _extract_face_transform_matrix(res)
            angles = _euler_from_transform_matrix(matrix) if matrix else None
            if angles:
                yaw_deg, pitch_deg, _roll_deg = angles
                if yaw_deg <= head_yaw_left_deg:
                    raw_head_pose = "LEFT"
                elif yaw_deg >= head_yaw_right_deg:
                    raw_head_pose = "RIGHT"
                elif pitch_deg <= head_pitch_up_deg:
                    raw_head_pose = "UP"
                elif pitch_deg >= head_pitch_down_deg:
                    raw_head_pose = "DOWN"
                else:
                    raw_head_pose = "FRONTAL"
            else:
                raw_head_pose = _classify_head_pose(
                    lm,
                    yaw_left_threshold=head_yaw_left_threshold,
                    yaw_right_threshold=head_yaw_right_threshold,
                    pitch_up_threshold=head_pitch_up_threshold,
                    pitch_down_threshold=head_pitch_down_threshold,
                )
        else:
            raw_head_pose = _classify_head_pose(
                lm,
                yaw_left_threshold=head_yaw_left_threshold,
                yaw_right_threshold=head_yaw_right_threshold,
                pitch_up_threshold=head_pitch_up_threshold,
                pitch_down_threshold=head_pitch_down_threshold,
            )
        if raw_head_pose != "FRONTAL":
            head_pose = raw_head_pose
            head_hold_used = 0
        elif previous_head_pose != "FRONTAL" and head_hold_used < max(0, head_pose_hold_frames):
            # Debounce small jitter back to FRONTAL so slight movements don't over-count.
            head_pose = previous_head_pose
            head_hold_used += 1
        else:
            head_pose = "FRONTAL"
            head_hold_used = 0
        previous_head_pose = head_pose
        if analysis_enabled:
            pose_counts[head_pose] += 1
        eye_dir = "CENTER"
        eye_frame_valid = False
        if len(lm) > max(RIGHT_IRIS_CENTER, RIGHT_LOWER_EYELID):
            iris = _iris_metrics(lm)
            if iris["lid_span"] < iris_blink_lid_span_min:
                blink_eye_frames += 1
                eye_frame_valid = False
            else:
                if head_pose == "FRONTAL":
                    clamped_y = max(-eye_vertical_baseline_clamp, min(eye_vertical_baseline_clamp, iris["y_norm"]))
                    vertical_eye_bias = (
                        ((1.0 - eye_vertical_baseline_alpha) * vertical_eye_bias)
                        + (eye_vertical_baseline_alpha * clamped_y)
                    )
                    if eye_adaptive_enabled:
                        # Update adaptive baseline from only CENTER-like frames to avoid biasing
                        # the stats when the user is actually looking up/down/left/right.
                        tentative_dir = _classify_black_dot_direction(
                            iris["x_norm"],
                            iris["y_norm"],
                            horizontal_center_band=eye_horizontal_center_band,
                            horizontal_bias=0.0,
                            vertical_up_band=eye_vertical_up_band,
                            vertical_down_band=eye_vertical_down_band,
                            vertical_bias=vertical_eye_bias,
                        )
                        if tentative_dir == "CENTER":
                            iris_x_stats.add(iris["x_norm"])
                            iris_y_stats.add(iris["y_norm"])

                use_adaptive = (
                    eye_adaptive_enabled
                    and head_pose == "FRONTAL"
                    and iris_x_stats.n >= eye_adaptive_min_samples
                    and iris_y_stats.n >= eye_adaptive_min_samples
                )
                if use_adaptive:
                    std_x = max(iris_x_stats.std, eye_adaptive_std_min)
                    std_y = max(iris_y_stats.std, eye_adaptive_std_min)
                    horizontal_band = _clamp(
                        eye_adaptive_k_horizontal * std_x, eye_adaptive_band_min, eye_adaptive_band_max
                    )
                    up_band = _clamp(eye_adaptive_k_up * std_y, eye_adaptive_band_min, eye_adaptive_band_max)
                    down_band = _clamp(eye_adaptive_k_down * std_y, eye_adaptive_band_min, eye_adaptive_band_max)
                    horizontal_bias = iris_x_stats.mean - 0.5
                    vertical_bias = iris_y_stats.mean
                else:
                    horizontal_band = eye_horizontal_center_band
                    up_band = eye_vertical_up_band
                    down_band = eye_vertical_down_band
                    horizontal_bias = 0.0
                    vertical_bias = vertical_eye_bias

                if previous_iris_x is not None and previous_iris_y is not None:
                    raw_dir = _classify_black_dot_direction(
                        iris["x_norm"],
                        iris["y_norm"],
                        horizontal_center_band=horizontal_band,
                        horizontal_bias=horizontal_bias,
                        vertical_up_band=up_band,
                        vertical_down_band=down_band,
                        vertical_bias=vertical_bias,
                    )
                    if (
                        abs(iris["x_norm"] - previous_iris_x) < iris_jitter_delta_min
                        and abs(iris["y_norm"] - previous_iris_y) < iris_jitter_delta_min
                    ):
                        if raw_dir != "CENTER":
                            eye_dir = raw_dir
                            steady_hold_used = 0
                        elif (
                            previous_eye_dir != "CENTER"
                            and steady_hold_used < max(0, eye_steady_direction_hold_frames)
                        ):
                            # Keep a short hold for stable gaze to avoid undercounting sustained looks.
                            eye_dir = previous_eye_dir
                            steady_hold_used += 1
                        else:
                            eye_dir = "CENTER"
                            steady_hold_used = 0
                    else:
                        eye_dir = raw_dir
                        steady_hold_used = 0
                else:
                    eye_dir = _classify_black_dot_direction(
                        iris["x_norm"],
                        iris["y_norm"],
                        horizontal_center_band=horizontal_band,
                        horizontal_bias=horizontal_bias,
                        vertical_up_band=up_band,
                        vertical_down_band=down_band,
                        vertical_bias=vertical_bias,
                    )
                previous_iris_x = iris["x_norm"]
                previous_iris_y = iris["y_norm"]
                previous_eye_dir = eye_dir
                eye_frame_valid = True
        else:
            unknown_eye_frames += 1
            eye_frame_valid = False

        if eye_frame_valid:
            valid_eye_frames += 1
        else:
            eye_dir = "CENTER"
        should_count_eye = (
            analysis_enabled
            and eye_frame_valid
            and (head_pose == "FRONTAL" or not eye_count_only_when_frontal)
        )
        if should_count_eye:
            eye_direction_sequence.append(eye_dir)
            if eye_dir in eye_direction_counts:
                eye_direction_counts[eye_dir] += 1
            if eye_dir != "CENTER":
                offscreen_frames += 1
                offscreen_indices.append(sampled - 1)
                if eye_dir in eye_indices_by_direction:
                    eye_indices_by_direction[eye_dir].append(sampled - 1)

            # Track sustained consecutive non-center gaze (> threshold seconds)
            current_idx = sampled - 1
            is_consecutive = last_counted_idx is not None and current_idx == (last_counted_idx + 1)
            last_counted_idx = current_idx
            if eye_dir == "CENTER":
                if run_len > sustained_best["frameCount"]:
                    sustained_best = {
                        "direction": run_dir,
                        "startIdx": run_start_idx,
                        "endIdx": (current_idx - 1) if current_idx is not None else None,
                        "frameCount": run_len,
                    }
                run_dir = None
                run_start_idx = None
                run_len = 0
            else:
                if is_consecutive and run_dir == eye_dir:
                    run_len += 1
                else:
                    if run_len > sustained_best["frameCount"]:
                        sustained_best = {
                            "direction": run_dir,
                            "startIdx": run_start_idx,
                            "endIdx": (current_idx - 1) if current_idx is not None else None,
                            "frameCount": run_len,
                        }
                    run_dir = eye_dir
                    run_start_idx = current_idx
                    run_len = 1
        else:
            # Break any ongoing run when we are not counting eyes (warmup, non-frontal gating, etc.)
            if run_len > sustained_best["frameCount"]:
                sustained_best = {
                    "direction": run_dir,
                    "startIdx": run_start_idx,
                    "endIdx": (sampled - 2) if sampled >= 2 else None,
                    "frameCount": run_len,
                }
            run_dir = None
            run_start_idx = None
            run_len = 0
            last_counted_idx = None
        if head_pose != "FRONTAL":
            improper_head_frames += 1
            improper_indices.append(sampled - 1)
            if head_pose in head_indices_by_direction:
                head_indices_by_direction[head_pose].append(sampled - 1)

    try:
        if use_seek:
            frame_ms = 1000.0 / max(native_fps, 1e-3)
            for frame_idx in range(0, total_frames, frame_step):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                current_ts = int(round(frame_idx * frame_ms))
                process_one_frame(frame, current_ts)
        else:
            frame_ms = max(1, int(round(1000.0 / max(native_fps, 1e-3))))
            video_timestamp_ms = 0
            frame_idx = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                current_ts = video_timestamp_ms
                video_timestamp_ms += frame_ms
                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue
                process_one_frame(frame, current_ts)
                frame_idx += 1
    finally:
        capture.release()
        landmarker.close()

    if duration <= 0:
        duration = round(last_ts_ms / 1000.0, 2)

    sampled = max(sampled, 1)
    offscreen_ratio = _safe_ratio(offscreen_frames, max(valid_eye_frames, 1), 0.0)
    analyzed_frames = max(analyzed_frames, 1)
    improper_head_ratio = improper_head_frames / analyzed_frames

    horizontal_scan_cycles = _count_scan_cycles(eye_direction_sequence, ("LEFT", "RIGHT", "CENTER")) + _count_scan_cycles(
        eye_direction_sequence, ("RIGHT", "LEFT", "CENTER")
    )
    vertical_scan_cycles = _count_scan_cycles(eye_direction_sequence, ("UP", "DOWN", "CENTER")) + _count_scan_cycles(
        eye_direction_sequence, ("DOWN", "UP", "CENTER")
    )
    left_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "LEFT")
    right_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "RIGHT")
    up_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "UP")
    down_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "DOWN")
    dominant_oscillation_units = {
        "LEFT": _count_repetitive_center_oscillation(eye_direction_sequence, "LEFT"),
        "RIGHT": _count_repetitive_center_oscillation(eye_direction_sequence, "RIGHT"),
        "UP": _count_repetitive_center_oscillation(eye_direction_sequence, "UP"),
        "DOWN": _count_repetitive_center_oscillation(eye_direction_sequence, "DOWN"),
    }
    repetitive_pattern_direction = max(dominant_oscillation_units, key=dominant_oscillation_units.get)
    directional_center_cycles_total = (
        left_center_cycles + right_center_cycles + up_center_cycles + down_center_cycles
    )
    repetitive_pattern_count = dominant_oscillation_units[repetitive_pattern_direction]
    total_scan_cycles = horizontal_scan_cycles + vertical_scan_cycles + directional_center_cycles_total

    reading_pattern_score = min(
        1.0,
        (offscreen_ratio * 0.45)
        + (min(repetitive_pattern_count, 10) / 10.0 * 0.35)
        + (min(repetitive_pattern_count, 10) / 10.0 * 0.20),
    )
    eye_reliability_score = _safe_ratio(valid_eye_frames, analyzed_frames, 0.0)
    eye_reliable = eye_reliability_score >= eye_reliable_min
    if sampled < 10 or not eye_reliable:
        pattern_verdict = "INCONCLUSIVE"
    elif reading_pattern_score >= 0.6:
        pattern_verdict = "READING_LIKE"
    elif reading_pattern_score >= 0.35:
        pattern_verdict = "MIXED"
    else:
        pattern_verdict = "NATURAL"

    eye_suspicious = eye_reliable and (
        offscreen_ratio >= cfg.offscreen_ratio_threshold
        or repetitive_pattern_count >= cfg.repetitive_pattern_threshold
        or total_scan_cycles >= eye_scan_cycles_threshold
    )
    suspicious = eye_suspicious or (improper_head_ratio >= cfg.improper_head_ratio_threshold)

    # Finalize sustained gaze best (in case video ends on non-center)
    if run_len > sustained_best["frameCount"]:
        sustained_best = {
            "direction": run_dir,
            "startIdx": run_start_idx,
            "endIdx": (sampled - 1),
            "frameCount": run_len,
        }
    sustained_direction = sustained_best.get("direction")
    sustained_frame_count = int(sustained_best.get("frameCount") or 0)
    sustained_duration_sec = round(_safe_ratio(sustained_frame_count, sample_fps, 0.0), 2)
    sustained_start_idx = sustained_best.get("startIdx")
    sustained_end_idx = sustained_best.get("endIdx")
    sustained_gaze = {
        "direction": sustained_direction,
        "durationSec": sustained_duration_sec,
        "startSec": round(_safe_ratio(float(sustained_start_idx or 0), sample_fps, 0.0), 2)
        if sustained_start_idx is not None
        else None,
        "endSec": round(_safe_ratio(float(sustained_end_idx or 0), sample_fps, 0.0), 2)
        if sustained_end_idx is not None
        else None,
        "frameCount": sustained_frame_count,
        "thresholdSec": sustained_gaze_sec_threshold,
        "detected": bool(sustained_direction) and (sustained_duration_sec >= sustained_gaze_sec_threshold),
    }

    return {
        "videoMeta": {
            "durationSec": round(duration, 2),
            "nativeFps": round(native_fps, 2),
            "sampleFps": round(sample_fps, 2),
            "sampledFrames": sampled,
        },
        "eyeMovement": {
            "offScreenFrameCount": offscreen_frames,
            "offScreenRatio": round(offscreen_ratio, 4),
            "missingFaceFrames": missing_face_frames,
            "validEyeFrames": valid_eye_frames,
            "unknownEyeFrames": unknown_eye_frames,
            "blinkEyeFrames": blink_eye_frames,
            "repetitivePatternCount": repetitive_pattern_count,
            "repetitivePatternDirection": repetitive_pattern_direction,
            "repetitivePatternDetected": repetitive_pattern_count >= cfg.repetitive_pattern_threshold,
            "directionCounts": eye_direction_counts,
            "scanCycles": {
                "horizontalLeftRightCenter": horizontal_scan_cycles,
                "verticalUpDownCenter": vertical_scan_cycles,
                "leftCenter": left_center_cycles,
                "rightCenter": right_center_cycles,
                "upCenter": up_center_cycles,
                "downCenter": down_center_cycles,
                "leftCenterOscillationUnits": dominant_oscillation_units["LEFT"],
                "rightCenterOscillationUnits": dominant_oscillation_units["RIGHT"],
                "upCenterOscillationUnits": dominant_oscillation_units["UP"],
                "downCenterOscillationUnits": dominant_oscillation_units["DOWN"],
                "dominantDirection": repetitive_pattern_direction,
                "dominantCount": repetitive_pattern_count,
                "directionToCenterTotal": directional_center_cycles_total,
                "total": total_scan_cycles,
            },
            "readingPatternScore": round(reading_pattern_score, 4),
            "patternVerdict": pattern_verdict,
            "sustainedGaze": sustained_gaze,
            "eyeTracking": {
                "reliabilityScore": round(eye_reliability_score, 4),
                "reliable": eye_reliable,
                "qualityBand": (
                    "strong"
                    if eye_reliability_score >= eye_reliable_strong
                    else "usable"
                    if eye_reliable
                    else "poor"
                ),
                "skipReason": None
                if eye_reliable
                else "low_visible_eye_frames_or_occlusion_glare_blinks",
            },
            "segments": (
                _segments_with_direction(eye_indices_by_direction["LEFT"], sample_fps, "LEFT")
                + _segments_with_direction(eye_indices_by_direction["RIGHT"], sample_fps, "RIGHT")
                + _segments_with_direction(eye_indices_by_direction["UP"], sample_fps, "UP")
                + _segments_with_direction(eye_indices_by_direction["DOWN"], sample_fps, "DOWN")
            ),
        },
        "headPose": {
            "counts": pose_counts,
            "improperHeadFrameCount": improper_head_frames,
            "improperHeadRatio": round(improper_head_ratio, 4),
            "segments": (
                _segments_with_direction(head_indices_by_direction["LEFT"], sample_fps, "LEFT")
                + _segments_with_direction(head_indices_by_direction["RIGHT"], sample_fps, "RIGHT")
                + _segments_with_direction(head_indices_by_direction["UP"], sample_fps, "UP")
                + _segments_with_direction(head_indices_by_direction["DOWN"], sample_fps, "DOWN")
                + _segments_with_direction(head_indices_by_direction["NO_FACE"], sample_fps, "NO_FACE")
            ),
        },
        "summary": {
            "suspicious": suspicious,
            "rules": {
                "warmupSec": proctor_warmup_sec,
                "cooldownSec": proctor_cooldown_sec,
                "sustainedGazeSec": sustained_gaze_sec_threshold,
                "offscreenRatioThreshold": cfg.offscreen_ratio_threshold,
                "improperHeadRatioThreshold": cfg.improper_head_ratio_threshold,
                "repetitivePatternThreshold": cfg.repetitive_pattern_threshold,
                "eyeReliabilityMin": eye_reliable_min,
                "eyeScanCyclesThreshold": eye_scan_cycles_threshold,
            },
        },
    }


class ProctorService:
    """Thin façade over :func:`analyze_eye_head_pose` for DI friendliness."""

    def analyze(self, video_path: Path, thresholds: ProctorThresholds | None = None) -> dict[str, Any]:
        return analyze_eye_head_pose(video_path, thresholds)


proctor_service = ProctorService()
