from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from app.mediapipe_lipsync import _resolve_face_landmarker_model


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


def _classify_head_pose(landmarks) -> str:
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]

    eye_center_x = (left_eye.x + right_eye.x) / 2.0
    eye_center_y = (left_eye.y + right_eye.y) / 2.0
    eye_span_x = abs(right_eye.x - left_eye.x)

    yaw = _safe_ratio(nose.x - eye_center_x, eye_span_x, 0.0)
    pitch = _safe_ratio(nose.y - eye_center_y, abs(chin.y - eye_center_y), 0.0)

    if yaw < -0.12:
        return "LEFT"
    if yaw > 0.12:
        return "RIGHT"
    if pitch < 0.15:
        return "UP"
    if pitch > 0.45:
        return "DOWN"
    return "FRONTAL"


def _classify_eye_direction(landmarks) -> str:
    left_outer = landmarks[LEFT_EYE_OUTER]
    left_inner = landmarks[LEFT_EYE_INNER]
    left_iris = landmarks[LEFT_IRIS_CENTER]

    right_inner = landmarks[RIGHT_EYE_INNER]
    right_outer = landmarks[RIGHT_EYE_OUTER]
    right_iris = landmarks[RIGHT_IRIS_CENTER]

    left_ratio = _safe_ratio(
        left_iris.x - left_outer.x,
        max(abs(left_inner.x - left_outer.x), 1e-6),
        0.5,
    )
    right_ratio = _safe_ratio(
        right_iris.x - right_inner.x,
        max(abs(right_outer.x - right_inner.x), 1e-6),
        0.5,
    )
    avg_lr = (left_ratio + right_ratio) / 2.0

    eye_center_y = (
        landmarks[LEFT_EYE_OUTER].y
        + landmarks[LEFT_EYE_INNER].y
        + landmarks[RIGHT_EYE_OUTER].y
        + landmarks[RIGHT_EYE_INNER].y
    ) / 4.0
    iris_center_y = (left_iris.y + right_iris.y) / 2.0
    vertical_delta = iris_center_y - eye_center_y

    if avg_lr < 0.30:
        return "LEFT"
    if avg_lr > 0.70:
        return "RIGHT"
    if vertical_delta < -0.020:
        return "UP"
    if vertical_delta > 0.020:
        return "DOWN"
    return "CENTER"


def _get_iris_tracking_metrics(landmarks) -> dict[str, float]:
    """
    Black-dot tracking metrics from iris landmarks.
    Returns normalized iris position and lid span for blink filtering.
    """
    left_outer = landmarks[LEFT_EYE_OUTER]
    left_inner = landmarks[LEFT_EYE_INNER]
    left_iris = landmarks[LEFT_IRIS_CENTER]
    right_inner = landmarks[RIGHT_EYE_INNER]
    right_outer = landmarks[RIGHT_EYE_OUTER]
    right_iris = landmarks[RIGHT_IRIS_CENTER]

    left_x = _safe_ratio(
        left_iris.x - left_outer.x,
        max(abs(left_inner.x - left_outer.x), 1e-6),
        0.5,
    )
    right_x = _safe_ratio(
        right_iris.x - right_inner.x,
        max(abs(right_outer.x - right_inner.x), 1e-6),
        0.5,
    )
    x_norm = (left_x + right_x) / 2.0

    eye_center_y = (
        landmarks[LEFT_EYE_OUTER].y
        + landmarks[LEFT_EYE_INNER].y
        + landmarks[RIGHT_EYE_OUTER].y
        + landmarks[RIGHT_EYE_INNER].y
    ) / 4.0
    left_lid_span = abs(
        landmarks[LEFT_LOWER_EYELID].y - landmarks[LEFT_UPPER_EYELID].y
    )
    right_lid_span = abs(
        landmarks[RIGHT_LOWER_EYELID].y - landmarks[RIGHT_UPPER_EYELID].y
    )
    avg_lid_span = (left_lid_span + right_lid_span) / 2.0
    y_norm = _safe_ratio(
        ((left_iris.y + right_iris.y) / 2.0) - eye_center_y,
        max(avg_lid_span, 1e-6),
        0.0,
    )
    return {"x_norm": x_norm, "y_norm": y_norm, "lid_span": avg_lid_span}


def _classify_black_dot_direction(
    x_norm: float,
    y_norm: float,
    *,
    horizontal_center_band: float = 0.12,
    vertical_center_band: float = 0.18,
) -> str:
    """
    Direction from iris center only (black-dot tracking).
    """
    if x_norm < (0.5 - horizontal_center_band):
        return "LEFT"
    if x_norm > (0.5 + horizontal_center_band):
        return "RIGHT"
    if y_norm < -vertical_center_band:
        return "UP"
    if y_norm > vertical_center_band:
        return "DOWN"
    return "CENTER"


def _classify_eye_direction_no_iris(lm) -> str:
    """Rough gaze proxy when iris landmarks are absent (468-point topology)."""
    left_outer = lm[LEFT_EYE_OUTER]
    right_outer = lm[RIGHT_EYE_OUTER]
    nose = lm[NOSE_TIP]
    eye_mid_x = (left_outer.x + right_outer.x) / 2.0
    eye_span = max(abs(right_outer.x - left_outer.x), 1e-6)
    yaw_proxy = (nose.x - eye_mid_x) / eye_span
    eye_mid_y = (left_outer.y + right_outer.y) / 2.0
    pitch_proxy = (nose.y - eye_mid_y) / max(abs(lm[CHIN].y - eye_mid_y), 1e-6)

    if yaw_proxy < -0.18:
        return "LEFT"
    if yaw_proxy > 0.18:
        return "RIGHT"
    if pitch_proxy < 0.12:
        return "UP"
    if pitch_proxy > 0.55:
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


def _count_scan_cycles(direction_sequence: list[str], pattern: tuple[str, str, str]) -> int:
    """
    Count ordered scan cycles in a compressed direction sequence.
    Example pattern: ("LEFT", "RIGHT", "CENTER").
    """
    if len(direction_sequence) < 3:
        return 0

    # Compress repeats: LEFT,LEFT,RIGHT,RIGHT,CENTER -> LEFT,RIGHT,CENTER
    compressed: list[str] = []
    for d in direction_sequence:
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


def _count_direction_to_center_cycles(direction_sequence: list[str], direction: str) -> int:
    """
    Count cycles of DIR -> CENTER in a compressed direction sequence.
    Example: LEFT,CENTER,LEFT,CENTER => 2
    """
    if len(direction_sequence) < 2:
        return 0

    compressed: list[str] = []
    for d in direction_sequence:
        if not compressed or compressed[-1] != d:
            compressed.append(d)

    cycles = 0
    for i in range(len(compressed) - 1):
        if compressed[i] == direction and compressed[i + 1] == "CENTER":
            cycles += 1
    return cycles


def _count_repetitive_center_oscillation(
    direction_sequence: list[str], direction: str
) -> int:
    """
    Strict repetitive unit count for:
    DIR -> CENTER -> DIR -> CENTER

    Non-overlapping counting on compressed sequence.
    """
    if len(direction_sequence) < 4:
        return 0

    compressed: list[str] = []
    for d in direction_sequence:
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
    """FaceLandmarker (Tasks API). Use top-level mp.Image — avoids vision.core.image NameError on some builds."""
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    model_path = _resolve_face_landmarker_model()
    base = base_options_lib.BaseOptions(model_asset_path=str(model_path))
    options = FaceLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
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


def analyze_eye_head_pose(video_path: Path, thresholds: ProctorThresholds | None = None) -> dict[str, Any]:
    cfg = thresholds or ProctorThresholds()
    max_frame_w = int(os.environ.get("PROCTOR_MAX_FRAME_WIDTH", "480") or "480")
    fps_fallback = float(os.environ.get("PROCTOR_FPS_FALLBACK", "25.0") or "25.0")
    fps_min = float(os.environ.get("PROCTOR_FPS_MIN", "5.0") or "5.0")
    fps_max = float(os.environ.get("PROCTOR_FPS_MAX", "120.0") or "120.0")
    iris_blink_lid_span_min = float(
        os.environ.get("EYE_IRIS_BLINK_LID_SPAN_MIN", "0.0055") or "0.0055"
    )
    iris_jitter_delta_min = float(
        os.environ.get("EYE_IRIS_JITTER_DELTA_MIN", "0.015") or "0.015"
    )
    eye_reliable_min = float(
        os.environ.get("EYE_RELIABILITY_MIN", "0.4") or "0.4"
    )
    eye_reliable_strong = float(
        os.environ.get("EYE_RELIABILITY_STRONG", "0.7") or "0.7"
    )
    eye_scan_cycles_threshold = int(
        os.environ.get("EYE_SCAN_CYCLES_THRESHOLD", "6") or "6"
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video for eye/head analysis")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    # Some WEBM decoders report absurd FPS (e.g. 1000). Clamp to sane range.
    native_fps = raw_fps if fps_min <= raw_fps <= fps_max else fps_fallback
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    has_sane_frame_count = 0 < total_frames < 10_000_000
    duration = _safe_ratio(total_frames, native_fps, 0.0) if has_sane_frame_count else 0.0

    frame_step = max(1, int(round(native_fps / max(cfg.sample_fps, 0.1))))
    sample_fps = _safe_ratio(native_fps, frame_step, 1.0)

    landmarker = _create_face_landmarker_video()
    use_seek = has_sane_frame_count and total_frames >= frame_step

    sampled = 0
    missing_face_frames = 0
    offscreen_frames = 0
    improper_head_frames = 0
    repetitive_pattern_count = 0
    pose_counts = {"FRONTAL": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "NO_FACE": 0}
    offscreen_indices: list[int] = []
    improper_indices: list[int] = []
    last_ts_ms = 0
    eye_direction_sequence: list[str] = []
    eye_direction_counts = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "CENTER": 0}
    valid_eye_frames = 0
    unknown_eye_frames = 0
    blink_eye_frames = 0
    previous_iris_x = None
    previous_iris_y = None

    def process_one_frame(frame_bgr: Any, current_ts_ms: int) -> None:
        nonlocal sampled, missing_face_frames, offscreen_frames, improper_head_frames
        nonlocal repetitive_pattern_count, last_ts_ms
        nonlocal valid_eye_frames, unknown_eye_frames, blink_eye_frames
        nonlocal previous_iris_x, previous_iris_y

        sampled += 1
        last_ts_ms = max(last_ts_ms, int(current_ts_ms))
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
            return

        lm = res.face_landmarks[0]
        if len(lm) <= CHIN:
            missing_face_frames += 1
            pose_counts["NO_FACE"] += 1
            offscreen_frames += 1
            improper_head_frames += 1
            offscreen_indices.append(sampled - 1)
            improper_indices.append(sampled - 1)
            return

        eye_dir = "CENTER"
        eye_frame_valid = False
        if len(lm) > max(RIGHT_IRIS_CENTER, RIGHT_LOWER_EYELID):
            iris = _get_iris_tracking_metrics(lm)
            if iris["lid_span"] < iris_blink_lid_span_min:
                blink_eye_frames += 1
                eye_frame_valid = False
            else:
                if previous_iris_x is not None and previous_iris_y is not None:
                    if (
                        abs(iris["x_norm"] - previous_iris_x) < iris_jitter_delta_min
                        and abs(iris["y_norm"] - previous_iris_y) < iris_jitter_delta_min
                    ):
                        eye_dir = "CENTER"
                    else:
                        eye_dir = _classify_black_dot_direction(
                            iris["x_norm"], iris["y_norm"]
                        )
                else:
                    eye_dir = _classify_black_dot_direction(
                        iris["x_norm"], iris["y_norm"]
                    )

                previous_iris_x = iris["x_norm"]
                previous_iris_y = iris["y_norm"]
                eye_frame_valid = True
        else:
            unknown_eye_frames += 1
            eye_frame_valid = False

        if eye_frame_valid:
            valid_eye_frames += 1
        else:
            eye_dir = "CENTER"
        head_pose = _classify_head_pose(lm)

        pose_counts[head_pose] += 1
        if eye_frame_valid:
            eye_direction_sequence.append(eye_dir)
            if eye_dir in eye_direction_counts:
                eye_direction_counts[eye_dir] += 1
            if eye_dir != "CENTER":
                offscreen_frames += 1
                offscreen_indices.append(sampled - 1)

        if head_pose != "FRONTAL":
            improper_head_frames += 1
            improper_indices.append(sampled - 1)

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
        # Robust fallback duration from processed timestamps.
        duration = round(last_ts_ms / 1000.0, 2)

    sampled = max(sampled, 1)
    offscreen_ratio = _safe_ratio(offscreen_frames, max(valid_eye_frames, 1), 0.0)
    improper_head_ratio = improper_head_frames / sampled
    horizontal_scan_cycles = _count_scan_cycles(
        eye_direction_sequence, ("LEFT", "RIGHT", "CENTER")
    ) + _count_scan_cycles(eye_direction_sequence, ("RIGHT", "LEFT", "CENTER"))
    vertical_scan_cycles = _count_scan_cycles(
        eye_direction_sequence, ("UP", "DOWN", "CENTER")
    ) + _count_scan_cycles(eye_direction_sequence, ("DOWN", "UP", "CENTER"))
    left_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "LEFT")
    right_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "RIGHT")
    up_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "UP")
    down_center_cycles = _count_direction_to_center_cycles(eye_direction_sequence, "DOWN")
    left_oscillation_units = _count_repetitive_center_oscillation(
        eye_direction_sequence, "LEFT"
    )
    right_oscillation_units = _count_repetitive_center_oscillation(
        eye_direction_sequence, "RIGHT"
    )
    up_oscillation_units = _count_repetitive_center_oscillation(
        eye_direction_sequence, "UP"
    )
    down_oscillation_units = _count_repetitive_center_oscillation(
        eye_direction_sequence, "DOWN"
    )
    dominant_oscillation_units = {
        "LEFT": left_oscillation_units,
        "RIGHT": right_oscillation_units,
        "UP": up_oscillation_units,
        "DOWN": down_oscillation_units,
    }
    repetitive_pattern_direction = max(
        dominant_oscillation_units, key=dominant_oscillation_units.get
    )
    directional_center_cycles_total = (
        left_center_cycles + right_center_cycles + up_center_cycles + down_center_cycles
    )
    # Strict repetitive unit: DIR->CENTER->DIR->CENTER
    repetitive_pattern_count = dominant_oscillation_units[repetitive_pattern_direction]
    total_scan_cycles = (
        horizontal_scan_cycles + vertical_scan_cycles + directional_center_cycles_total
    )

    reading_pattern_score = min(
        1.0,
        (offscreen_ratio * 0.45)
        + (min(repetitive_pattern_count, 10) / 10.0 * 0.35)
        + (min(repetitive_pattern_count, 10) / 10.0 * 0.20),
    )
    eye_reliability_score = _safe_ratio(valid_eye_frames, sampled, 0.0)
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
    suspicious = eye_suspicious or (
        improper_head_ratio >= cfg.improper_head_ratio_threshold
    )

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
                "leftCenterOscillationUnits": left_oscillation_units,
                "rightCenterOscillationUnits": right_oscillation_units,
                "upCenterOscillationUnits": up_oscillation_units,
                "downCenterOscillationUnits": down_oscillation_units,
                "dominantDirection": repetitive_pattern_direction,
                "dominantCount": repetitive_pattern_count,
                "directionToCenterTotal": directional_center_cycles_total,
                "total": total_scan_cycles,
            },
            "readingPatternScore": round(reading_pattern_score, 4),
            "patternVerdict": pattern_verdict,
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
            "segments": _segment_from_indices(offscreen_indices, sample_fps),
        },
        "headPose": {
            "counts": pose_counts,
            "improperHeadFrameCount": improper_head_frames,
            "improperHeadRatio": round(improper_head_ratio, 4),
            "segments": _segment_from_indices(improper_indices, sample_fps),
        },
        "summary": {
            "suspicious": suspicious,
            "rules": {
                "offscreenRatioThreshold": cfg.offscreen_ratio_threshold,
                "improperHeadRatioThreshold": cfg.improper_head_ratio_threshold,
                "repetitivePatternThreshold": cfg.repetitive_pattern_threshold,
                "eyeReliabilityMin": eye_reliable_min,
                "eyeScanCyclesThreshold": eye_scan_cycles_threshold,
            },
        },
    }
