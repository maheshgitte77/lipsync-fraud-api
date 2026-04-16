# Proctor Signals Field Guide

This document explains how each important field is calculated for:

- `POST /analyze/proctor-signals`
- `POST /analyze`

It should be updated whenever calculation logic changes in:

- `app/proctor_signals.py`
- `app/main.py`
- `app/mediapipe_lipsync.py`

---

## 1) Request Parameters (`/analyze/proctor-signals`)

- `videoUrl`: public/signed URL to video file.
- `questionId`, `candidateId`: passthrough metadata.
- `sampleFps`: target sampling rate for eye/head analysis. Example: `1` means ~1 analyzed frame per second.
- `offscreenRatioThreshold`: threshold for eye off-screen ratio flag.
- `improperHeadRatioThreshold`: threshold for non-frontal head flag.
- `repetitivePatternThreshold`: threshold for repetitive eye alternation count.
- `skipSyncNet`: if `true`, skip SyncNet for this call.

---

## 2) Top-level Response

- `jobId`: generated request ID.
- `videoUrl`, `questionId`, `candidateId`: passthrough values.
- `lipSync`: lip-sync subsystem output.
- `eyeMovement`: eye movement + reading-pattern metrics.
- `headPose`: head-pose metrics.
- `videoMeta`: runtime metadata used in calculations.
- `integrityAnalysis`: final flags for this endpoint.
- `summary`: final suspicious decision and active threshold rules.

---

## 3) `videoMeta` Fields

- `nativeFps`
  - FPS read from video metadata, sanitized to range `[PROCTOR_FPS_MIN, PROCTOR_FPS_MAX]`.
  - Fallback: `PROCTOR_FPS_FALLBACK`.
- `sampleFps`
  - Computed from frame step.
  - Approx formula: `nativeFps / frameStep`.
- `sampledFrames`
  - Number of processed samples (not total raw frames).
- `durationSec`
  - Primary: `totalFrames / nativeFps`.
  - Fallback: last processed timestamp when metadata is bad.
- `trim` (optional, merged by API when env trim is used)
  - `trimEnabled`, `trimMaxSeconds`, `trimApplied`, `sourceDurationSec`, `analyzedDurationSec` from ffprobe + optional first-`N`s clip (`LIPSYNC_VIDEO_TRIM`, `LIPSYNC_TRIM_MAX_SECONDS`). When `trimApplied` is true, eye/head and lip-sync used only the trimmed file.

---

## 4) Lip Sync (`lipSync`)

### 4.1 Fusion

- `fusion.mode`: selected decision mode (`all`, `any`, `best`, `syncnet_only`, `mediapipe_only`).
- `fusion.syncNetSkipped`: true when SyncNet is disabled per request/env.
- `fusion.mediapipeLipSyncSkipped`: true when MediaPipe lip-sync is disabled.
- `fusion.flagSource`: which source is allowed to trigger `LIP_SYNC_MISMATCH`.

### 4.2 SyncNet

- `syncnet.scores.min_dist`: lower is better.
- `syncnet.scores.confidence`: higher is better.
- `syncnet.passed`:
  - pass if `min_dist <= MIN_DIST_PASS` and `confidence >= CONFIDENCE_PASS`.

### 4.3 MediaPipe lip-sync

- Uses lip openness vs audio energy correlation.
- Can be skipped by config.

### 4.4 Lip-sync flag generation (`integrityAnalysis.flags`)

`LIP_SYNC_MISMATCH` depends on `PROCTOR_LIPSYNC_FLAG_SOURCE`:

- `syncnet_only` (recommended)
- `fused`
- `mediapipe_only`
- `none`

---

## 5) Eye Movement (`eyeMovement`)

Eye tracking uses iris center ("black dot") landmarks with quality gating.

### 5.1 Quality counters

- `validEyeFrames`: frames with usable iris data.
- `unknownEyeFrames`: frames where eye/iris landmarks unavailable.
- `blinkEyeFrames`: frames excluded due to blink-like low eyelid span.

### 5.2 Direction counters

- `directionCounts.LEFT/RIGHT/UP/DOWN/CENTER`
  - computed only from valid eye frames.

### 5.3 Off-screen metrics

- `offScreenFrameCount`: valid frames where direction != `CENTER`.
- `offScreenRatio`:
  - formula: `offScreenFrameCount / max(validEyeFrames, 1)`.

### 5.4 Repetitive pattern

- `repetitivePatternCount`:
  - strict oscillation-unit count from one dominant direction:
  - unit definition: `DIR -> CENTER -> DIR -> CENTER` (non-overlapping).
- `repetitivePatternDirection`:
  - direction with highest strict oscillation-unit count.
- `repetitivePatternDetected`:
  - `repetitivePatternCount >= repetitivePatternThreshold`.

### 5.5 Scan cycles

- `horizontalLeftRightCenter`: cycles matching `LEFT->RIGHT->CENTER` or reverse.
- `verticalUpDownCenter`: cycles matching `UP->DOWN->CENTER` or reverse.
- `leftCenter`, `rightCenter`, `upCenter`, `downCenter`:
  - counts of `DIR->CENTER` transitions after sequence compression.
- `leftCenterOscillationUnits`, `rightCenterOscillationUnits`, `upCenterOscillationUnits`, `downCenterOscillationUnits`:
  - strict `DIR->CENTER->DIR->CENTER` unit counts per direction.
- `dominantDirection`, `dominantCount`:
  - direction with highest strict oscillation-unit count and its value.
- `directionToCenterTotal`:
  - `leftCenter + rightCenter + upCenter + downCenter`.
- `total`:
  - `horizontalLeftRightCenter + verticalUpDownCenter + directionToCenterTotal`.

### 5.6 Reading pattern score

- `readingPatternScore` in `[0,1]`:
  - `0.45 * offScreenRatio`
  - `+ 0.35 * min(repetitivePatternCount,10)/10`
  - `+ 0.20 * min(repetitivePatternCount,10)/10`

- `patternVerdict`:
  - `INCONCLUSIVE` if sampled too low or eye reliability low
  - `READING_LIKE` if score >= `0.6`
  - `MIXED` if score >= `0.35`
  - `NATURAL` otherwise

### 5.7 Reliability block (`eyeTracking`)

- `reliabilityScore = validEyeFrames / sampledFrames`
- `reliable = reliabilityScore >= eyeReliabilityMin`
- `qualityBand`:
  - `strong` if >= `EYE_RELIABILITY_STRONG`
  - `usable` if >= `EYE_RELIABILITY_MIN`
  - `poor` otherwise
- `skipReason`: populated when not reliable.

### 5.8 Eye flag generation

`OFF_SCREEN_GAZE` is generated only when eye tracking is reliable and:

- `offScreenRatio >= offscreenRatioThreshold`, OR
- `repetitivePatternDetected == true`.

`SUBTLE_READING` is generated when:

- eye tracking reliable, AND
- `patternVerdict == READING_LIKE`.

---

## 6) Head Pose (`headPose`)

- `counts.FRONTAL/LEFT/RIGHT/UP/DOWN/NO_FACE`
- `improperHeadFrameCount`: frames where pose != `FRONTAL`.
- `improperHeadRatio`:
  - `improperHeadFrameCount / sampledFrames`.
- `segments`: grouped time ranges where improper pose occurred.

Flag generated:

- `READING_FROM_EXTERNAL` when `improperHeadRatio >= improperHeadRatioThreshold`.

---

## 7) Segments Format

Segment object:

- `startSec`, `endSec`: numeric time bounds.
- `range`: human-readable `MM:SS-MM:SS`.
- `frameCount`: sampled-frame count in that segment.

---

## 8) Final Decision (`integrityAnalysis` and `summary`)

- `integrityAnalysis.flags`: list of generated flags.
- `integrityAnalysis.verdict`:
  - `SUSPECT` if at least one flag is generated.
  - `CLEAR` otherwise.
- `summary.suspicious`: boolean mirror of flag presence.
- `summary.signalCount`: `len(flags)`.
- `summary.rules`: effective thresholds used for this request.

---

## 9) Worked Example (from sample)

Given:

- `offScreenFrameCount = 48`
- `validEyeFrames = 135`
- `offscreenRatioThreshold = 0.10`

Calculation:

- `offScreenRatio = 48 / 135 = 0.3556` (35.56%)

Decision:

- `0.3556 >= 0.10` => `OFF_SCREEN_GAZE` flag generated.

---

## 10) Key Env Variables

- Lip-sync:
  - `LIPSYNC_FUSION`
  - `LIPSYNC_DISABLE_MEDIAPIPE_LIPSYNC`
  - `PROCTOR_LIPSYNC_FLAG_SOURCE`
  - `MIN_DIST_PASS`
  - `CONFIDENCE_PASS`

- Eye/head:
  - `PROCTOR_SAMPLE_FPS`
  - `PROCTOR_OFFSCREEN_RATIO_THRESHOLD`
  - `PROCTOR_IMPROPER_HEAD_RATIO_THRESHOLD`
  - `PROCTOR_REPETITIVE_PATTERN_THRESHOLD`
  - `PROCTOR_FPS_FALLBACK`
  - `PROCTOR_FPS_MIN`
  - `PROCTOR_FPS_MAX`
  - `EYE_IRIS_BLINK_LID_SPAN_MIN`
  - `EYE_IRIS_JITTER_DELTA_MIN`
  - `EYE_RELIABILITY_MIN`
  - `EYE_RELIABILITY_STRONG`
  - `EYE_SCAN_CYCLES_THRESHOLD`

---

## 11) Maintenance Rule

If you change any formula, threshold semantics, or flag condition in code, update this file in the same commit.



Implemented rules:

LIP_SYNC_MISMATCH if lipSync.syncnet.passed == false
READING_FROM_EXTERNAL if reliable and:
(offScreenRatio >= offscreenRatioThreshold AND repetitivePatternCount >= repetitivePatternThreshold)
OR readingPatternScore >= 0.60
EYE_MOVEMENT if (LEFT+RIGHT+UP+DOWN)/validEyeFrames >= 0.5
IMPROPER_HEAD_POSE if improperHeadRatio >= improperHeadRatioThreshold