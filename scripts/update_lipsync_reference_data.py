#!/usr/bin/env python3
"""
Refresh testdata/lipsync_rnd_reference.json

  python scripts/update_lipsync_reference_data.py --recompute-sweeps
      Recompute threshold grid from embedded baselines (fast).

  python scripts/update_lipsync_reference_data.py --measure
      Re-run MediaPipe + SyncNet on each clip; update baselines (slow).

Paths are relative to project root (parent of scripts/).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "testdata" / "lipsync_rnd_reference.json"

SWEEP_MIN_DIST = [5.5, 6.0, 6.25, 6.5, 6.575, 6.6, 6.65, 7.0, 7.5, 8.0, 8.5, 9.0]
SWEEP_MP_CORR = [0.20, 0.22, 0.25, 0.28, 0.29, 0.30, 0.35, 0.40]


def _load() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def syncnet_pass(sn: dict, min_dist_max: float, conf_min: float) -> bool:
    return sn["min_dist"] <= min_dist_max and sn["confidence"] >= conf_min


def mediapipe_pass(mp: dict, corr_min: float) -> bool:
    return mp["correlation"] >= corr_min


def fusion_all(sn_ok: bool, mp_ok: bool) -> bool:
    return sn_ok and mp_ok


def recompute_sweeps(data: dict) -> None:
    conf = float(data["threshold_sweep"].get("confidence_pass_fixed", 3.0))
    clips = data["clips"]
    grid = []
    for md in SWEEP_MIN_DIST:
        for mc in SWEEP_MP_CORR:
            row: dict = {
                "MIN_DIST_PASS": md,
                "MEDIAPIPE_CORR_THRESHOLD": mc,
                "CONFIDENCE_PASS": conf,
                "clips": {},
            }
            for c in clips:
                cid = c["id"]
                sn = c["baseline"]["syncnet"]
                mp = c["baseline"]["mediapipe"]
                s_ok = syncnet_pass(sn, md, conf)
                m_ok = mediapipe_pass(mp, mc)
                row["clips"][cid] = {
                    "syncnet_pass": s_ok,
                    "mediapipe_pass": m_ok,
                    "fusion_all_pass": fusion_all(s_ok, m_ok),
                }
            grid.append(row)

    data["threshold_sweep"]["grid"] = grid
    data["threshold_sweep"]["summary"] = (
        f"Grid {len(SWEEP_MIN_DIST)} x {len(SWEEP_MP_CORR)}; "
        f"CONFIDENCE_PASS={conf}; fusion=all"
    )

    hits = [
        r
        for r in grid
        if r["clips"].get("vgg_example_avi", {}).get("fusion_all_pass")
        and not r["clips"].get("videoplayback_mp4", {}).get("fusion_all_pass")
    ]
    data["threshold_sweep"]["presets_pass_vgg_only_all"] = hits


def measure_clips(data: dict) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    from app.mediapipe_lipsync import analyze_mediapipe_correlation
    from app import main as api

    for c in data["clips"]:
        rel = c["path"]
        vpath = (ROOT / rel).resolve()
        if not vpath.is_file():
            print(f"Skip missing file: {vpath}")
            continue

        mp = analyze_mediapipe_correlation(
            vpath,
            corr_threshold=0.0,
            corr_borderline=-1.0,
        )
        if mp.get("error"):
            print(f"MediaPipe error {c['id']}: {mp['error']}")
            mp_base = {"error": mp["error"]}
        else:
            mp_base = {
                "correlation": mp["scores"]["correlation"],
                "frames_processed": mp["scores"]["frames_processed"],
                "fps_used": mp["scores"]["fps_used"],
                "mediapipe_backend": mp["scores"].get("mediapipe_backend", ""),
            }

        job_id = uuid.uuid4().hex[:12]
        ref = f"rnd_{job_id}"
        job_dir = api.TEMP_DIR / ref
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw = api.run_syncnet(vpath, ref, job_dir)
            raw.pop("raw_log_tail", None)
            sn_base = {
                "min_dist": raw["min_dist"],
                "confidence": raw["confidence"],
                "av_offset_frames": raw["av_offset_frames"],
                "tracks_evaluated": raw["tracks_evaluated"],
            }
        except Exception as e:
            print(f"SyncNet error {c['id']}: {e}")
            sn_base = {"error": str(e)[:500]}
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

        c["baseline"] = {
            "source": "scripts/update_lipsync_reference_data.py --measure",
            "syncnet": sn_base,
            "mediapipe": mp_base,
        }
        print(f"Updated {c['id']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recompute-sweeps",
        action="store_true",
        help="Fill threshold_sweep.grid from current baselines",
    )
    ap.add_argument(
        "--measure",
        action="store_true",
        help="Re-run pipelines and refresh clip baselines (slow)",
    )
    args = ap.parse_args()

    if not args.recompute_sweeps and not args.measure:
        ap.print_help()
        return 1

    data = _load()
    if args.measure:
        measure_clips(data)
    if args.recompute_sweeps or args.measure:
        recompute_sweeps(data)
    _save(data)
    print(f"Wrote {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
