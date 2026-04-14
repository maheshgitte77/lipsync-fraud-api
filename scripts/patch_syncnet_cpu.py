#!/usr/bin/env python3
"""
One-time patch so joonson/syncnet_python runs on CPU (no CUDA).

Usage:
  python scripts/patch_syncnet_cpu.py path/to/syncnet_python

Upstream hard-codes .cuda() and S3FD(device='cuda'). This script switches those to CPU.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def patch_syncnet_instance(path: Path) -> bool:
    s = path.read_text(encoding="utf-8", errors="replace")
    if "self.device = torch.device('cpu')" in s and ".cuda()" not in s:
        return False
    orig = s
    s = re.sub(
        r"self\.__S__\s*=\s*S\(num_layers_in_fc_layers\s*=\s*num_layers_in_fc_layers\)\.cuda\(\);",
        "self.device = torch.device('cpu');\n        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).to(self.device);",
        s,
        count=1,
    )
    s = s.replace(".cuda()", ".to(self.device)")
    if s == orig:
        return False
    path.write_text(s, encoding="utf-8")
    return True


def patch_box_utils_numpy2(path: Path) -> bool:
    """NumPy 2.x removed np.int."""
    s = path.read_text(encoding="utf-8", errors="replace")
    old = "return np.array(keep).astype(np.int)"
    new = "return np.array(keep).astype(np.int64)"
    if old not in s:
        return False
    path.write_text(s.replace(old, new, 1), encoding="utf-8")
    return True


def patch_run_pipeline(path: Path) -> bool:
    s = path.read_text(encoding="utf-8", errors="replace")
    if "S3FD(device='cpu')" in s:
        return False
    orig = s
    s = s.replace("S3FD(device='cuda')", "S3FD(device='cpu')")
    if s == orig:
        return False
    path.write_text(s, encoding="utf-8")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python patch_syncnet_cpu.py <syncnet_python_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    si = root / "SyncNetInstance.py"
    rp = root / "run_pipeline.py"
    bu = root / "detectors" / "s3fd" / "box_utils.py"
    if not si.is_file() or not rp.is_file():
        print(f"Expected SyncNet repo at {root} (missing SyncNetInstance.py or run_pipeline.py)")
        return 1

    a = patch_syncnet_instance(si)
    b = patch_run_pipeline(rp)
    c = patch_box_utils_numpy2(bu) if bu.is_file() else False
    if a:
        print(f"Patched {si}")
    if b:
        print(f"Patched {rp}")
    if c:
        print(f"Patched {bu} (NumPy 2.x)")
    if not a and not b and not c:
        print("No changes (already patched or unexpected file contents).")
    else:
        print("Done. CPU mode is slow; a small GPU instance is recommended for production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
