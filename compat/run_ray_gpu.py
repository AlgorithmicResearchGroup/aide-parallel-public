#!/usr/bin/env python3
"""Deprecated wrapper: use src/aide_runner.py."""

import runpy
import sys
from pathlib import Path

print("[DEPRECATED] run_ray_gpu.py is deprecated. Use python src/aide_runner.py instead.", file=sys.stderr)
repo_root = Path(__file__).resolve().parents[1]
runpy.run_path(str(repo_root / "src" / "aide_runner.py"), run_name="__main__")
