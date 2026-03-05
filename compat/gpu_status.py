#!/usr/bin/env python3
"""Deprecated wrapper: use src/cluster_gpu_status.py."""

import runpy
import sys
from pathlib import Path

print("[DEPRECATED] gpu_status.py is deprecated. Use python src/cluster_gpu_status.py instead.", file=sys.stderr)
repo_root = Path(__file__).resolve().parents[1]
runpy.run_path(str(repo_root / "src" / "cluster_gpu_status.py"), run_name="__main__")
