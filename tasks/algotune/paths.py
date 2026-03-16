"""Helpers for resolving the vendored AlgoTune runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_VENDOR_ROOT = TASK_DIR / "vendor" / "AlgoTune"


def resolve_algotune_root() -> Path:
    """Resolve the AlgoTune runtime path.

    Resolution order:
    1. `ALGOTUNE_PATH` if set
    2. vendored runtime under `tasks/algotune/vendor/AlgoTune`
    """
    env_path = os.getenv("ALGOTUNE_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_VENDOR_ROOT.resolve()


def resolve_algotune_tasks_dir() -> Path:
    return resolve_algotune_root() / "AlgoTuneTasks"


def ensure_algotune_on_path() -> Path:
    root = resolve_algotune_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
