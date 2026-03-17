#!/usr/bin/env python3
"""Strict AlgoTune evaluation wrapper.

This repo intentionally exposes only the benchmark-faithful AlgoTune path.
Search-time evaluation uses the train split; final reporting uses the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .paths import ensure_algotune_on_path
    from .strict_benchmark import STRICT_MODE_NAME, evaluate_solver_split
except ImportError:
    from paths import ensure_algotune_on_path
    from strict_benchmark import STRICT_MODE_NAME, evaluate_solver_split


ensure_algotune_on_path()


def evaluate_task(
    task_name: str,
    solver_path: str,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a solver using the strict vendored AlgoTune pipeline only."""
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported AlgoTune split: {split}")
    return evaluate_solver_split(
        task_name=task_name,
        solver_path=solver_path,
        split=split,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AlgoTune solver using the strict vendored benchmark pipeline",
    )
    parser.add_argument("--task", required=True, help="AlgoTune task name")
    parser.add_argument("--solution-path", required=True, help="Path to solver.py")
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    args = parser.parse_args()

    results = evaluate_task(
        task_name=args.task,
        solver_path=args.solution_path,
        split=args.split,
    )

    if results.get("error"):
        print(f"error: {results['error']}")
        print("speedup: 0.0")
    elif not results.get("correct", False):
        print("error: Strict AlgoTune evaluation did not produce a valid solution")
        print("speedup: 0.0")
    else:
        print(f"speedup: {results['speedup']:.4f}")


if __name__ == "__main__":
    main()
