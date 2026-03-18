#!/usr/bin/env python3
"""Strict KernelBench evaluator wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strict_benchmark import (
    evaluate_kernelbench_task_strict,
    get_original_model_source,
    load_model_source,
)


def evaluate_kernelbench_task(
    task_id: str,
    solution_path: str,
    device: str = "cuda",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    measure_performance: bool = True,
    verbose: bool = True,
    build_dir: str | None = None,
    baseline_path: str | None = None,
    reference_baseline: str | None = None,
) -> dict:
    return evaluate_kernelbench_task_strict(
        task_id=task_id,
        solution_path=solution_path,
        device=device,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        measure_performance=measure_performance,
        verbose=verbose,
        build_dir=build_dir,
        baseline_path=baseline_path,
        reference_baseline=reference_baseline,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict KernelBench evaluation")
    parser.add_argument("--task-id", required=True, help="KernelBench task ID like 1_19")
    parser.add_argument("--solution-path", required=True, help="Path to optimized ModelNew source")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--kb-baseline-path", default=None)
    parser.add_argument("--kb-reference-baseline", default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    results = evaluate_kernelbench_task(
        task_id=args.task_id,
        solution_path=args.solution_path,
        device=args.device,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        measure_performance=True,
        verbose=not args.json,
        build_dir=args.build_dir,
        baseline_path=args.kb_baseline_path,
        reference_baseline=args.kb_reference_baseline,
    )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print("\n" + "=" * 50)
        print("KERNELBENCH STRICT EVALUATION")
        print("=" * 50)
        print(f"Compilation: {'✓' if results.get('compiled') else '✗'}")
        print(f"Correctness: {'✓' if results.get('correct') else '✗'}")
        print(f"Speedup: {results.get('speedup', 0.0):.4f}")
        print(f"Baseline: {results.get('baseline_path')}")
        if results.get("error"):
            print(f"Error: {results['error']}")
        print(f"\nspeedup: {results.get('speedup', 0.0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
