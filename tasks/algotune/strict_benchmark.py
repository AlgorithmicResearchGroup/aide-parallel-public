"""Strict AlgoTune benchmark helpers.

This module is intentionally conservative. It validates that the environment
matches the benchmark contract and uses the vendored AlgoTune evaluation path
for train-search and final-test evaluation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .paths import ensure_algotune_on_path, resolve_algotune_tasks_dir


ALGOTUNE_ROOT = ensure_algotune_on_path()
ALGOTUNE_TASKS_DIR = resolve_algotune_tasks_dir()

STRICT_MODE_NAME = "benchmark_strict"


def _task_module_path(task_name: str) -> Path:
    return ALGOTUNE_TASKS_DIR / task_name / f"{task_name}.py"


def _import_task_module(task_name: str) -> None:
    task_file = _task_module_path(task_name)
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    spec = importlib.util.spec_from_file_location(task_name, task_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for task '{task_name}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def validate_benchmark_environment(*, check_all_tasks: bool = True) -> dict[str, Any]:
    """Validate that the current environment is suitable for strict AlgoTune runs."""
    errors: list[str] = []
    warnings: list[str] = []
    checked_task_count = 0
    failed_tasks: list[dict[str, str]] = []

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info[:2] != (3, 10):
        errors.append(
            f"Strict AlgoTune runs require Python 3.10, found Python {python_version}"
        )

    required_imports = [
        "orjson",
        "numpy",
        "scipy",
        "yaml",
        "AlgoTuneTasks.base",
        "AlgoTuneTasks.factory",
        "AlgoTuner.utils.evaluator.main",
        "AlgoTuner.utils.evaluator.baseline_manager",
        "AlgoTuner.utils.timing_config",
    ]
    for module_name in required_imports:
        try:
            __import__(module_name)
        except Exception as exc:
            errors.append(f"Failed to import {module_name}: {exc}")

    if not ALGOTUNE_ROOT.exists():
        errors.append(f"AlgoTune root does not exist: {ALGOTUNE_ROOT}")
    if not ALGOTUNE_TASKS_DIR.exists():
        errors.append(f"AlgoTune task directory does not exist: {ALGOTUNE_TASKS_DIR}")

    if check_all_tasks and not errors and ALGOTUNE_TASKS_DIR.exists():
        for task_dir in sorted(ALGOTUNE_TASKS_DIR.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith(("_", ".")):
                continue
            task_name = task_dir.name
            checked_task_count += 1
            try:
                _import_task_module(task_name)
            except Exception as exc:
                failed_tasks.append({"task_name": task_name, "error": str(exc)})
        if failed_tasks:
            sample = ", ".join(
                f"{item['task_name']}: {item['error']}" for item in failed_tasks[:5]
            )
            errors.append(
                f"Failed to import {len(failed_tasks)} AlgoTune task modules. Examples: {sample}"
            )

    return {
        "ok": not errors,
        "mode": STRICT_MODE_NAME,
        "python_version": python_version,
        "algotune_root": str(ALGOTUNE_ROOT),
        "task_dir": str(ALGOTUNE_TASKS_DIR),
        "checked_task_count": checked_task_count,
        "failed_tasks": failed_tasks,
        "errors": errors,
        "warnings": warnings,
    }


def assert_benchmark_environment(*, check_all_tasks: bool = True) -> dict[str, Any]:
    result = validate_benchmark_environment(check_all_tasks=check_all_tasks)
    if not result["ok"]:
        detail = "; ".join(result["errors"])
        raise RuntimeError(f"AlgoTune strict benchmark environment validation failed: {detail}")
    return result


@contextmanager
def _candidate_execution_env(task_name: str, solver_path: str | Path):
    solver_dir = str(Path(solver_path).resolve().parent)
    previous = {
        "AGENT_MODE": os.environ.get("AGENT_MODE"),
        "CODE_DIR": os.environ.get("CODE_DIR"),
        "CURRENT_TASK_NAME": os.environ.get("CURRENT_TASK_NAME"),
    }
    os.environ["AGENT_MODE"] = "1"
    os.environ["CODE_DIR"] = solver_dir
    os.environ["CURRENT_TASK_NAME"] = task_name
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_solver_for_compile_check(solver_path: str | Path) -> tuple[bool, str | None]:
    solver_file = Path(solver_path)
    if not solver_file.exists():
        return False, f"Solver file not found: {solver_file}"
    spec = importlib.util.spec_from_file_location("strict_solver_probe", solver_file)
    if spec is None or spec.loader is None:
        return False, f"Could not load module spec for solver: {solver_file}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return False, str(exc)
    if not hasattr(module, "Solver"):
        return False, "Solver class not found in solver.py"
    try:
        module.Solver()
    except Exception as exc:
        return False, f"Solver class failed to initialize: {exc}"
    return True, None


def _resolve_strict_dataset(
    *,
    task_name: str,
    task_instance: Any,
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    from AlgoTuner.config.loader import load_config
    from AlgoTuner.utils.dataset_manager import DatasetManager
    from AlgoTuner.utils.hf_datasets import ensure_hf_dataset

    config = load_config()
    dataset_cfg = config.get("dataset", {}) or {}
    train_size = int(dataset_cfg.get("train_size", 100))
    test_size = int(dataset_cfg.get("test_size", 100))
    target_time_ms = task_instance._get_target_time_ms()

    candidate_roots: list[Path] = []
    hf_revision = os.environ.get("ALGOTUNE_HF_REVISION")
    if hf_revision and hf_revision != "main":
        hf_root = ensure_hf_dataset(task_name)
        if hf_root:
            candidate_roots.append(Path(hf_root))
    for candidate in (
        task_instance.data_dir,
        os.environ.get("DATA_DIR"),
        task_instance.get_task_directory(),
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path not in candidate_roots:
            candidate_roots.append(path)

    for candidate_root in candidate_roots:
        manager = DatasetManager(str(candidate_root))
        train_info = manager.find_dataset(
            task_name,
            target_time_ms=target_time_ms,
            subset="train",
            train_size=train_size,
            test_size=test_size,
        )
        test_info = manager.find_dataset(
            task_name,
            target_time_ms=target_time_ms,
            subset="test",
            train_size=train_size,
            test_size=test_size,
        )
        if train_info and test_info:
            dataset_dir = Path(train_info.path).parent
            return dataset_dir, {"train_size": train_size, "test_size": test_size}, None

    return None, None, (
        "Strict benchmark mode requires an existing official AlgoTune dataset "
        f"for task '{task_name}' with train_size={train_size} and test_size={test_size}. "
        "No matching dataset files were found in the configured data locations. "
        "If you use the Hugging Face dataset path, set ALGOTUNE_HF_REVISION to a pinned "
        "non-'main' revision before running benchmark_strict."
    )


def _normalize_dataset_eval_result(
    *,
    split: str,
    compiled: bool,
    compile_error: str | None,
    eval_output: Any,
) -> dict[str, Any]:
    aggregate_metrics: dict[str, Any] = {}
    error: str | None = compile_error

    if isinstance(eval_output, dict):
        if eval_output.get("evaluation_type") == "error":
            error = eval_output.get("error") or eval_output.get("error_context") or error
        aggregate_metrics = eval_output.get("aggregate_metrics", {}) or {}
    else:
        aggregate_metrics = getattr(eval_output, "aggregate_metrics", {}) or {}

    overall_valid = bool(aggregate_metrics.get("overall_valid", False))
    observed_mean_speedup = aggregate_metrics.get("mean_speedup")
    observed_median_speedup = aggregate_metrics.get("median_speedup")
    num_evaluated = aggregate_metrics.get("num_evaluated")
    num_valid = aggregate_metrics.get("num_valid")
    validity_rate = aggregate_metrics.get("validity_rate")
    if validity_rate is None and isinstance(num_evaluated, int) and num_evaluated > 0 and isinstance(num_valid, int):
        validity_rate = num_valid / num_evaluated
    metric_value = observed_mean_speedup if overall_valid and error is None else None

    if error is None and not compiled:
        error = "Solver failed compile/load checks before strict evaluation"
    if error is None and not overall_valid:
        error = f"Strict {split} evaluation produced invalid solutions"

    status = "succeeded" if metric_value is not None else "failed"
    result = {
        "mode": STRICT_MODE_NAME,
        "split": split,
        "status": status,
        "compiled": compiled,
        "correct": overall_valid,
        "valid_metric": metric_value,
        "speedup": float(metric_value) if metric_value is not None else 0.0,
        "observed_mean_speedup": observed_mean_speedup,
        "mean_speedup": observed_mean_speedup,
        "median_speedup": observed_median_speedup,
        "success_rate": aggregate_metrics.get("success_rate"),
        "validity_rate": validity_rate,
        "num_evaluated": num_evaluated,
        "num_valid": num_valid,
        "num_invalid": aggregate_metrics.get("num_invalid"),
        "num_errors": aggregate_metrics.get("num_errors"),
        "num_timeouts": aggregate_metrics.get("num_timeouts"),
        "avg_solver_time_ms": aggregate_metrics.get("avg_solver_time_ms"),
        "avg_baseline_time_ms": aggregate_metrics.get("avg_oracle_time_ms"),
        "aggregate_metrics": aggregate_metrics,
        "error": error,
    }
    return result


def evaluate_solver_split(
    *,
    task_name: str,
    solver_path: str,
    split: str,
) -> dict[str, Any]:
    """Run strict AlgoTune evaluation on the requested split."""
    assert_benchmark_environment(check_all_tasks=False)

    from AlgoTuneTasks.factory import TaskFactory
    from AlgoTuner.utils.evaluator.baseline_manager import BaselineManager
    from AlgoTuner.utils.evaluator.main import evaluate_code_on_dataset
    from AlgoTuner.utils.timing_config import DEV_RUNS, EVAL_RUNS

    compiled, compile_error = _load_solver_for_compile_check(solver_path)
    if not compiled:
        return _normalize_dataset_eval_result(
            split=split,
            compiled=False,
            compile_error=compile_error,
            eval_output={"evaluation_type": "error", "error": compile_error},
        )

    task_instance = TaskFactory(task_name)
    dataset_dir, dataset_cfg, dataset_error = _resolve_strict_dataset(
        task_name=task_name,
        task_instance=task_instance,
    )
    if dataset_error:
        return _normalize_dataset_eval_result(
            split=split,
            compiled=True,
            compile_error=None,
            eval_output={"evaluation_type": "error", "error": dataset_error},
        )
    if dataset_dir is None or dataset_cfg is None:
        return _normalize_dataset_eval_result(
            split=split,
            compiled=True,
            compile_error=None,
            eval_output={
                "evaluation_type": "error",
                "error": f"Strict {split} evaluation could not resolve dataset paths",
            },
        )

    task_instance.data_dir = str(dataset_dir)
    baseline_manager = BaselineManager(task_instance)
    train_iter, test_iter = task_instance.load_dataset(
        train_size=dataset_cfg["train_size"],
        test_size=dataset_cfg["test_size"],
    )
    dataset_iterable = train_iter if split == "train" else test_iter
    num_runs = DEV_RUNS if split == "train" else EVAL_RUNS

    with _candidate_execution_env(task_name, solver_path):
        eval_output = evaluate_code_on_dataset(
            task_obj=task_instance,
            dataset_iterable=dataset_iterable,
            baseline_manager=baseline_manager,
            data_subset=split,
            default_num_eval_runs=num_runs,
            test_mode=False,
        )

    return _normalize_dataset_eval_result(
        split=split,
        compiled=compiled,
        compile_error=None,
        eval_output=eval_output,
    )


def format_validation_result(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)
