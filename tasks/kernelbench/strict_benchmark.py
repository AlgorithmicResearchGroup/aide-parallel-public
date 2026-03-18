#!/usr/bin/env python3
"""Strict publication-grade KernelBench evaluation helpers."""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

import torch


STRICT_MODE_NAME = "benchmark_strict"
DEFAULT_REFERENCE_BASELINE_FILENAME = "baseline_time_torch.json"
SUPPORTED_REFERENCE_GPU_FAMILIES = ("H100", "A100", "L40S", "L4", "T4", "A10G")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def kernelbench_root() -> Path:
    env_root = os.environ.get("KERNELBENCH_PATH")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return _repo_root() / "tasks" / "kernel_bench" / "KernelBench"


def dataset_root() -> Path:
    explicit = os.environ.get("KERNELBENCH_DATASET_PATH")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if (candidate / "level1").exists():
            return candidate
        if (candidate / "KernelBench" / "level1").exists():
            return candidate / "KernelBench"
        return candidate

    root = kernelbench_root()
    nested = root / "KernelBench"
    if (nested / "level1").exists():
        return nested
    return root


def reference_timing_root() -> Path:
    return kernelbench_root() / "results" / "timing"


@contextmanager
def kernelbench_src_imports():
    kb_root = str(kernelbench_root())
    repo_src_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "src" or key.startswith("src.")
    }
    repo_src_paths = list(sys.path)
    for key in list(repo_src_modules):
        sys.modules.pop(key, None)
    if kb_root in sys.path:
        sys.path.remove(kb_root)
    sys.path.insert(0, kb_root)
    try:
        yield
    finally:
        for key in [name for name in list(sys.modules) if name == "src" or name.startswith("src.")]:
            sys.modules.pop(key, None)
        sys.path[:] = repo_src_paths
        sys.modules.update(repo_src_modules)


def available_reference_baselines() -> list[str]:
    timing_root = reference_timing_root()
    if not timing_root.exists():
        return []
    baselines: list[str] = []
    for child in sorted(timing_root.iterdir()):
        if child.is_dir() and (child / DEFAULT_REFERENCE_BASELINE_FILENAME).exists():
            baselines.append(child.name)
    return baselines


def _normalize_gpu_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", name.upper()).strip()


def detect_runtime_metadata() -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "kernelbench_root": str(kernelbench_root()),
        "dataset_root": str(dataset_root()),
    }


def _infer_gpu_family(reference_baseline: str) -> str | None:
    upper_name = reference_baseline.upper()
    for family in SUPPORTED_REFERENCE_GPU_FAMILIES:
        if family in upper_name:
            return family
    return None


def _validate_reference_baseline_hardware(reference_baseline: str, gpu_name: str | None) -> None:
    family = _infer_gpu_family(reference_baseline)
    if family is None or not gpu_name:
        return
    if family not in _normalize_gpu_name(gpu_name):
        raise RuntimeError(
            f"Reference baseline '{reference_baseline}' targets {family}, but active GPU is '{gpu_name}'. "
            "Use a matching vendored reference baseline or generate a local baseline on this hardware."
        )


def resolve_reference_baseline_path(reference_baseline: str) -> Path:
    timing_root = reference_timing_root()
    direct = timing_root / reference_baseline
    if direct.is_file():
        return direct.resolve()
    if direct.is_dir():
        candidate = direct / DEFAULT_REFERENCE_BASELINE_FILENAME
        if candidate.exists():
            return candidate.resolve()
    candidate = timing_root / reference_baseline / DEFAULT_REFERENCE_BASELINE_FILENAME
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Unknown KernelBench reference baseline '{reference_baseline}'. "
        f"Available baselines: {', '.join(available_reference_baselines()) or 'none'}"
    )


def resolve_baseline_path(
    *,
    baseline_path: str | None = None,
    reference_baseline: str | None = None,
) -> Path:
    path_arg = baseline_path or os.environ.get("AIDE_KERNELBENCH_BASELINE_PATH")
    reference_arg = reference_baseline or os.environ.get("AIDE_KERNELBENCH_REFERENCE_BASELINE")

    if bool(path_arg) == bool(reference_arg):
        raise RuntimeError(
            "KernelBench strict runs require exactly one of "
            "--kb-baseline-path or --kb-reference-baseline."
        )

    if path_arg:
        path = Path(path_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"KernelBench baseline file does not exist: {path}")
        return path

    return resolve_reference_baseline_path(str(reference_arg))


def _load_baseline_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_level(task_id: str) -> int:
    try:
        return int(task_id.split("_", 1)[0])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid KernelBench task id: {task_id}") from exc


def _task_filename(task_id: str) -> str:
    project_root = str(_repo_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from tasks.kernelbench.kb_tasks import get_task_info  # noqa: WPS433

    return Path(get_task_info(task_id)["file_path"]).name


def lookup_baseline_time(task_id: str, baseline_json: dict[str, Any]) -> float:
    level = _task_level(task_id)
    level_key = f"level{level}"
    if level_key not in baseline_json:
        raise KeyError(
            f"Baseline file does not include {level_key}. "
            "Vendored eager baselines currently cover levels 1-3 only."
        )

    filename = _task_filename(task_id)
    level_data = baseline_json[level_key]
    if filename not in level_data:
        raise KeyError(f"Baseline file does not include task entry '{filename}' in {level_key}")

    baseline_entry = level_data[filename]
    if isinstance(baseline_entry, dict):
        value = baseline_entry.get("mean", baseline_entry.get("time"))
    else:
        value = baseline_entry
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"Invalid baseline time for task {task_id}: {baseline_entry!r}")
    return float(value)


def load_model_source(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as handle:
        return handle.read()


def get_original_model_source(task_id: str) -> tuple[str, str]:
    with kernelbench_src_imports():
        construct_kernelbench_dataset = import_module("src.dataset").construct_kernelbench_dataset

    parts = task_id.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid KernelBench task id: {task_id}")
    level = int(parts[0])
    task_num = parts[1]

    dataset = construct_kernelbench_dataset(level)
    for file_path in dataset:
        filename = Path(file_path).name
        if filename.split("_")[0] == task_num:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Task file not found: {path}")
            return str(path), load_model_source(str(path))

    raise ValueError(f"Task {task_id} not found in level {level} dataset")


def validate_benchmark_environment(
    *,
    baseline_path: str | None = None,
    reference_baseline: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    metadata = detect_runtime_metadata()
    errors: list[str] = []
    warnings: list[str] = []
    baseline_resolved: Path | None = None
    baseline_json: dict[str, Any] | None = None

    kb_root = Path(metadata["kernelbench_root"])
    ds_root = Path(metadata["dataset_root"])
    if not kb_root.exists():
        errors.append(f"KernelBench root not found: {kb_root}")
    if not ds_root.exists():
        errors.append(f"KernelBench dataset root not found: {ds_root}")
    if kb_root.exists():
        try:
            with kernelbench_src_imports():
                import_module("src.eval")
                import_module("src.dataset")
        except Exception as exc:
            errors.append(f"KernelBench imports failed: {exc}")
    if not metadata["cuda_available"]:
        errors.append("CUDA is required for strict KernelBench runs, but torch.cuda.is_available() is false")

    try:
        baseline_resolved = resolve_baseline_path(
            baseline_path=baseline_path,
            reference_baseline=reference_baseline,
        )
        baseline_json = _load_baseline_json(baseline_resolved)
    except Exception as exc:
        errors.append(str(exc))

    if reference_baseline and metadata["gpu_name"]:
        try:
            _validate_reference_baseline_hardware(reference_baseline, metadata["gpu_name"])
        except Exception as exc:
            errors.append(str(exc))

    if task_id and baseline_json is not None:
        try:
            lookup_baseline_time(task_id, baseline_json)
        except Exception as exc:
            errors.append(str(exc))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "runtime": metadata,
        "baseline_path": str(baseline_resolved) if baseline_resolved else None,
        "reference_baseline": reference_baseline or os.environ.get("AIDE_KERNELBENCH_REFERENCE_BASELINE"),
        "available_reference_baselines": available_reference_baselines(),
        "task_id": task_id,
    }


def assert_benchmark_environment(
    *,
    baseline_path: str | None = None,
    reference_baseline: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    result = validate_benchmark_environment(
        baseline_path=baseline_path,
        reference_baseline=reference_baseline,
        task_id=task_id,
    )
    if not result["ok"]:
        raise RuntimeError("; ".join(result["errors"]))
    return result


def generate_eager_baseline(*, hardware_name: str) -> Path:
    kb_root = kernelbench_root()
    if str(kb_root) not in sys.path:
        sys.path.insert(0, str(kb_root))
    from scripts.generate_baseline_time import record_baseline_times  # noqa: WPS433

    output_relpath = f"{hardware_name}/{DEFAULT_REFERENCE_BASELINE_FILENAME}"
    record_baseline_times(
        use_torch_compile=False,
        torch_compile_backend=None,
        torch_compile_options=None,
        file_name=output_relpath,
    )
    return (reference_timing_root() / output_relpath).resolve()


def evaluate_kernelbench_task_strict(
    *,
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
) -> dict[str, Any]:
    if device != "cuda":
        raise RuntimeError("Strict KernelBench evaluation requires device='cuda'")

    env_info = assert_benchmark_environment(
        baseline_path=baseline_path,
        reference_baseline=reference_baseline,
        task_id=task_id,
    )

    with kernelbench_src_imports():
        eval_mod = import_module("src.eval")
    KernelExecResult = eval_mod.KernelExecResult
    eval_kernel_against_ref = eval_mod.eval_kernel_against_ref

    solution_file = Path(solution_path)
    if not solution_file.exists():
        raise FileNotFoundError(f"Optimized solution path does not exist: {solution_file}")

    if build_dir is None:
        build_dir = str(Path(__file__).parent / "cuda_build_cache")
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    task_file, original_source = get_original_model_source(task_id)
    optimized_source = load_model_source(str(solution_file))

    if verbose:
        print(f"[kernelbench] strict baseline path: {env_info['baseline_path']}")
        print(f"[kernelbench] strict reference baseline: {env_info['reference_baseline']}")
        print(f"[kernelbench] loaded original model from: {task_file}")
        print(f"[kernelbench] loaded optimized model from: {solution_file}")

    result: KernelExecResult = eval_kernel_against_ref(
        original_model_src=original_source,
        custom_model_src=optimized_source,
        seed_num=42,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        verbose=verbose,
        measure_performance=measure_performance,
        build_dir=str(build_dir),
        device=torch.device("cuda:0"),
    )

    baseline_json = _load_baseline_json(Path(env_info["baseline_path"]))
    baseline_time = lookup_baseline_time(task_id, baseline_json)
    optimized_time = result.runtime if isinstance(result.runtime, (int, float)) and result.runtime > 0 else None
    speedup = None
    if optimized_time is not None:
        speedup = baseline_time / float(optimized_time)

    output = {
        "mode": STRICT_MODE_NAME,
        "compiled": result.compiled,
        "correct": result.correctness,
        "speedup": float(speedup) if speedup is not None and result.correctness else 0.0,
        "valid_metric": float(speedup) if speedup is not None and result.correctness else None,
        "baseline_time_ms": baseline_time,
        "optimized_time_ms": float(optimized_time) if optimized_time is not None else None,
        "baseline_path": env_info["baseline_path"],
        "reference_baseline": env_info["reference_baseline"],
        "gpu_name": env_info["runtime"]["gpu_name"],
        "torch_version": env_info["runtime"]["torch_version"],
        "cuda_version": env_info["runtime"]["cuda_version"],
        "metadata": result.metadata,
        "error": None,
    }
    if not result.compiled:
        output["error"] = result.metadata.get("compilation_error") or "Compilation failed"
    elif not result.correctness:
        output["error"] = (
            result.metadata.get("runtime_error")
            or result.metadata.get("correctness_issue")
            or "KernelBench correctness check failed"
        )

    return output
