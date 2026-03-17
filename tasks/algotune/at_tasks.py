"""AlgoTune task discovery and metadata helpers."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from .paths import ensure_algotune_on_path, resolve_algotune_tasks_dir
except ImportError:
    from paths import ensure_algotune_on_path, resolve_algotune_tasks_dir

ALGOTUNE_ROOT = ensure_algotune_on_path()
ALGOTUNE_TASKS_DIR = resolve_algotune_tasks_dir()

TASK_RUNTIME_REQUIREMENTS: dict[str, list[tuple[str, str, str]]] = {
    "cumulative_simpson_1d": [
        ("scipy.integrate", "cumulative_simpson", "requires scipy.integrate.cumulative_simpson"),
    ],
    "cumulative_simpson_multid": [
        ("scipy.integrate", "cumulative_simpson", "requires scipy.integrate.cumulative_simpson"),
    ],
    "dijkstra_from_indices": [
        ("scipy.sparse", "random_array", "requires scipy.sparse.random_array"),
    ],
    "shortest_path_dijkstra": [
        ("scipy.sparse", "random_array", "requires scipy.sparse.random_array"),
    ],
}

_TASK_COMPATIBILITY_CACHE: dict[str, dict[str, Any]] = {}


def get_all_tasks() -> List[str]:
    """List available AlgoTune task names."""
    tasks: list[str] = []
    if not ALGOTUNE_TASKS_DIR.exists():
        raise FileNotFoundError(f"AlgoTuneTasks directory not found: {ALGOTUNE_TASKS_DIR}")

    for item in ALGOTUNE_TASKS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith(("_", ".")):
            task_file = item / f"{item.name}.py"
            if task_file.exists():
                tasks.append(item.name)

    return sorted(tasks)


def get_task_description(task_name: str) -> str:
    desc_file = ALGOTUNE_TASKS_DIR / task_name / "description.txt"
    if not desc_file.exists():
        return f"No description available for task: {task_name}"
    return desc_file.read_text(encoding="utf-8")


def get_task_code(task_name: str) -> str:
    task_file = ALGOTUNE_TASKS_DIR / task_name / f"{task_name}.py"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    return task_file.read_text(encoding="utf-8")


def extract_solve_method(task_code: str) -> str:
    pattern = r"(def solve\(self.*?(?=\n    def |\nclass |\Z))"
    match = re.search(pattern, task_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "# Could not extract solve() method"


def get_task_category(description: str) -> str:
    match = re.search(r"Category:\s*([\w_]+)", description, re.IGNORECASE)
    if match:
        return match.group(1)
    return "general"


def get_task_info(task_name: str) -> Dict[str, Any]:
    description = get_task_description(task_name)
    category = get_task_category(description)
    suggested_steps = {
        "matrix_operations": 8,
        "optimization": 10,
        "graph": 8,
        "signal_processing": 8,
        "machine_learning": 10,
        "cryptography": 6,
        "general": 8,
    }.get(category, 8)

    return {
        "name": task_name,
        "category": category,
        "description": description,
        "suggested_steps": suggested_steps,
    }


def get_task_info_with_code(task_name: str) -> Dict[str, Any]:
    try:
        from .algotune_prompts import generate_rich_goal
    except ImportError:
        from algotune_prompts import generate_rich_goal

    info = get_task_info(task_name)
    task_code = get_task_code(task_name)
    info["task_code"] = task_code
    info["solve_method"] = extract_solve_method(task_code)
    info["goal"] = generate_rich_goal(task_name)
    return info


def load_task_class(task_name: str):
    task_file = ALGOTUNE_TASKS_DIR / task_name / f"{task_name}.py"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    spec = importlib.util.spec_from_file_location(task_name, task_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for task: {task_name}")
    module = importlib.util.module_from_spec(spec)

    from AlgoTuneTasks.base import TASK_REGISTRY, Task

    spec.loader.exec_module(module)

    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name]()

    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, Task) and obj is not Task:
            return obj()

    raise ValueError(f"Could not find Task class for: {task_name}")


def check_task_runtime_compatibility(task_name: str) -> Dict[str, Any]:
    cached = _TASK_COMPATIBILITY_CACHE.get(task_name)
    if cached is not None:
        return cached

    if task_name in TASK_RUNTIME_REQUIREMENTS:
        for module_name, attr_name, reason in TASK_RUNTIME_REQUIREMENTS[task_name]:
            try:
                module = __import__(module_name, fromlist=[attr_name])
            except Exception as exc:
                result = {
                    "compatible": False,
                    "reason": f"{reason}: failed to import {module_name} ({exc})",
                }
                _TASK_COMPATIBILITY_CACHE[task_name] = result
                return result
            if not hasattr(module, attr_name):
                result = {
                    "compatible": False,
                    "reason": f"{reason}: {module_name}.{attr_name} is unavailable",
                }
                _TASK_COMPATIBILITY_CACHE[task_name] = result
                return result

    try:
        load_task_class(task_name)
    except Exception as exc:
        result = {
            "compatible": False,
            "reason": f"task import failed: {exc}",
        }
        _TASK_COMPATIBILITY_CACHE[task_name] = result
        return result

    result = {"compatible": True, "reason": None}
    _TASK_COMPATIBILITY_CACHE[task_name] = result
    return result


if __name__ == "__main__":
    tasks = get_all_tasks()
    print(f"Found {len(tasks)} AlgoTune tasks:")
    for index, task in enumerate(tasks[:10], 1):
        info = get_task_info(task)
        print(f"  {index}. {task} ({info['category']})")
    if len(tasks) > 10:
        print(f"  ... and {len(tasks) - 10} more")
