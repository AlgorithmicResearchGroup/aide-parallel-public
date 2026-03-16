"""Prompt generation helpers for AlgoTune tasks."""

from __future__ import annotations

import re

try:
    from .at_tasks import get_task_code, get_task_description, extract_solve_method
except ImportError:
    from at_tasks import get_task_code, get_task_description, extract_solve_method


def extract_is_solution_method(task_code: str) -> str:
    pattern = r"(    def is_solution\([\s\S]*?(?=\n    def |\Z))"
    match = re.search(pattern, task_code)
    if not match:
        return "# Could not extract is_solution() method"
    method_code = match.group(1).strip()
    return "\n".join(f"{idx + 1}: {line}" for idx, line in enumerate(method_code.splitlines()))


def get_available_packages() -> str:
    packages = [
        "cvxpy",
        "cython",
        "dace",
        "jax[cpu]",
        "networkx",
        "numba",
        "numpy",
        "ortools",
        "pandas",
        "scikit-learn",
        "scipy",
        "sympy",
        "torch",
    ]
    return "\n".join(f" - {pkg}" for pkg in sorted(packages))


def generate_rich_goal(task_name: str) -> str:
    description = get_task_description(task_name)
    task_code = get_task_code(task_name)
    solve_method = extract_solve_method(task_code)
    is_solution_method = extract_is_solution_method(task_code)
    packages = get_available_packages()
    solve_with_numbers = "\n".join(
        f"{idx + 1}: {line}" for idx, line in enumerate(solve_method.splitlines())
    )

    goal = f"""Your objective is to define a class named `Solver` in `solver.py` with a method:
```
class Solver:
    def solve(self, problem, **kwargs) -> Any:
        \"\"\"Your implementation goes here.\"\"\"
        ...
```

IMPORTANT: Compilation time of your init function will not count towards your function's runtime.

This `solve` function will be the entrypoint called by the evaluation harness. Strive to align your class and method implementation as closely as possible with the desired performance criteria.
For each instance, your function can run for at most 10x the reference runtime for that instance. Strive to have your implementation run as fast as possible while returning the same output as the reference function for the same input.

Apart from the default Python packages, you have access to the following additional packages:
{packages}

**TASK DESCRIPTION:**
{description}

Below is the reference implementation. Your function should run much quicker.

**REFERENCE IMPLEMENTATION:**
{solve_with_numbers}

This function will be used to check if your solution is valid for a given problem. If it returns False, it means the solution is invalid:

**VALIDATION FUNCTION:**
{is_solution_method}
"""
    return goal.strip()
