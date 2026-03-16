"""Prepare an AlgoTune solver template for AIDE."""

from __future__ import annotations

from pathlib import Path

try:
    from .at_tasks import extract_solve_method, get_task_code, get_task_description
except ImportError:
    from at_tasks import extract_solve_method, get_task_code, get_task_description


def prepare_task(task_name: str, solver_path: str) -> None:
    description = get_task_description(task_name)
    task_code = get_task_code(task_name)
    solve_method = extract_solve_method(task_code)
    imports = extract_imports(task_code)

    solver_file = Path(solver_path)
    solver_file.parent.mkdir(parents=True, exist_ok=True)
    solver_file.write_text(
        generate_solver_template(
            task_name=task_name,
            description=description,
            solve_method=solve_method,
            imports=imports,
        ),
        encoding="utf-8",
    )


def extract_imports(task_code: str) -> str:
    imports: list[str] = []
    seen: set[str] = set()
    for line in task_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if "AlgoTuneTasks" not in stripped and "AlgoTuner" not in stripped:
                if stripped not in seen:
                    imports.append(line)
                    seen.add(stripped)
    return "\n".join(imports)


def generate_solver_template(task_name: str, description: str, solve_method: str, imports: str) -> str:
    return f'''"""
AlgoTune Task: {task_name}

{description}

YOUR TASK:
Implement the Solver class with an optimized solve() method.
The solve() method must:
1. Accept the same input format as described above
2. Return the same output format as described above
3. Produce correct results validated by is_solution()
4. Run faster than the reference implementation
"""

{imports}
import numpy as np


class Solver:
    """Optimized solver for {task_name}."""

    def solve(self, problem, **kwargs):
        """Return a valid solution for the provided problem dictionary."""
        # TODO: Replace this placeholder with an optimized implementation.
        # Reference solve() method:
        # {solve_method.replace(chr(10), chr(10) + "        # ")}
        raise NotImplementedError("Implement your optimized solution")
'''


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare AlgoTune task")
    parser.add_argument("task_name", help="Name of the AlgoTune task")
    parser.add_argument("--output", "-o", default="solver.py", help="Output file path")
    args = parser.parse_args()

    prepare_task(args.task_name, args.output)
    print(f"Created solver template: {args.output}")
