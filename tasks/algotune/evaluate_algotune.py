#!/usr/bin/env python3
"""
AlgoTune Evaluation Script

Evaluates a solver implementation against the AlgoTune task using the official
AlgoTune benchmarking harness for accurate, comparable measurements.

Outputs speedup metric in format: "speedup: X.XXXX"
"""

import argparse
import importlib.util
import logging
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .paths import ensure_algotune_on_path
except ImportError:
    from paths import ensure_algotune_on_path

# Add AlgoTune to path.
ALGOTUNE_ROOT = ensure_algotune_on_path()

# Import AlgoTune's official harness components
from AlgoTuneTasks.base import TASK_REGISTRY, Task
from AlgoTuner.utils.benchmark import run_benchmark
from AlgoTuner.utils.evaluator.scoring import calculate_input_speedup
from AlgoTuner.utils.timing_config import EVAL_RUNS, WARMUPS
from AlgoTuner.utils.blas_utils import set_blas_threads, log_current_blas_threads
from AlgoTuner.utils.dace_config import initialize_dace_for_process

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Official AlgoTune parameters
NUM_PROBLEMS = 100  # test_size from config
NUM_RUNS = EVAL_RUNS  # 10 measurement runs per problem
NUM_WARMUPS = WARMUPS  # 1 warmup run before measurement
BASE_SEED = 42

# Match isolated_benchmark.py MAX_WORKER_CPUS for fair comparison
# The official harness limits isolated workers to 8 threads
MAX_BLAS_THREADS = 8
KNOWN_NON_ISOLATED_TASKS = {
    "aes_gcm_encryption",
    "base64_encoding",
    "chacha_encryption",
    "gzip_compression",
    "sha256_hashing",
}


def should_use_subprocess_isolation(task_name: str, requested: bool) -> bool:
    """Resolve isolation mode with an env override and known task caveats."""
    if sys.platform == "darwin":
        logger.info("Disabling AlgoTune subprocess isolation on macOS due to fork-safety issues.")
        return False
    if os.getenv("ALGOTUNE_DISABLE_SUBPROCESS_ISOLATION", "0") == "1":
        logger.info("Disabling AlgoTune subprocess isolation via ALGOTUNE_DISABLE_SUBPROCESS_ISOLATION=1")
        return False
    if requested and task_name in KNOWN_NON_ISOLATED_TASKS:
        logger.info("Disabling subprocess isolation for known byte-oriented task: %s", task_name)
        return False
    return requested


def initialize_evaluation_environment(solver_code_dir: str = None) -> None:
    """
    Initialize the evaluation environment to match AlgoTune's official harness.

    This ensures:
    1. BLAS threads are limited to MAX_BLAS_THREADS (matching isolated_benchmark.py)
    2. DaCe cache is configured to use the solver's CODE_DIR
    3. Environment variables are set consistently

    Args:
        solver_code_dir: Directory containing the solver (used for DaCe cache)
    """
    # Set ALGOTUNE_BLAS_THREADS env var FIRST - this is checked by set_blas_threads()
    # in precise_timing.py and ensures consistent thread count throughout evaluation
    os.environ["ALGOTUNE_BLAS_THREADS"] = str(MAX_BLAS_THREADS)

    # Set BLAS threads to match isolated_benchmark.py MAX_WORKER_CPUS
    # This ensures baseline and solver comparisons are fair
    n_threads = set_blas_threads(MAX_BLAS_THREADS)
    logger.info(f"BLAS threads configured: {n_threads} (max: {MAX_BLAS_THREADS})")
    log_current_blas_threads("Evaluation environment: ")

    # Initialize DaCe configuration if CODE_DIR is set
    if solver_code_dir:
        os.environ["CODE_DIR"] = solver_code_dir

    initialize_dace_for_process()
    logger.debug("DaCe configuration initialized")


def setup_solver_for_isolated_benchmark(solver_path: str, task_name: str) -> str:
    """
    Create a directory structure compatible with AlgoTune's isolated benchmark.

    In AGENT_MODE=1, AlgoTune's resolve_task_code_dir returns CODE_DIR directly,
    and isolated_benchmark.py then looks for solver.py in that directory.

    So we create: {code_dir}/solver.py

    Args:
        solver_path: Path to the solver file
        task_name: Name of the AlgoTune task

    Returns:
        code_dir path to pass to run_benchmark
    """
    solver_path = Path(solver_path)
    if not solver_path.exists():
        raise FileNotFoundError(f"Solver file not found: {solver_path}")

    # Create temp directory with solver.py directly inside
    # In agent mode, AlgoTune expects {CODE_DIR}/solver.py
    eval_dir = Path(tempfile.mkdtemp(prefix="algotune_eval_"))

    # Copy solver directly to eval_dir/solver.py
    shutil.copy(solver_path, eval_dir / "solver.py")

    logger.debug(f"Set up solver for isolated benchmark: {eval_dir / 'solver.py'}")
    return str(eval_dir)


def load_solver_class(solver_path: str):
    """
    Load the Solver class from a file.

    Args:
        solver_path: Path to solver.py

    Returns:
        Solver class instance
    """
    solver_file = Path(solver_path)
    if not solver_file.exists():
        raise FileNotFoundError(f"Solver file not found: {solver_path}")

    spec = importlib.util.spec_from_file_location("solver", solver_file)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Failed to load solver: {e}")
        raise

    if not hasattr(module, 'Solver'):
        raise ValueError("Solver class not found in solver.py")

    return module.Solver()


def load_task(task_name: str) -> Task:
    """
    Load and return the Task instance.

    Args:
        task_name: Name of the AlgoTune task

    Returns:
        Task instance
    """
    # Import the task module to register it
    task_dir = ALGOTUNE_ROOT / "AlgoTuneTasks" / task_name
    task_file = task_dir / f"{task_name}.py"

    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    spec = importlib.util.spec_from_file_location(task_name, task_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get task from registry
    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name]()

    raise ValueError(f"Task {task_name} not found in registry")


def load_test_problems(task: Task, n_problems: int, seed: int, fast_mode: bool = False) -> List[Dict]:
    """
    Load test problems using AlgoTune's official load_dataset() method.

    This ensures we use the same problems as the official evaluation:
    - If JSONL files exist, loads from them
    - If not, generates and caches them for reproducibility

    Args:
        task: Task instance
        n_problems: Number of problems to load
        seed: Random seed (used if generating new dataset)
        fast_mode: If True, skip load_dataset() and use direct generation (faster but less faithful)

    Returns:
        List of problem dictionaries with metadata
    """
    if fast_mode:
        logger.info("Fast mode enabled - using direct problem generation (skipping load_dataset)")
        return _generate_test_problems_fallback(task, n_problems, seed)

    try:
        # Use AlgoTune's official dataset loading
        # This handles JSONL caching automatically
        logger.info(f"Loading dataset via task.load_dataset(test_size={n_problems}, random_seed={seed})")
        train_iter, test_iter = task.load_dataset(
            train_size=n_problems,  # We don't use train, but API requires it
            test_size=n_problems,
            random_seed=seed
        )

        # Convert test iterator to list
        # Each item is a dict with 'problem', 'seed', 'baseline_time_ms', etc.
        problems = []
        for i, record in enumerate(test_iter):
            if i >= n_problems:
                break
            # Extract the actual problem data
            if isinstance(record, dict):
                problem_data = record.get('problem', record)
                # Store metadata for potential use
                problem_data['_metadata'] = {
                    'seed': record.get('seed'),
                    'baseline_time_ms': record.get('baseline_time_ms'),
                    'k': record.get('k'),
                }
            else:
                problem_data = record
            problems.append(problem_data)

        logger.info(f"Loaded {len(problems)} test problems from dataset")
        return problems

    except Exception as e:
        logger.warning(f"Failed to load dataset via load_dataset(): {e}")
        logger.info("Falling back to generate_problem() for problem generation")
        return _generate_test_problems_fallback(task, n_problems, seed)


def _generate_test_problems_fallback(task: Task, n_problems: int, seed: int) -> List[Dict]:
    """
    Fallback: Generate test problems directly if load_dataset() fails.

    Args:
        task: Task instance
        n_problems: Number of problems to generate
        seed: Base random seed

    Returns:
        List of problem dictionaries
    """
    problems = []
    for i in range(n_problems):
        # Use consistent seeds for reproducibility
        # Problem size varies slightly to test robustness
        n = 100 + (i % 50)  # Sizes from 100-149
        problem = task.generate_problem(n=n, random_seed=seed + i)
        problems.append(problem)
    return problems


def evaluate_single_problem(
    task: Task,
    task_name: str,
    solver,
    solver_code_dir: str,
    problem: Dict,
    num_runs: int,
    warmup_runs: int,
    timeout_seconds: float = 60.0,
    use_isolated_subprocess: bool = True
) -> Dict[str, Any]:
    """
    Evaluate solver on a single problem using AlgoTune's official benchmark harness.

    Args:
        task: Task instance
        task_name: Name of the task (for subprocess identification)
        solver: Solver instance (for validation run)
        solver_code_dir: Directory containing {task_name}/solver.py for subprocess
        problem: Problem dictionary
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        timeout_seconds: Timeout per benchmark
        use_isolated_subprocess: Whether to use subprocess isolation (default True)

    Returns:
        Dictionary with speedup and timing info
    """
    result = {
        "valid": False,
        "speedup": None,
        "baseline_time_ms": None,
        "solver_time_ms": None,
        "error": None
    }

    # Set environment variables for subprocess identification
    os.environ["CURRENT_TASK_NAME"] = task_name
    os.environ["CODE_DIR"] = solver_code_dir
    # AGENT_MODE=1 tells AlgoTune to load solver from CODE_DIR instead of task.solve
    os.environ["AGENT_MODE"] = "1"

    # Benchmark baseline (task.solve) using official harness
    # Use direct mode (force_subprocess=False) since baseline is reference code
    try:
        baseline_result = run_benchmark(
            func=task.solve,
            args=(problem,),
            num_runs=num_runs,
            num_warmups=warmup_runs,
            timeout_seconds=timeout_seconds,
            force_subprocess=False,  # Direct mode for baseline
        )
        if not baseline_result.get('success', False):
            result["error"] = f"Baseline benchmark failed: {baseline_result.get('error', 'unknown')}"
            return result
        baseline_time_s = baseline_result['mean']
        result["baseline_time_ms"] = baseline_time_s * 1000
    except Exception as e:
        result["error"] = f"Baseline failed: {e}"
        return result

    # Benchmark solver using official harness with subprocess isolation
    try:
        solver_result = run_benchmark(
            func=solver.solve,
            args=(problem,),
            num_runs=num_runs,
            num_warmups=warmup_runs,
            timeout_seconds=timeout_seconds,
            force_subprocess=use_isolated_subprocess,  # True = isolated subprocess, False = direct
            working_dir=solver_code_dir,  # Directory containing solver.py
        )
        if not solver_result.get('success', False):
            result["error"] = f"Solver benchmark failed: {solver_result.get('error', 'unknown')}"
            return result
        # Get timing - isolated benchmark returns different keys than direct benchmark
        # Try 'mean' first (direct mode), then 'mean_time_ms' (standard), then 'min_time_ms' (isolated)
        solver_time_s = solver_result.get('mean')
        if solver_time_s is not None:
            result["solver_time_ms"] = solver_time_s * 1000
        elif solver_result.get('mean_time_ms') is not None:
            result["solver_time_ms"] = solver_result['mean_time_ms']
        elif solver_result.get('min_time_ms') is not None:
            result["solver_time_ms"] = solver_result['min_time_ms']
        else:
            result["error"] = "Solver benchmark returned no timing data"
            return result
    except NotImplementedError:
        result["error"] = "Solver not implemented (NotImplementedError)"
        return result
    except Exception as e:
        result["error"] = f"Solver failed: {e}"
        return result

    # Run solver once more for validation (benchmark strips result to save memory)
    try:
        solver_output = solver.solve(problem)
    except Exception as e:
        result["error"] = f"Solver validation run failed: {e}"
        return result

    # Validate solution
    try:
        is_valid = task.is_solution(problem, solver_output)
        result["valid"] = is_valid
    except Exception as e:
        result["error"] = f"Validation failed: {e}"
        result["valid"] = False
        return result

    # Calculate speedup using official formula
    speedup = calculate_input_speedup(
        solver_time_ms=result["solver_time_ms"],
        baseline_time_ms=result["baseline_time_ms"],
        is_valid=is_valid
    )
    result["speedup"] = speedup

    return result


def evaluate_task(
    task_name: str,
    solver_path: str,
    n_problems: int = NUM_PROBLEMS,
    n_runs: int = NUM_RUNS,
    warmup_runs: int = NUM_WARMUPS,
    seed: int = BASE_SEED,
    use_isolated_subprocess: bool = True,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a solver against a task using AlgoTune's official benchmark harness.

    Args:
        task_name: Name of the AlgoTune task
        solver_path: Path to solver.py
        n_problems: Number of test problems (default: 100)
        n_runs: Number of timing runs per problem (default: 10)
        warmup_runs: Number of warmup runs (default: 1)
        seed: Random seed
        use_isolated_subprocess: Use subprocess isolation for timing (default True)
        fast_mode: Skip load_dataset() and use direct generation (faster but less faithful)

    Returns:
        Evaluation results dictionary
    """
    use_isolated_subprocess = should_use_subprocess_isolation(task_name, use_isolated_subprocess)

    results = {
        "task_name": task_name,
        "compiled": False,
        "correct": False,
        "speedup": 0.0,
        "mean_speedup": None,
        "median_speedup": None,
        "error": None,
        "n_problems": n_problems,
        "n_runs": n_runs,
        "warmup_runs": warmup_runs,
        "isolated_subprocess": use_isolated_subprocess,
        "fast_mode": fast_mode,
    }

    solver_code_dir = None

    try:
        # Load task
        try:
            task = load_task(task_name)
            logger.info(f"Loaded task: {task_name}")
        except Exception as e:
            results["error"] = f"Failed to load task: {e}"
            logger.error(results["error"])
            return results

        # Set up solver directory structure for subprocess isolation
        try:
            solver_code_dir = setup_solver_for_isolated_benchmark(solver_path, task_name)
            logger.info(f"Set up solver for isolated benchmark: {solver_code_dir}")
        except Exception as e:
            results["error"] = f"Failed to set up solver directory: {e}"
            logger.error(results["error"])
            return results

        # Initialize evaluation environment (BLAS threads, DaCe cache)
        # This must be done AFTER solver_code_dir is set up
        initialize_evaluation_environment(solver_code_dir)

        # Load solver (for validation runs in main process)
        try:
            solver = load_solver_class(solver_path)
            results["compiled"] = True
            logger.info(f"Loaded solver from: {solver_path}")
        except Exception as e:
            results["error"] = f"Failed to load solver: {e}"
            logger.error(results["error"])
            return results

        # Load test problems (uses official load_dataset() with JSONL caching, unless fast_mode)
        try:
            problems = load_test_problems(task, n_problems=n_problems, seed=seed, fast_mode=fast_mode)
            logger.info(f"Loaded {len(problems)} test problems")
        except Exception as e:
            results["error"] = f"Failed to load problems: {e}"
            logger.error(results["error"])
            return results

        # Evaluate on each problem
        speedups = []
        all_valid = True
        baseline_times = []
        solver_times = []

        for i, problem in enumerate(problems):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Evaluating problem {i+1}/{len(problems)}...")

            problem_result = evaluate_single_problem(
                task=task,
                task_name=task_name,
                solver=solver,
                solver_code_dir=solver_code_dir,
                problem=problem,
                num_runs=n_runs,
                warmup_runs=warmup_runs,
                use_isolated_subprocess=use_isolated_subprocess,
            )

            if problem_result.get("error"):
                # Critical error - stop evaluation
                results["error"] = f"Problem {i}: {problem_result['error']}"
                logger.error(results["error"])
                return results

            if not problem_result.get("valid", False):
                all_valid = False
                logger.warning(f"Problem {i}: Solution validation failed")

            if problem_result.get("speedup") is not None:
                speedups.append(problem_result["speedup"])

            if problem_result.get("baseline_time_ms"):
                baseline_times.append(problem_result["baseline_time_ms"])
            if problem_result.get("solver_time_ms"):
                solver_times.append(problem_result["solver_time_ms"])

        # Calculate aggregate metrics (official AlgoTune method: mean of per-problem speedups)
        results["correct"] = all_valid

        if speedups:
            # Filter out None and infinite speedups for mean calculation
            finite_speedups = [s for s in speedups if s is not None and s != float('inf')]

            if finite_speedups:
                mean_speedup = statistics.mean(finite_speedups)
                median_speedup = statistics.median(finite_speedups)
                results["mean_speedup"] = mean_speedup
                results["median_speedup"] = median_speedup
                # Use mean speedup as the primary metric (matches AlgoTune)
                results["speedup"] = mean_speedup

                logger.info(f"Mean speedup: {mean_speedup:.4f}x")
                logger.info(f"Median speedup: {median_speedup:.4f}x")
            elif all(s == float('inf') for s in speedups if s is not None):
                # All speedups are infinite (solver is instant)
                results["speedup"] = float('inf')
                results["mean_speedup"] = float('inf')
                results["median_speedup"] = float('inf')
                logger.info("Speedup: infinite (solver time ≈ 0)")

        if baseline_times:
            results["avg_baseline_time_ms"] = statistics.mean(baseline_times)
        if solver_times:
            results["avg_solver_time_ms"] = statistics.mean(solver_times)

        if not all_valid:
            logger.warning("Some solutions were invalid - speedup may be affected")
            # If not all valid, speedup is 0 per AlgoTune rules
            if not all_valid and results["speedup"] != 0.0:
                logger.warning("Setting speedup to 0 due to invalid solutions")
                results["speedup"] = 0.0

        return results

    finally:
        # Clean up temporary solver directory
        if solver_code_dir:
            try:
                shutil.rmtree(solver_code_dir, ignore_errors=True)
                logger.debug(f"Cleaned up solver directory: {solver_code_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up solver directory: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AlgoTune solver using official benchmark harness"
    )
    parser.add_argument("--task", required=True, help="AlgoTune task name")
    parser.add_argument("--solution-path", required=True, help="Path to solver.py")
    parser.add_argument(
        "--n-problems", type=int, default=NUM_PROBLEMS,
        help=f"Number of test problems (default: {NUM_PROBLEMS})"
    )
    parser.add_argument(
        "--n-runs", type=int, default=NUM_RUNS,
        help=f"Number of timing runs per problem (default: {NUM_RUNS})"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=NUM_WARMUPS,
        help=f"Number of warmup runs (default: {NUM_WARMUPS})"
    )
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Random seed")
    parser.add_argument(
        "--no-subprocess-isolation", action="store_true",
        help="Disable subprocess isolation (faster but less accurate)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: skip load_dataset() k-estimation and use direct problem generation. "
             "Faster for development but less faithful to official evaluation."
    )

    args = parser.parse_args()

    results = evaluate_task(
        task_name=args.task,
        solver_path=args.solution_path,
        n_problems=args.n_problems,
        n_runs=args.n_runs,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        use_isolated_subprocess=not args.no_subprocess_isolation,
        fast_mode=args.fast
    )

    # Output in format AIDE expects
    if results["error"]:
        print(f"error: {results['error']}")
        print("speedup: 0.0")
    elif not results["correct"]:
        print("error: Solution validation failed")
        print("speedup: 0.0")
    else:
        print(f"speedup: {results['speedup']:.4f}")


if __name__ == "__main__":
    main()
