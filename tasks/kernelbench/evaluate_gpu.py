#!/usr/bin/env python3
"""
KernelBench GPU Evaluation Script
This script uses the official KernelBench evaluation functions to ensure exact compatibility
with the published ICML'25 benchmark.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add KernelBench to path using environment-aware resolution.
repo_root = Path(__file__).resolve().parents[2]
possible_paths = []
if os.environ.get("KERNELBENCH_PATH"):
    possible_paths.append(Path(os.environ["KERNELBENCH_PATH"]).expanduser().resolve())
possible_paths.extend(
    [
        repo_root / "tasks" / "kernel_bench" / "KernelBench",
        Path(__file__).parent.parent / "kernel_bench" / "KernelBench",
    ]
)

kernelbench_path = None
for path in possible_paths:
    if path and path.exists():
        kernelbench_path = path
        break

if not kernelbench_path:
    print("Error: Could not find KernelBench directory. Tried:")
    for path in possible_paths:
        if path:
            print(f"  - {path}")
    sys.exit(1)

sys.path.insert(0, str(kernelbench_path))

# Import KernelBench's official evaluation functions
# Dependencies will be installed via runtime_env
try:
    from src.eval import eval_kernel_against_ref, KernelExecResult
    from src.dataset import construct_kernelbench_dataset
except ImportError as e:
    print(f"Error: Could not import KernelBench evaluation functions: {e}")
    print(f"Make sure KernelBench is available at: {kernelbench_path}")
    print(f"You may need to install missing dependencies or check the path")
    sys.exit(1)


def load_model_source(file_path: str) -> str:
    """Load model source code from file."""
    with open(file_path, 'r') as f:
        return f.read()


def get_original_model_source(task_id: str, kernelbench_path: Path) -> tuple[str, str]:
    """Get the original model source code for a task.

    Returns:
        tuple: (task_file_path, source_code)
    """
    # Parse task ID (e.g., "1_23" -> level 1, task 23 which is "23_Softmax.py")
    parts = task_id.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid task ID format: {task_id}. Expected format: 'level_number' (e.g., '1_23')")

    level = int(parts[0])
    task_num = parts[1]

    # Load dataset for this level - it returns a list of file paths
    dataset = construct_kernelbench_dataset(level)

    # Find the task file by matching the task number in the filename
    # Task files are named like "23_Softmax.py" or "1_Square_matrix_multiplication_.py"
    for file_path in dataset:
        filename = Path(file_path).name
        # Extract the number from the beginning of the filename
        if filename.split('_')[0] == task_num:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Task file not found: {file_path}")
            return file_path, load_model_source(file_path)

    raise ValueError(f"Task {task_id} not found in level {level} dataset")


def evaluate_kernelbench_task(
    task_id: str,
    solution_path: str,
    device: str = "cuda",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    measure_performance: bool = True,
    verbose: bool = True,
    build_dir: str = None
) -> dict:
    """
    Evaluate a KernelBench task using the official evaluation pipeline.

    Args:
        task_id: Task identifier (e.g., "1_23" for level 1, task 23)
        solution_path: Path to the solution Python file containing ModelNew
        device: Device to run on (default: "cuda")
        num_correct_trials: Number of correctness trials (default: 5, matching KernelBench)
        num_perf_trials: Number of performance trials (default: 100, matching KernelBench)
        measure_performance: Whether to measure performance if correct (default: True)
        verbose: Print detailed output (default: True)
        build_dir: Directory for CUDA compilation cache (default: auto-generated)

    Returns:
        dict: Evaluation results with speedup and correctness information
    """

    # Set up build directory for CUDA compilation cache
    if build_dir is None:
        build_dir = Path(__file__).parent / "cuda_build_cache"
    build_dir = Path(build_dir)
    build_dir.mkdir(exist_ok=True, parents=True)

    # Get paths
    kernelbench_path = Path(os.environ.get("KERNELBENCH_PATH", str(repo_root / "tasks" / "kernel_bench" / "KernelBench"))).expanduser().resolve()

    # Load original model source
    try:
        task_file, original_source = get_original_model_source(task_id, kernelbench_path)
        if verbose:
            print(f"Loaded original model from: {task_file}")
    except Exception as e:
        print(f"Error loading original model: {e}")
        return {"speedup": 0.0, "correct": False, "error": str(e)}

    # Load optimized model source
    try:
        optimized_source = load_model_source(solution_path)
        if verbose:
            print(f"Loaded optimized model from: {solution_path}")
    except Exception as e:
        print(f"Error loading optimized model: {e}")
        return {"speedup": 0.0, "correct": False, "error": str(e)}

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    # Create torch device with proper index for CUDA
    if device == "cuda":
        torch_device = torch.device("cuda:0")  # Use first GPU
    else:
        torch_device = torch.device("cpu")

    if verbose:
        print(f"\nEvaluating on device: {torch_device}")
        print(f"Correctness trials: {num_correct_trials}")
        print(f"Performance trials: {num_perf_trials}")
        print("-" * 50)

    # Call KernelBench's official evaluation function
    try:
        result: KernelExecResult = eval_kernel_against_ref(
            original_model_src=original_source,
            custom_model_src=optimized_source,
            seed_num=42,  # Default seed for reproducibility
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            measure_performance=measure_performance and device != "cpu",
            build_dir=str(build_dir),
            device=torch_device
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"speedup": 0.0, "correct": False, "error": str(e)}

    # Process results
    output = {
        "compiled": result.compiled,
        "correct": result.correctness,
        "speedup": 0.0,
        "metadata": result.metadata
    }

    # Calculate speedup if performance was measured
    if result.correctness and result.runtime > 0:
        # Try to get baseline time from KernelBench's pre-computed baseline files
        baseline_time = None

        # Try to fetch baseline time from pre-computed files
        try:
            # Import the fetch_baseline_time function from KernelBench
            from src.eval import fetch_baseline_time

            # Try different possible baseline time file paths
            baseline_files = [
                kernelbench_path / "results" / "timing" / "H100_PCIe_LambdaLabs" / "baseline_time_torch.json",
                kernelbench_path / "results" / "timing" / "H100_SXM_LAMBDA" / "baseline_time_torch.json",
                kernelbench_path / "results" / "timing" / "H100_80GB_HBM3" / "baseline_time_torch.json",
            ]

            # Parse task ID to get level and filename
            parts = task_id.split('_')
            level = int(parts[0])
            task_num = parts[1]

            # Get the dataset to find the actual filename
            dataset = construct_kernelbench_dataset(level)
            task_filename = None
            for file_path in dataset:
                if Path(file_path).name.split('_')[0] == task_num:
                    task_filename = Path(file_path).name
                    break

            if task_filename:
                # Try each baseline file
                for baseline_file in baseline_files:
                    if baseline_file.exists():
                        try:
                            import json
                            with open(baseline_file, 'r') as f:
                                baseline_json = json.load(f)

                            level_key = f"level{level}"
                            if level_key in baseline_json and task_filename in baseline_json[level_key]:
                                baseline_data = baseline_json[level_key][task_filename]
                                if isinstance(baseline_data, dict):
                                    baseline_time = baseline_data.get('mean', baseline_data.get('time'))
                                else:
                                    baseline_time = baseline_data
                                print(f"Using baseline time from {baseline_file.name}: {baseline_time} ms")
                                break
                        except Exception as e:
                            continue
        except Exception as e:
            print(f"Could not fetch baseline time from files: {e}")

        # If we found a baseline time, calculate speedup
        if baseline_time and baseline_time > 0:
            speedup = baseline_time / result.runtime
            output["speedup"] = speedup
            output["original_time"] = baseline_time
            output["optimized_time"] = result.runtime
            print(f"Calculated speedup: {speedup:.4f}x (baseline: {baseline_time:.3f}ms, optimized: {result.runtime:.3f}ms)")
        else:
            # Fail loudly if we can't load baseline
            raise ValueError(f"Failed to load baseline time for task {task_id}")

    # Print results in a clear format
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    if not result.compiled:
        print("❌ COMPILATION FAILED")
        if "compilation_error" in result.metadata:
            print(f"   Error: {result.metadata['compilation_error'][:200]}...")
    else:
        print("✅ Compilation successful")

    if result.compiled:
        correctness_info = result.metadata.get("correctness_trials", f"(0/{num_correct_trials})")
        if result.correctness:
            print(f"✅ CORRECTNESS PASSED {correctness_info}")
            print(f"   Tolerance used: rtol=1e-2, atol=1e-2 (KernelBench official)")
        else:
            print(f"❌ CORRECTNESS FAILED {correctness_info}")
            print(f"   Tolerance used: rtol=1e-2, atol=1e-2 (KernelBench official)")
            if "max_difference" in result.metadata:
                print(f"   Max difference: {result.metadata['max_difference']}")
                # Provide helpful feedback about numerical precision
                max_diff = result.metadata.get('max_difference', 0)
                if isinstance(max_diff, (int, float)) and max_diff < 0.1:
                    print(f"   ℹ️  Note: This may be a numerical precision issue from CUDA kernel optimization")
                    print(f"   ℹ️  Small differences are expected when using custom CUDA kernels vs cuDNN")
            if "runtime_error" in result.metadata:
                print(f"   Runtime error: {result.metadata['runtime_error'][:200]}...")

    if result.correctness and output["speedup"] > 0:
        if output["speedup"] >= 1.0:
            print(f"🚀 SPEEDUP: {output['speedup']:.4f}x (faster than baseline!)")
        else:
            print(f"🐌 SPEEDUP: {output['speedup']:.4f}x (slower than baseline)")
        original_time = output.get('original_time', 'N/A')
        optimized_time = output.get('optimized_time', 'N/A')
        if isinstance(original_time, (int, float)):
            print(f"   Original time: {original_time:.3f} ms")
        else:
            print(f"   Original time: {original_time}")
        if isinstance(optimized_time, (int, float)):
            print(f"   Optimized time: {optimized_time:.3f} ms")
        else:
            print(f"   Optimized time: {optimized_time}")
        if result.runtime_stats:
            print(f"   Stats: {result.runtime_stats}")
    elif result.correctness:
        print("⚠️  Performance not measured (CPU mode or measurement disabled)")
    else:
        print("⚠️  Performance not measured (correctness failed)")

    print("=" * 50)

    # Return speedup value for AIDE to parse
    # Report actual speedup if code runs correctly, even if < 1.0
    # Only report 0.0 for compilation or runtime failures
    if result.correctness and output["speedup"] > 0:
        print(f"\nspeedup: {output['speedup']:.4f}")
    elif result.correctness and "speedup" in output:
        # Code runs correctly but might be slower than baseline
        # Still report the actual speedup value (e.g., 0.7 means 30% slower)
        print(f"\nspeedup: {output.get('speedup', 0.0):.4f}")
    else:
        # Compilation failed or runtime error - report 0.0
        print("\nspeedup: 0.0000")

    return output


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate KernelBench GPU kernel optimizations using official KernelBench evaluation"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="Task ID (e.g., '1_23' for level 1, task 23 - Softmax)"
    )
    parser.add_argument(
        "--solution-path",
        type=str,
        required=True,
        help="Path to the solution Python file containing ModelNew class"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--num-correct-trials",
        type=int,
        default=5,
        help="Number of correctness trials (default: 5, matching KernelBench)"
    )
    parser.add_argument(
        "--num-perf-trials",
        type=int,
        default=100,
        help="Number of performance trials (default: 100, matching KernelBench)"
    )
    parser.add_argument(
        "--no-performance",
        action="store_true",
        help="Skip performance measurement (only check correctness)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Directory for CUDA compilation cache"
    )

    args = parser.parse_args()

    # Evaluate the task
    results = evaluate_kernelbench_task(
        task_id=args.task_id,
        solution_path=args.solution_path,
        device=args.device,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        measure_performance=not args.no_performance,
        verbose=not args.quiet,
        build_dir=args.build_dir
    )

    # Exit with appropriate code
    if results["correct"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
