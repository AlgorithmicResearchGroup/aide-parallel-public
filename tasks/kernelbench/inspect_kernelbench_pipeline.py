#!/usr/bin/env python3
"""
Inspection script for KernelBench pipeline.

This script shows exactly:
1. What the AIDE model sees (the complete prompt/goal)
2. What the agent prompts are (the instructions given to AIDE)
3. How the task is graded (evaluation process and metrics)

Usage:
    python inspect_kernelbench_pipeline.py <task_id>
    python inspect_kernelbench_pipeline.py 1_19  # For ReLU task
"""

import sys
import os
import argparse
from pathlib import Path
import json
import subprocess
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_section(title, content=None):
    """Print a formatted section header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)
    if content:
        print(content)


def inspect_task(task_id: str, verbose: bool = False):
    """
    Inspect all aspects of a KernelBench task pipeline.

    Args:
        task_id: KernelBench task ID (e.g., "1_19")
        verbose: If True, print full content; if False, truncate long sections
    """

    print(f"\n{'#' * 100}")
    print(f"  KERNELBENCH PIPELINE INSPECTION FOR TASK: {task_id}")
    print(f"{'#' * 100}")

    # Import required modules
    from kb_tasks import get_task_info, get_task_info_with_code
    from prepare_kernelbench_task import prepare_task
    from kernelbench_prompts import generate_enhanced_task_info

    # Get basic task info
    basic_info = get_task_info(task_id)

    print_section("TASK METADATA")
    print(f"  Task ID:     {task_id}")
    print(f"  Name:        {basic_info['name']}")
    print(f"  Level:       {basic_info['level']}")
    print(f"  Category:    {basic_info['category']}")
    print(f"  File Path:   {basic_info['file_path']}")
    print(f"  Steps:       {basic_info['suggested_steps']}")

    # =========================================================================
    # SECTION 1: WHAT THE MODEL SEES (Complete Prompt/Goal)
    # =========================================================================

    print_section("1. WHAT THE AIDE MODEL SEES (THE COMPLETE PROMPT/GOAL)")

    # Get enhanced task info with rich goal
    enhanced_info = get_task_info_with_code(task_id)

    # This is exactly what gets passed to AIDE as the "goal" in the contract
    aide_sees = enhanced_info['goal']

    if verbose:
        print(aide_sees)
    else:
        # Show first 3000 chars and last 500 chars
        if len(aide_sees) > 4000:
            print(aide_sees[:3000])
            print("\n... [CONTENT TRUNCATED - Use --verbose to see full content] ...\n")
            print(aide_sees[-500:])
        else:
            print(aide_sees)

    print(f"\n  [Total prompt length: {len(aide_sees)} characters]")

    # =========================================================================
    # SECTION 2: THE PREPARED optimize.py FILE
    # =========================================================================

    print_section("2. THE PREPARED optimize.py FILE (WHAT AIDE WILL MODIFY)")

    # Create a temporary file to see what prepare_task generates
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Prepare the task
        prepare_task(task_id, tmp_path)

        # Read what was generated
        with open(tmp_path, 'r') as f:
            optimize_content = f.read()

        if verbose:
            print(optimize_content)
        else:
            # Show key sections
            lines = optimize_content.split('\n')

            # Find and show key sections
            print("\n[Showing key sections of optimize.py]\n")

            # Show header
            for i, line in enumerate(lines[:50]):
                print(line)
                if "IMPORTANT:" in line:
                    break

            print("\n... [MIDDLE SECTIONS OMITTED] ...\n")

            # Show the ModelNew template at the end
            start_idx = -1
            for i, line in enumerate(lines):
                if "class ModelNew" in line:
                    start_idx = i
                    break

            if start_idx > 0:
                print("Template for ModelNew class that AIDE will implement:")
                print("-" * 50)
                for line in lines[start_idx:min(start_idx + 30, len(lines))]:
                    print(line)

        print(f"\n  [Total optimize.py length: {len(optimize_content)} characters]")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # =========================================================================
    # SECTION 3: AGENT PROMPTS AND CONTRACT
    # =========================================================================

    print_section("3. AGENT PROMPTS AND CONTRACT DETAILS")

    # Load and show the contract template
    contract_path = Path(__file__).parent / "contract_gpu.yaml"
    if contract_path.exists():
        with open(contract_path, 'r') as f:
            contract = f.read()

        print("\nBase Contract Template (contract_gpu.yaml):")
        print("-" * 50)
        for line in contract.split('\n')[:30]:
            print(line)
        print("...")

    # Show what gets injected
    print("\n\nDynamic Contract Values for this task:")
    print("-" * 50)
    print(f"  goal: [The rich goal shown in Section 1 - {len(aide_sees)} chars]")
    print(f"  eval: python evaluate_gpu.py --task-id {task_id} --solution-path optimize.py --device cuda")
    print(f"  steps: {basic_info['suggested_steps']}")
    print(f"  metric: speedup")

    # Show the AIDE Experiment configuration
    print("\n\nAIDE Experiment Configuration:")
    print("-" * 50)
    print("  exp = aide.Experiment(")
    print(f"      data_dir='tasks/kernelbench/',")
    print(f"      goal=[THE RICH GOAL FROM SECTION 1],")
    print(f"      eval='python evaluate_gpu.py --task-id {task_id} ...'")
    print("  )")
    print("  exp.cfg.agent.code.model = [model from args or contract]")
    print("  exp.cfg.agent.feedback.model = [feedback_model from args or contract]")
    print(f"  exp.cfg.exec.timeout = 3600  # seconds")

    # Show the agent's internal prompts
    print("\n\nAgent's Internal Process:")
    print("-" * 50)
    print("1. AIDE reads the goal (Section 1) and understands the task")
    print("2. AIDE examines optimize.py (Section 2) with the original Model class")
    print("3. AIDE generates code to create ModelNew class")
    print("4. AIDE runs evaluation to get speedup metric")
    print("5. AIDE iterates based on feedback to improve speedup")
    print(f"6. Process repeats for {basic_info['suggested_steps']} steps")

    # =========================================================================
    # SECTION 4: HOW THE TASK IS GRADED
    # =========================================================================

    print_section("4. HOW THE TASK IS GRADED (EVALUATION PROCESS)")

    print("\nEvaluation Script: evaluate_gpu.py")
    print("-" * 50)

    print("\nStep-by-step Evaluation Process:")
    print()
    print("1. LOAD ORIGINAL MODEL:")
    print(f"   - Load task file: {basic_info['file_path']}")
    print("   - Create instance: original_model = Model(*init_inputs)")
    print()

    print("2. LOAD OPTIMIZED MODEL:")
    print("   - Load optimize.py (modified by AIDE)")
    print("   - Create instance: optimized_model = ModelNew(*init_inputs)")
    print("   - If ModelNew doesn't exist, fall back to Model (with warning)")
    print()

    print("3. GET INPUTS:")
    print("   - Call get_inputs() from task file")
    print(f"   - For this task, inputs will be tensors with specific shapes")

    # Load the actual task to show input dimensions
    try:
        with open(basic_info['file_path'], 'r') as f:
            task_code = f.read()

        # Extract batch_size and dimensions
        for line in task_code.split('\n'):
            if 'batch_size' in line or 'dim' in line:
                if '=' in line and not line.strip().startswith('#'):
                    print(f"   - {line.strip()}")
    except:
        pass

    print()
    print("4. CORRECTNESS CHECK:")
    print("   - Run both models with same inputs")
    print("   - original_output = original_model(*inputs)")
    print("   - optimized_output = optimized_model(*inputs)")
    print("   - Check: torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-5)")
    print("   - If outputs don't match → correctness = False (but still measure speed)")
    print()

    print("5. PERFORMANCE MEASUREMENT:")
    print("   - Warmup: Run each model 3 times (default)")
    print("   - Timing: Run each model 10 times (default)")
    print("   - Use CUDA events for accurate GPU timing:")
    print("     ```python")
    print("     start_event.record()")
    print("     output = model(*inputs)")
    print("     end_event.record()")
    print("     torch.cuda.synchronize()")
    print("     time = start_event.elapsed_time(end_event)")
    print("     ```")
    print("   - Calculate average time for each model")
    print()

    print("6. CALCULATE SPEEDUP:")
    print("   - speedup = original_time / optimized_time")
    print("   - Example interpretations:")
    print("     * speedup = 1.0 → Same speed as original")
    print("     * speedup = 2.0 → 2x faster (optimized takes half the time)")
    print("     * speedup = 0.5 → 2x slower (optimization failed)")
    print()

    print("7. RETURN METRICS TO AIDE:")
    print("   - Primary metric: speedup (higher is better)")
    print("   - Also tracked: correctness, original_time, optimized_time")
    print("   - AIDE uses speedup to guide optimization")
    print()

    print("GRADING CRITERIA:")
    print("-" * 50)
    print("  ✅ PASS: correctness = True AND speedup > 1.0")
    print("  ⚠️  SLOW: correctness = True BUT speedup ≤ 1.0")
    print("  ❌ FAIL: correctness = False (regardless of speedup)")
    print()

    print("TARGET SPEEDUPS BY LEVEL:")
    print("-" * 50)
    print(f"  Level {basic_info['level']}: ", end="")
    if basic_info['level'] == 1:
        print("1.5x - 3x typical (single kernel optimizations)")
    elif basic_info['level'] == 2:
        print("2x - 5x typical (fusion benefits)")
    elif basic_info['level'] == 3:
        print("1.5x - 4x typical (architecture-dependent)")
    else:
        print("1.2x - 2x typical (already optimized baselines)")

    # =========================================================================
    # SECTION 5: COMPLETE EVALUATION COMMAND
    # =========================================================================

    print_section("5. COMPLETE EVALUATION COMMAND")

    eval_cmd = f"python evaluate_gpu.py --task-id {task_id} --solution-path optimize.py --device cuda"
    print(f"\n  {eval_cmd}")

    print("\n\nCommand-line Options:")
    print("-" * 50)
    print("  --task-id        : KernelBench task identifier")
    print("  --solution-path  : Path to optimized solution (optimize.py)")
    print("  --device         : Device to run on (cuda or cpu)")
    print("  --n-warmup       : Number of warmup iterations (default: 3)")
    print("  --n-trials       : Number of timing trials (default: 10)")

    # =========================================================================
    # SECTION 6: SUMMARY
    # =========================================================================

    print_section("6. COMPLETE PIPELINE SUMMARY")

    print(f"""
Task: {task_id} - {basic_info['name']}

PIPELINE FLOW:
1. Task Preparation:
   - Original task loaded from: {basic_info['file_path']}
   - Task prepared into: optimize.py
   - Examples loaded from: KernelBench/src/prompts/

2. AIDE Optimization:
   - AIDE receives: {len(aide_sees)} character prompt with full context
   - AIDE modifies: optimize.py to create ModelNew class
   - AIDE runs for: {basic_info['suggested_steps']} steps
   - Each step evaluated for speedup

3. Evaluation per Step:
   - Load both Model and ModelNew
   - Check correctness (tolerance: rtol=1e-3, atol=1e-5)
   - Measure performance (10 trials, GPU timing)
   - Calculate speedup metric
   - Return to AIDE for next iteration

4. Success Metrics:
   - Must maintain correctness
   - Target speedup > 1.0 (faster than original)
   - Ideal speedup for Level {basic_info['level']}: {'1.5-3x' if basic_info['level'] == 1 else '2-5x' if basic_info['level'] == 2 else '1.5-4x'}

5. Final Output:
   - Best speedup achieved across all steps
   - Final optimized code in optimize.py
   - Results logged to MLflow if enabled
""")


def main():
    parser = argparse.ArgumentParser(description="Inspect KernelBench pipeline for a task")
    parser.add_argument("task_id", help="KernelBench task ID (e.g., 1_19)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show full content without truncation")
    args = parser.parse_args()

    try:
        inspect_task(args.task_id, verbose=args.verbose)
    except Exception as e:
        print(f"\nError inspecting task {args.task_id}: {e}")
        print("\nMake sure the task ID is valid (e.g., 1_19, 2_1, 3_28)")
        sys.exit(1)


if __name__ == "__main__":
    main()
