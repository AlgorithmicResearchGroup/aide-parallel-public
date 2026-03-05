#!/usr/bin/env python3
"""
Quick script to show what AIDE sees for a KernelBench task.

Usage:
    python show_aide_context.py <task_id>
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def show_context(task_id: str):
    """Show the key context AIDE receives for a task."""

    from kb_tasks import get_task_info_with_code

    # Get the enhanced task info
    info = get_task_info_with_code(task_id)

    print("\n" + "=" * 80)
    print(f"AIDE CONTEXT FOR TASK: {task_id} - {info['name']}")
    print("=" * 80)

    print(f"\nLevel: {info['level']}")
    print(f"Category: {info['category']}")
    print(f"Steps: {info['suggested_steps']}")
    print(f"Evaluation: {info['eval_command']}")

    print("\n" + "-" * 80)
    print("WHAT AIDE SEES (First 2000 chars of prompt):")
    print("-" * 80)

    prompt = info['goal']
    print(prompt[:2000])

    if len(prompt) > 2000:
        print(f"\n... [{len(prompt) - 2000} more characters] ...")

    print("\n" + "-" * 80)
    print("ORIGINAL MODEL CODE AIDE NEEDS TO OPTIMIZE:")
    print("-" * 80)

    # Extract just the Model class from the original code
    code_lines = info['original_code'].split('\n')
    in_model = False
    model_lines = []

    for line in code_lines:
        if 'class Model' in line:
            in_model = True
        if in_model:
            model_lines.append(line)
            if line.strip() and not line.startswith(' ') and not line.startswith('\t') and in_model:
                # Likely end of class
                if 'def get_' in line or 'batch_size' in line:
                    break

    print('\n'.join(model_lines[:30]))

    print("\n" + "-" * 80)
    print("KEY REQUIREMENTS AIDE MUST FOLLOW:")
    print("-" * 80)
    print("1. Create a class named 'ModelNew' (not 'Model')")
    print("2. ModelNew must have same __init__ and forward signatures")
    print("3. Output must match within tolerance: rtol=1e-3, atol=1e-5")
    print("4. Optimize for GPU performance (maximize speedup)")
    print(f"5. Target speedup for Level {info['level']}: ", end="")

    if info['level'] == 1:
        print("1.5x - 3x")
    elif info['level'] == 2:
        print("2x - 5x (fusion benefits)")
    elif info['level'] == 3:
        print("1.5x - 4x")
    else:
        print("1.2x - 2x")

    print("\n" + "-" * 80)
    print("GRADING:")
    print("-" * 80)
    print("speedup = original_time / optimized_time")
    print("  - speedup > 1.0 = faster (good)")
    print("  - speedup = 2.0 = 2x faster")
    print("  - speedup < 1.0 = slower (bad)")
    print("\nEvaluation runs both models and measures:")
    print("  1. Correctness (outputs match)")
    print("  2. Performance (GPU timing)")
    print("  3. Returns speedup metric to AIDE")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_aide_context.py <task_id>")
        print("Example: python show_aide_context.py 1_19")
        sys.exit(1)

    show_context(sys.argv[1])