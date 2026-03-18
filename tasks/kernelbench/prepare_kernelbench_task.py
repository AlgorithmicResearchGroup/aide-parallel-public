#!/usr/bin/env python3
"""
Prepare KernelBench tasks for AIDE optimization.

This script loads the original KernelBench task code and creates a properly
structured optimize.py file that AIDE can work with, including the original
Model class and instructions for creating ModelNew.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Tuple, Optional


def _kernelbench_root() -> Path:
    """Resolve KernelBench repository root."""
    repo_root = Path(__file__).resolve().parents[2]
    env_root = os.getenv("KERNELBENCH_PATH")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return repo_root / "tasks" / "kernel_bench" / "KernelBench"


def extract_imports(code: str) -> str:
    """Extract all import statements from the code."""
    lines = code.split('\n')
    imports = []

    for line in lines:
        # Get import statements
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            imports.append(line)
        # Stop at first non-import, non-comment line
        elif line.strip() and not line.strip().startswith('#'):
            if not any(keyword in line for keyword in ['import', 'from']):
                break

    return '\n'.join(imports)


def extract_class_and_functions(code: str) -> Tuple[str, str, str]:
    """
    Extract the Model class, helper functions, and global variables.
    Returns: (model_class_code, helper_functions, global_vars)
    """
    tree = ast.parse(code)

    model_class = None
    functions = []
    globals_lines = []

    # Track line numbers for proper extraction
    lines = code.split('\n')

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Model':
            # Extract Model class with proper indentation
            start_line = node.lineno - 1
            end_line = node.end_lineno
            model_class = '\n'.join(lines[start_line:end_line])

        elif isinstance(node, ast.FunctionDef):
            # Skip get_inputs and get_init_inputs (we'll handle separately)
            if node.name not in ['get_inputs', 'get_init_inputs']:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                func_code = '\n'.join(lines[start_line:end_line])
                functions.append(func_code)

    # Extract global variables (batch_size, dimensions, etc.)
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            # Look for variable assignments
            if '=' in stripped and not any(keyword in stripped for keyword in
                ['import', 'from', 'class', 'def', 'return', 'if', 'for', 'while']):
                # This might be a global variable
                if any(var in stripped for var in ['batch_size', 'dim', 'size', 'length',
                                                   'channels', 'hidden', 'vocab', 'sequence']):
                    globals_lines.append(line)

    model_code = model_class if model_class else ""
    helper_code = '\n\n'.join(functions) if functions else ""
    global_vars = '\n'.join(globals_lines) if globals_lines else ""

    return model_code, helper_code, global_vars


def extract_input_specs(code: str) -> Tuple[str, str]:
    """Extract get_inputs and get_init_inputs functions."""
    tree = ast.parse(code)
    lines = code.split('\n')

    get_inputs_code = ""
    get_init_inputs_code = ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == 'get_inputs':
                start_line = node.lineno - 1
                end_line = node.end_lineno
                get_inputs_code = '\n'.join(lines[start_line:end_line])
            elif node.name == 'get_init_inputs':
                start_line = node.lineno - 1
                end_line = node.end_lineno
                get_init_inputs_code = '\n'.join(lines[start_line:end_line])

    return get_inputs_code, get_init_inputs_code


def load_example_for_level(level: int) -> Tuple[str, str]:
    """Load a relevant example for the given level."""
    examples_dir = _kernelbench_root() / "src" / "prompts"

    # Choose example based on level
    if level == 1:
        # Simple kernel example
        original_example = examples_dir / "model_ex_1.py"
        optimized_example = examples_dir / "model_new_ex_1.py"
    elif level == 2:
        # Fusion example
        original_example = examples_dir / "model_ex_2.py"
        optimized_example = examples_dir / "model_new_ex_2.py"
    else:
        # Default to first example
        original_example = examples_dir / "model_ex_1.py"
        optimized_example = examples_dir / "model_new_ex_1.py"

    original_code = ""
    optimized_code = ""

    if original_example.exists():
        with open(original_example, 'r') as f:
            original_code = f.read()

    if optimized_example.exists():
        with open(optimized_example, 'r') as f:
            optimized_code = f.read()

    return original_code, optimized_code


def generate_optimize_template(
    task_id: str,
    task_name: str,
    imports: str,
    model_code: str,
    helper_code: str,
    global_vars: str,
    get_inputs: str,
    get_init_inputs: str,
    example_original: str,
    example_optimized: str,
    level: int
) -> str:
    """Generate the complete optimize.py template with all context."""

    # Level-specific optimization hints
    if level == 1:
        optimization_hints = """
# Level 1 Optimization Strategies:
# - Replace PyTorch operations with custom CUDA kernels
# - Optimize memory access patterns (coalescing)
# - Use shared memory for data reuse
# - Leverage tensor cores for matrix operations
# - Minimize memory transfers between kernels
"""
    elif level == 2:
        optimization_hints = """
# Level 2 Optimization Strategies:
# - FUSE multiple operations into single kernels
# - Eliminate intermediate tensor materializations
# - Combine memory reads/writes across operations
# - Reduce kernel launch overhead
# - Create compound operations (e.g., Conv2D+ReLU+BiasAdd in one kernel)
"""
    elif level == 3:
        optimization_hints = """
# Level 3 Optimization Strategies:
# - Optimize entire architecture components
# - Fuse operations across layers
# - Implement architecture-specific optimizations (FlashAttention, etc.)
# - Optimize memory layout for the full model
# - Consider algorithmic improvements
"""
    else:
        optimization_hints = """
# Level 4 Optimization Strategies:
# - Optimize large transformer models
# - Implement memory-efficient attention (FlashAttention, online softmax)
# - Optimize embedding and projection layers
# - Handle large sequence lengths efficiently
# - Consider model-specific optimizations
"""

    template = f'''"""
KernelBench Task: {task_id} - {task_name}
Level: {level}

This file contains the original Model class that needs to be optimized.
Your task is to create a ModelNew class that:
1. Has the EXACT same interface (__init__ and forward signatures)
2. Produces outputs within the official KernelBench tolerance (rtol=1e-2, atol=1e-2)
3. Runs faster than the original on GPU

IMPORTANT: You must create a class called "ModelNew" (not "Model")
"""

# Original imports from the task
{imports}

# Additional imports you may need for optimization
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Global variables from original task
{global_vars}

# Helper functions from original task
{helper_code}

# Input/Init specifications from original task
{get_inputs}

{get_init_inputs}

###############################################################################
# ORIGINAL MODEL (DO NOT MODIFY - FOR REFERENCE ONLY)
###############################################################################

{model_code}

###############################################################################
# EXAMPLE OPTIMIZATION PATTERN
###############################################################################

# Here's an example of optimization from a similar task:

# === Original Example ===
"""
{example_original}
"""

# === Optimized Example ===
"""
{example_optimized}
"""

###############################################################################
# YOUR OPTIMIZED IMPLEMENTATION
###############################################################################

{optimization_hints}

# TODO: Create your optimized ModelNew class below
# Remember:
# 1. Keep the same __init__ signature as Model
# 2. Keep the same forward signature as Model
# 3. Maintain numerical accuracy within the official KernelBench tolerance (rtol=1e-2, atol=1e-2)
# 4. Focus on GPU performance optimization

class ModelNew(nn.Module):
    """Optimized version of Model for {task_name}"""

    def __init__(self, *args, **kwargs):
        """Initialize with same signature as Model.__init__"""
        super().__init__()
        # TODO: Implement optimized initialization
        # You can copy from Model and modify, or completely reimplement
        pass

    def forward(self, *args, **kwargs):
        """Forward pass with same signature as Model.forward"""
        # TODO: Implement optimized forward pass
        # This is where your main optimizations should go
        pass

# Optional: Add custom CUDA kernels here
# Example structure:
#
# cuda_source = \'\'\'
# #include <torch/extension.h>
# #include <cuda_runtime.h>
#
# __global__ void custom_kernel(...) {{
#     // Your CUDA kernel implementation
# }}
#
# torch::Tensor custom_op(torch::Tensor input) {{
#     // Launch kernel and return result
# }}
# \'\'\'
#
# cpp_source = \'\'\'
# torch::Tensor custom_op(torch::Tensor input);
# \'\'\'
#
# custom_module = load_inline(
#     name='custom_ops',
#     cpp_sources=cpp_source,
#     cuda_sources=cuda_source,
#     functions=['custom_op'],
#     verbose=True
# )
'''

    return template


def prepare_task(task_id: str, output_path: str, kb_tasks_path: str = None):
    """
    Main function to prepare a KernelBench task for AIDE optimization.

    Args:
        task_id: KernelBench task identifier (e.g., "1_19")
        output_path: Path to write the prepared optimize.py file
        kb_tasks_path: Path to kb_tasks.py module
    """
    # Import kb_tasks to get task info
    import sys
    if kb_tasks_path:
        sys.path.insert(0, str(Path(kb_tasks_path).parent))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    from kb_tasks import get_task_info

    # Get task information
    task_info = get_task_info(task_id)

    print(f"Preparing task: {task_info['name']} (Level {task_info['level']})")
    print(f"Task file: {task_info['file_path']}")

    # Load the original task code
    with open(task_info['file_path'], 'r') as f:
        original_code = f.read()

    # Extract components
    imports = extract_imports(original_code)
    model_code, helper_code, global_vars = extract_class_and_functions(original_code)
    get_inputs_code, get_init_inputs_code = extract_input_specs(original_code)

    # Load examples for this level
    example_original, example_optimized = load_example_for_level(task_info['level'])

    # Generate the template
    optimize_content = generate_optimize_template(
        task_id=task_id,
        task_name=task_info['name'],
        imports=imports,
        model_code=model_code,
        helper_code=helper_code,
        global_vars=global_vars,
        get_inputs=get_inputs_code,
        get_init_inputs=get_init_inputs_code,
        example_original=example_original,
        example_optimized=example_optimized,
        level=task_info['level']
    )

    # Write to output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(optimize_content)

    print(f"✓ Task prepared and written to: {output_path}")
    print(f"  - Original Model class included: {'✓' if model_code else '✗'}")
    print(f"  - Helper functions included: {'✓' if helper_code else '✗'}")
    print(f"  - Examples included: {'✓' if example_original else '✗'}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare KernelBench task for AIDE")
    parser.add_argument("task_id", help="KernelBench task ID (e.g., 1_19)")
    parser.add_argument("-o", "--output", default="optimize.py",
                       help="Output file path (default: optimize.py)")
    args = parser.parse_args()

    prepare_task(args.task_id, args.output)
