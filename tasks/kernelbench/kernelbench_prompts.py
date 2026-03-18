#!/usr/bin/env python3
"""
Rich prompt generation for KernelBench tasks.

This module generates comprehensive, contextual prompts for AIDE that include:
- Original code
- Examples
- Specifications
- Level-specific optimization strategies
"""

import ast
import os
from pathlib import Path
from typing import Dict, Any, Optional, List


def _kernelbench_root() -> Path:
    """Resolve KernelBench repository root."""
    repo_root = Path(__file__).resolve().parents[2]
    env_root = os.getenv("KERNELBENCH_PATH")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return repo_root / "tasks" / "kernel_bench" / "KernelBench"


def load_example_snippets(level: int) -> Dict[str, str]:
    """Load relevant example snippets for the given level."""
    examples_dir = _kernelbench_root() / "src" / "prompts"
    examples = {}

    # Try to load examples
    try:
        # Load a simple example for all levels
        ex1_path = examples_dir / "model_ex_1.py"
        ex1_new_path = examples_dir / "model_new_ex_1.py"

        if ex1_path.exists():
            with open(ex1_path, 'r') as f:
                examples['simple_original'] = f.read()

        if ex1_new_path.exists():
            with open(ex1_new_path, 'r') as f:
                examples['simple_optimized'] = f.read()

        # Load chain-of-thought examples if available
        cot_dir = examples_dir / "cot"
        if cot_dir.exists():
            cot_files = list(cot_dir.glob("*.py"))
            if cot_files:
                with open(cot_files[0], 'r') as f:
                    examples['cot_example'] = f.read()

    except Exception as e:
        print(f"Warning: Could not load examples: {e}")

    return examples


def extract_model_signature(code: str) -> Dict[str, str]:
    """Extract the Model class signatures for __init__ and forward methods."""
    try:
        tree = ast.parse(code)
        signatures = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Model':
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == '__init__':
                            args = [arg.arg for arg in item.args.args]
                            signatures['init'] = f"__init__({', '.join(args)})"
                        elif item.name == 'forward':
                            args = [arg.arg for arg in item.args.args]
                            signatures['forward'] = f"forward({', '.join(args)})"

        return signatures
    except:
        return {'init': '__init__(self, ...)', 'forward': 'forward(self, x, ...)'}


def generate_rich_goal(
    task_id: str,
    task_name: str,
    task_code: str,
    level: int,
    category: str,
    examples: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate a comprehensive goal/prompt for AIDE that includes all necessary context.

    Args:
        task_id: Task identifier (e.g., "1_19")
        task_name: Human-readable task name (e.g., "ReLU")
        task_code: The complete original task Python code
        level: Task level (1-4)
        category: Task category (e.g., "activation", "convolution")
        examples: Optional dictionary of example code snippets

    Returns:
        A rich, contextual goal string for AIDE
    """

    # Extract signatures from the task code
    signatures = extract_model_signature(task_code)

    # Level-specific optimization strategies
    level_strategies = {
        1: """
LEVEL 1 OPTIMIZATION STRATEGIES (Single Kernels):
- Replace PyTorch ops with custom CUDA kernels
- Optimize memory access patterns for coalescing
- Use shared memory for frequently accessed data
- Leverage tensor cores for applicable operations (matmul, conv)
- Minimize global memory reads/writes
- Use appropriate block and grid dimensions
- Consider warp-level primitives
""",
        2: """
LEVEL 2 OPTIMIZATION STRATEGIES (Operator Fusion):
- FUSE multiple operations into single kernels
- Eliminate intermediate tensor allocations
- Combine elementwise operations
- Merge normalization with other ops
- Reduce kernel launch overhead
- Share memory loads across fused operations
- Example: Conv2D + ReLU + BiasAdd should be one kernel
""",
        3: """
LEVEL 3 OPTIMIZATION STRATEGIES (Full Architectures):
- Optimize critical path operations
- Implement architecture-specific optimizations
- For attention: Consider FlashAttention, online softmax
- For CNNs: Fuse conv-bn-relu patterns
- For RNNs: Optimize gate computations
- Global memory management across layers
- Consider graph-level optimizations
""",
        4: """
LEVEL 4 OPTIMIZATION STRATEGIES (Large Models):
- Optimize transformer attention mechanisms
- Implement memory-efficient algorithms (FlashAttention)
- Optimize large matrix multiplications
- Handle long sequences efficiently
- Optimize embedding and output projections
- Consider model-specific patterns
- Memory bandwidth optimization for large tensors
"""
    }

    # Category-specific hints
    category_hints = {
        "matmul": "Consider tensor cores, tiling, shared memory for tiles",
        "activation": "Simple elementwise - focus on memory bandwidth",
        "convolution": "Use cudnn or custom tiled implementation",
        "normalization": "Fuse with adjacent operations, optimize reductions",
        "attention": "Consider FlashAttention, online softmax, KV cache optimization",
        "pooling": "Optimize memory access patterns, consider fusion",
        "loss": "Fuse computation with gradient calculation if possible",
    }

    strategy = level_strategies.get(level, level_strategies[1])
    category_hint = category_hints.get(category, "")

    # Build the comprehensive goal
    goal = f"""
================================================================================
KERNELBENCH OPTIMIZATION TASK: {task_id} - {task_name}
Level: {level} | Category: {category}
================================================================================

YOUR OBJECTIVE:
Create an optimized GPU implementation of the Model class below. You must create
a new class called "ModelNew" that:
1. Has EXACTLY the same interface (same __init__ and forward signatures)
2. Produces numerically equivalent outputs within the official KernelBench tolerance (rtol=1e-2, atol=1e-2)
3. Runs FASTER on GPU (maximize speedup while maintaining correctness)

================================================================================
ORIGINAL MODEL CLASS TO OPTIMIZE:
================================================================================

{task_code}

================================================================================
INTERFACE REQUIREMENTS:
================================================================================

Your ModelNew class MUST have these exact signatures:
- {signatures.get('init', '__init__(self, ...)')}
- {signatures.get('forward', 'forward(self, x, ...)')}

The inputs and outputs must be tensor-compatible with the original Model.

================================================================================
OPTIMIZATION STRATEGY FOR THIS LEVEL:
================================================================================
{strategy}

{f"Category-Specific Hint: {category_hint}" if category_hint else ""}

================================================================================
PERFORMANCE METRICS:
================================================================================

Your optimization will be evaluated on:
1. CORRECTNESS: Output must match within the official KernelBench tolerance (rtol=1e-2, atol=1e-2)
2. SPEEDUP: Time(Original) / Time(Optimized) - higher is better
   - Speedup > 1.0 means your optimization is faster (good)
   - Speedup = 2.0 means 2x faster (excellent)
   - Speedup < 1.0 means slower (optimization failed)

Target speedups by level:
- Level 1: 1.5x - 3x typical
- Level 2: 2x - 5x typical (fusion benefit)
- Level 3: 1.5x - 4x typical (architecture-dependent)
- Level 4: 1.2x - 2x typical (already optimized baselines)

"""

    # Add examples if available
    if examples and 'simple_original' in examples:
        goal += """
================================================================================
EXAMPLE OPTIMIZATION PATTERN:
================================================================================

Here's an example showing how to optimize a simple operation:

--- Original PyTorch Implementation ---
```python
class Model(nn.Module):
    def forward(self, x, y):
        return x + y  # Simple addition
```

--- Optimized CUDA Implementation ---
```python
from torch.utils.cpp_extension import load_inline

cuda_source = '''
#include <torch/extension.h>

__global__ void add_kernel(float* x, float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] + y[idx];
    }
}

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto output = torch::empty_like(x);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
'''

add_module = load_inline(
    name='add_cuda',
    cpp_sources='torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y);',
    cuda_sources=cuda_source,
    functions=['add_cuda']
)

class ModelNew(nn.Module):
    def forward(self, x, y):
        return add_module.add_cuda(x, y)
```

"""

    goal += """
================================================================================
IMPLEMENTATION REQUIREMENTS:
================================================================================

1. Create a class named exactly "ModelNew" (not "Model" or anything else)
2. Import all necessary modules (torch, nn, load_inline for CUDA, etc.)
3. Preserve the exact same interface as the original Model
4. You may use:
   - Custom CUDA kernels (recommended for best performance)
   - PyTorch JIT compilation
   - Optimized PyTorch operations
   - CuDNN operations
   - Any valid optimization technique

5. Common patterns to consider:
   - Kernel fusion (combine multiple ops)
   - Memory coalescing (optimize access patterns)
   - Shared memory usage (for data reuse)
   - Tensor core utilization (for matmul/conv)
   - Reduction optimization (for normalization)

================================================================================
START YOUR IMPLEMENTATION:
================================================================================

Remember: The goal is to create ModelNew with the same interface but better GPU
performance. Focus on the specific operations in the Model and apply appropriate
optimization techniques for maximum speedup.
"""

    return goal


def generate_enhanced_task_info(task_id: str) -> Dict[str, Any]:
    """
    Generate enhanced task information including the original code and rich goals.

    Args:
        task_id: KernelBench task identifier

    Returns:
        Enhanced task information dictionary
    """
    # Import kb_tasks
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kb_tasks import get_task_info

    # Get basic task info
    task_info = get_task_info(task_id)

    # Load the original task code
    with open(task_info['file_path'], 'r') as f:
        task_code = f.read()

    # Load examples for this level
    examples = load_example_snippets(task_info['level'])

    # Generate rich goal with code
    task_info['original_code'] = task_code
    task_info['rich_goal'] = generate_rich_goal(
        task_id=task_id,
        task_name=task_info['name'],
        task_code=task_code,
        level=task_info['level'],
        category=task_info['category'],
        examples=examples
    )

    # Add evaluation command
    task_info['eval_command'] = f"python evaluate_gpu.py --task-id {task_id} --solution-path optimize.py --device cuda"

    return task_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate rich prompts for KernelBench tasks")
    parser.add_argument("task_id", help="KernelBench task ID (e.g., 1_19)")
    args = parser.parse_args()

    # Generate and print enhanced task info
    info = generate_enhanced_task_info(args.task_id)

    print("=" * 80)
    print(f"Task: {info['name']} ({args.task_id})")
    print(f"Level: {info['level']}")
    print(f"Category: {info['category']}")
    print("=" * 80)
    print("\nRich Goal Generated (preview):")
    print("-" * 40)
    print(info['rich_goal'][:1000] + "..." if len(info['rich_goal']) > 1000 else info['rich_goal'])
    print("-" * 40)
    print(f"\nEvaluation Command: {info['eval_command']}")
