"""
KernelBench Task Registry and Metadata

Maps task IDs to file paths, descriptions, and optimization goals.
Supports all 270 tasks across 4 levels.
"""

import os
from pathlib import Path


def _resolve_kernelbench_dataset_root() -> Path:
    """Resolve the KernelBench dataset root in an environment-agnostic way."""
    repo_root = Path(__file__).resolve().parents[2]

    env_path = os.getenv("KERNELBENCH_DATASET_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if (candidate / "level1").exists():
            return candidate
        if (candidate / "KernelBench" / "level1").exists():
            return candidate / "KernelBench"
        return candidate

    default_root = repo_root / "tasks" / "kernel_bench" / "KernelBench" / "KernelBench"
    if default_root.exists():
        return default_root

    fallback_root = repo_root / "tasks" / "kernel_bench" / "KernelBench"
    return fallback_root


# Base path to KernelBench dataset
KB_BASE = _resolve_kernelbench_dataset_root()

# Level 1: Single Kernel Operators (100 tasks)
LEVEL1_TASKS = {
    # Matrix Operations
    "1_1": {"name": "Square matrix multiplication", "category": "matmul"},
    "1_3": {"name": "Batched matrix multiplication", "category": "matmul"},
    "1_10": {"name": "3D tensor matrix multiplication", "category": "matmul"},
    "1_11": {"name": "4D tensor matrix multiplication", "category": "matmul"},

    # Activation Functions
    "1_19": {"name": "ReLU", "category": "activation"},
    "1_20": {"name": "LeakyReLU", "category": "activation"},
    "1_21": {"name": "Sigmoid", "category": "activation"},
    "1_22": {"name": "Tanh", "category": "activation"},
    "1_23": {"name": "Softmax", "category": "activation"},
    "1_24": {"name": "LogSoftmax", "category": "activation"},
    "1_25": {"name": "Swish", "category": "activation"},
    "1_26": {"name": "GELU", "category": "activation"},
    "1_27": {"name": "SELU", "category": "activation"},

    # Normalization
    "1_33": {"name": "BatchNorm", "category": "normalization"},
    "1_34": {"name": "InstanceNorm", "category": "normalization"},
    "1_35": {"name": "GroupNorm", "category": "normalization"},
    "1_36": {"name": "RMSNorm", "category": "normalization"},
    "1_40": {"name": "LayerNorm", "category": "normalization"},

    # Pooling
    "1_41": {"name": "Max Pooling 1D", "category": "pooling"},
    "1_42": {"name": "Max Pooling 2D", "category": "pooling"},
    "1_43": {"name": "Max Pooling 3D", "category": "pooling"},
    "1_44": {"name": "Average Pooling 1D", "category": "pooling"},
    "1_45": {"name": "Average Pooling 2D", "category": "pooling"},
    "1_46": {"name": "Average Pooling 3D", "category": "pooling"},

    # Convolutions
    "1_50": {"name": "Conv1D", "category": "convolution"},
    "1_51": {"name": "Conv2D", "category": "convolution"},
    "1_52": {"name": "Conv3D", "category": "convolution"},
    "1_65": {"name": "DepthwiseConv2D", "category": "convolution"},
    "1_71": {"name": "ConvTranspose2D", "category": "convolution"},

    # Loss Functions
    "1_84": {"name": "CrossEntropyLoss", "category": "loss"},
    "1_88": {"name": "BCELoss", "category": "loss"},
    "1_91": {"name": "MSELoss", "category": "loss"},
}

# Level 2: Fusion Patterns (100 tasks)
LEVEL2_TASKS = {
    "2_1": {"name": "Conv2D + ReLU + BiasAdd", "category": "conv_fusion"},
    "2_4": {"name": "Conv2d + Mish + Mish", "category": "conv_fusion"},
    "2_9": {"name": "Matmul + Subtract + Multiply + ReLU", "category": "matmul_fusion"},
    "2_12": {"name": "Gemm + Multiply + LeakyReLU", "category": "matmul_fusion"},
    "2_22": {"name": "Matmul + Scale + ResidualAdd + Clamp + LogSumExp + Mish", "category": "matmul_fusion"},
    "2_28": {"name": "BMM + InstanceNorm + Sum + ResidualAdd + Multiply", "category": "complex_fusion"},
    "2_52": {"name": "Conv2d + Activation + BatchNorm", "category": "conv_fusion"},
    "2_73": {"name": "Conv2d + BatchNorm + Scaling", "category": "conv_fusion"},
}

# Level 3: Full Architectures (50 tasks)
LEVEL3_TASKS = {
    "3_1": {"name": "MLP", "category": "basic"},
    "3_5": {"name": "AlexNet", "category": "cnn"},
    "3_8": {"name": "ResNetBasicBlock", "category": "cnn"},
    "3_9": {"name": "ResNet18", "category": "cnn"},
    "3_10": {"name": "ResNet101", "category": "cnn"},
    "3_11": {"name": "VGG16", "category": "cnn"},
    "3_19": {"name": "MobileNetV1", "category": "mobile"},
    "3_20": {"name": "MobileNetV2", "category": "mobile"},
    "3_22": {"name": "EfficientNetB0", "category": "efficient"},
    "3_28": {"name": "VisionTransformer", "category": "transformer"},
    "3_30": {"name": "SwinTransformerV2", "category": "transformer"},
    "3_35": {"name": "LSTM", "category": "rnn"},
    "3_39": {"name": "GRU", "category": "rnn"},
    "3_43": {"name": "MinGPTCausalAttention", "category": "attention"},
    "3_48": {"name": "Mamba2", "category": "ssm"},
}

# Level 4: HuggingFace Models (20 tasks)
LEVEL4_TASKS = {
    "4_1": {"name": "gpt-neo-2.7B_bs1_seq32", "category": "gpt"},
    "4_2": {"name": "gpt-neo-2.7B_bs32_seq256", "category": "gpt"},
    "4_3": {"name": "gpt-neo-2.7B_bs32_seq2047", "category": "gpt"},
    "4_4": {"name": "opt-1.3b_bs1_seq256", "category": "opt"},
    "4_7": {"name": "gpt2_bs1_seq256", "category": "gpt"},
    "4_10": {"name": "bigbird-roberta_bs1_seq256", "category": "bert"},
    "4_14": {"name": "bart-large_bs32_seq256", "category": "bart"},
}

def get_task_info(task_id: str) -> dict:
    """
    Get complete task information including path and optimization goal.

    Args:
        task_id: Task identifier like "1_19" or "3_28"

    Returns:
        Dictionary with task metadata
    """
    level = int(task_id.split("_")[0])

    # Get task metadata
    if level == 1:
        tasks = LEVEL1_TASKS
        level_name = "level1"
        base_steps = 15
    elif level == 2:
        tasks = LEVEL2_TASKS
        level_name = "level2"
        base_steps = 20
    elif level == 3:
        tasks = LEVEL3_TASKS
        level_name = "level3"
        base_steps = 25
    elif level == 4:
        tasks = LEVEL4_TASKS
        level_name = "level4"
        base_steps = 30
    else:
        raise ValueError(f"Invalid level in task_id: {task_id}")

    if task_id not in tasks:
        # For tasks not explicitly listed, generate generic info
        task_num = int(task_id.split("_")[1])
        task_info = {
            "name": f"Level {level} Task {task_num}",
            "category": "generic"
        }
    else:
        task_info = tasks[task_id]

    # Construct file path
    task_num = task_id.split("_")[1]

    # Find the actual file (handling different naming patterns)
    level_dir = KB_BASE / level_name

    # Try different naming patterns
    possible_names = [
        f"{task_num}_*.py",  # Most common pattern
        f"{task_num}-*.py",  # Alternative pattern
    ]

    task_file = None
    for pattern in possible_names:
        matches = list(level_dir.glob(pattern))
        if matches:
            task_file = matches[0]
            break

    if not task_file:
        # If no pattern matches, try exact number
        exact_matches = list(level_dir.glob(f"{task_num}.py"))
        if exact_matches:
            task_file = exact_matches[0]
        else:
            # Last resort: list all files and find one starting with the number
            all_files = sorted(level_dir.glob("*.py"))
            for f in all_files:
                if f.stem.startswith(f"{task_num}_") or f.stem.startswith(f"{task_num}-") or f.stem == task_num:
                    task_file = f
                    break

    if not task_file:
        raise FileNotFoundError(f"Could not find task file for {task_id} in {level_dir}")

    # Generate optimization goal based on level
    if level == 1:
        goal = f"""Optimize the {task_info['name']} kernel for the benchmark GPU used in this run.
Focus on:
- Memory coalescing and bandwidth optimization
- Warp efficiency and divergence reduction
- Register usage optimization
- Shared memory utilization where beneficial
- Tensor Core usage for applicable operations

The kernel should maintain numerical correctness while maximizing performance."""

    elif level == 2:
        goal = f"""Optimize the fused operation {task_info['name']} for the benchmark GPU used in this run.
Key optimization opportunities:
- Kernel fusion to eliminate intermediate tensor materialization
- Reduce memory bandwidth by combining operations
- Optimize data layout for fused execution
- Consider shared memory for data reuse
- Minimize kernel launch overhead

Focus on creating a single, efficient fused kernel that performs all operations."""

    elif level == 3:
        goal = f"""Optimize the complete {task_info['name']} architecture for the benchmark GPU used in this run.
Optimization strategies:
- Global kernel fusion opportunities across the architecture
- Optimize critical path operations (e.g., attention, convolutions)
- Consider algorithmic improvements (e.g., FlashAttention for transformers)
- Memory-efficient implementations for large tensors
- Architecture-specific optimizations

Look for opportunities to fuse multiple layers or operations while maintaining the model's functionality."""

    elif level == 4:
        goal = f"""Optimize the HuggingFace model {task_info['name']} for the benchmark GPU used in this run.
Focus areas:
- Attention mechanism optimization (if applicable)
- Efficient handling of large sequence lengths
- Memory-efficient implementations for large models
- Optimize embedding and output projections
- Consider advanced techniques like FlashAttention, online softmax

This is a production model, so maintain exact numerical behavior while maximizing throughput."""

    return {
        "task_id": task_id,
        "level": level,
        "name": task_info["name"],
        "category": task_info["category"],
        "file_path": str(task_file),
        "goal": goal,
        "suggested_steps": base_steps,
    }


def get_task_info_with_code(task_id: str) -> dict:
    """
    Get complete task information including the original code and rich goal.

    Args:
        task_id: Task identifier like "1_19" or "3_28"

    Returns:
        Dictionary with task metadata, original code, and rich goal
    """
    # Get basic task info
    info = get_task_info(task_id)

    # Load the original task code
    with open(info['file_path'], 'r') as f:
        task_code = f.read()

    # Import the rich prompt generator
    from kernelbench_prompts import generate_rich_goal, load_example_snippets

    # Load examples for this level
    examples = load_example_snippets(info['level'])

    # Generate rich goal with full context
    info['original_code'] = task_code
    info['goal'] = generate_rich_goal(
        task_id=task_id,
        task_name=info['name'],
        task_code=task_code,
        level=info['level'],
        category=info['category'],
        examples=examples
    )

    # Add evaluation command
    info['eval_command'] = f"python evaluate_gpu.py --task-id {task_id} --solution-path optimize.py --device cuda"

    return info


def list_tasks_by_level(level: int) -> list:
    """List all task IDs for a given level."""
    if level == 1:
        base_tasks = list(LEVEL1_TASKS.keys())
        # Add remaining Level 1 tasks (up to 100)
        all_tasks = base_tasks + [f"1_{i}" for i in range(1, 101) if f"1_{i}" not in base_tasks]
    elif level == 2:
        base_tasks = list(LEVEL2_TASKS.keys())
        # Add remaining Level 2 tasks (up to 100)
        all_tasks = base_tasks + [f"2_{i}" for i in range(1, 101) if f"2_{i}" not in base_tasks]
    elif level == 3:
        base_tasks = list(LEVEL3_TASKS.keys())
        # Add remaining Level 3 tasks (up to 50)
        all_tasks = base_tasks + [f"3_{i}" for i in range(1, 51) if f"3_{i}" not in base_tasks]
    elif level == 4:
        base_tasks = list(LEVEL4_TASKS.keys())
        # Add remaining Level 4 tasks (up to 20)
        all_tasks = base_tasks + [f"4_{i}" for i in range(1, 21) if f"4_{i}" not in base_tasks]
    else:
        raise ValueError(f"Invalid level: {level}")

    return sorted(all_tasks, key=lambda x: int(x.split("_")[1]))


def get_all_tasks() -> list[str]:
    """Return the full KernelBench task inventory."""
    tasks: list[str] = []
    for level in (1, 2, 3, 4):
        tasks.extend(list_tasks_by_level(level))
    return tasks


def get_all_categories() -> list[str]:
    """Return the known category names from the curated metadata."""
    categories = {
        info["category"]
        for group in (LEVEL1_TASKS, LEVEL2_TASKS, LEVEL3_TASKS, LEVEL4_TASKS)
        for info in group.values()
    }
    categories.add("generic")
    return sorted(categories)

def get_representative_subset() -> list:
    """Get a representative subset of tasks for quick testing."""
    return [
        # Level 1 - Core operations
        "1_1",   # Matrix multiplication
        "1_19",  # ReLU
        "1_23",  # Softmax
        "1_26",  # GELU
        "1_40",  # LayerNorm
        "1_51",  # Conv2D

        # Level 2 - Fusion patterns
        "2_1",   # Conv2D + ReLU + BiasAdd
        "2_9",   # Matmul fusion chain

        # Level 3 - Architectures
        "3_9",   # ResNet18
        "3_28",  # Vision Transformer
        "3_43",  # MinGPT Attention

        # Level 4 - Large models
        "4_2",   # GPT-Neo 2.7B
    ]

if __name__ == "__main__":
    # Test the registry
    print("Testing KernelBench Task Registry\n")

    # Test getting info for a specific task
    task_info = get_task_info("1_19")
    print(f"Task 1_19 Info:")
    print(f"  Name: {task_info['name']}")
    print(f"  File: {task_info['file_path']}")
    print(f"  Category: {task_info['category']}")
    print(f"  Suggested Steps: {task_info['suggested_steps']}")
    print()

    # List tasks by level
    level1_tasks = list_tasks_by_level(1)
    print(f"Level 1 has {len(level1_tasks)} tasks")
    print(f"First 5: {level1_tasks[:5]}")
    print()

    # Get representative subset
    subset = get_representative_subset()
    print(f"Representative subset ({len(subset)} tasks):")
    for task_id in subset:
        info = get_task_info(task_id)
        print(f"  {task_id}: {info['name']}")
