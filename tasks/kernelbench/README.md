# KernelBench Integration for AIDE

This directory contains the integration of KernelBench tasks into the AIDE distributed optimization system.

## Overview

KernelBench is a comprehensive benchmark suite containing 270 GPU kernel optimization tasks across 4 difficulty levels:
- **Level 1**: 100 single kernel operators (matmul, activations, convolutions, etc.)
- **Level 2**: 100 fusion patterns (combining 2-5 operations)
- **Level 3**: 50 full architectures (ResNet, VisionTransformer, LSTM, etc.)
- **Level 4**: 20 HuggingFace pre-trained models

## Files

- `kb_tasks.py` - Task registry and metadata for all 270 tasks
- `evaluate_gpu.py` - Evaluation script that interfaces with KernelBench
- `contract_gpu.yaml` - Single flexible contract for all tasks
- `optimize.py` - Placeholder file that AIDE modifies with optimized code

## Usage

### Running Individual Tasks

Run these commands from the repository root:

```bash
# Run a Level 1 task (ReLU activation)
./cli/aide-run --task kernelbench --kb-task 1_19 --num-experiments 16

# Run a Level 2 fusion task with more steps
./cli/aide-run --task kernelbench --kb-task 2_1 --steps 25

# Run a Level 3 architecture (Vision Transformer)
./cli/aide-run --task kernelbench --kb-task 3_28 --steps 30 --gpu-fraction 1.0

# Run with specific model
./cli/aide-run --task kernelbench --kb-task 1_23 --model gpt-4
```

### Task ID Format

Task IDs follow the pattern: `{level}_{number}`
- Level 1: `1_1` to `1_100`
- Level 2: `2_1` to `2_100`
- Level 3: `3_1` to `3_50`
- Level 4: `4_1` to `4_20`

### Running Task Sequences

Use the helper script to run multiple tasks:

```bash
# Run a quick test (5 representative tasks)
./cli/run-kb-sequence quick

# Run important Level 1 tasks
./cli/run-kb-sequence level1

# Run custom task list
./cli/run-kb-sequence 1_1 1_19 1_23 2_1

# Run with custom settings
GPU_FRACTION=0.25 NUM_EXPERIMENTS=32 ./cli/run-kb-sequence matmul
```

### Available Presets

- `quick` - 5 representative Level 1 tasks
- `level1` - 10 important Level 1 tasks
- `level2` - 5 fusion tasks from Level 2
- `level3` - 5 architecture tasks from Level 3
- `matmul` - All matrix multiplication variants
- `activations` - All activation functions
- `test` - Just 2 tasks for testing

## Important Tasks by Category

### Matrix Operations (Level 1)
- `1_1` - Square matrix multiplication
- `1_3` - Batched matrix multiplication
- `1_10` - 3D tensor matrix multiplication

### Activations (Level 1)
- `1_19` - ReLU
- `1_23` - Softmax
- `1_26` - GELU

### Normalization (Level 1)
- `1_33` - BatchNorm
- `1_40` - LayerNorm
- `1_36` - RMSNorm

### Convolutions (Level 1)
- `1_51` - Conv2D
- `1_65` - DepthwiseConv2D

### Fusion Patterns (Level 2)
- `2_1` - Conv2D + ReLU + BiasAdd
- `2_9` - Matmul + Subtract + Multiply + ReLU

### Architectures (Level 3)
- `3_9` - ResNet18
- `3_28` - Vision Transformer
- `3_43` - MinGPT Causal Attention
- `3_35` - LSTM

### Large Models (Level 4)
- `4_1` - GPT-Neo 2.7B (small batch)
- `4_2` - GPT-Neo 2.7B (medium batch)

## Metrics

The primary metric is **speedup** - the ratio of original PyTorch execution time to optimized kernel time.
- `speedup > 1.0` means faster than PyTorch
- `speedup = 2.0` means 2x faster
- `speedup < 1.0` means slower (optimization failed)

## Tips for Optimization

1. **Start with Level 1** - Single kernels are easier to optimize
2. **Level 2 focuses on fusion** - Combine operations to reduce memory traffic
3. **Level 3 needs global thinking** - Consider the entire architecture
4. **Level 4 requires advanced techniques** - FlashAttention, online softmax, etc.

## Monitoring Progress

If you enable `AIDE_ENABLE_MLFLOW=1`, KernelBench runs are logged to MLflow:
- Speedup metrics for each iteration
- Code artifacts for generated kernels
- Comparison across different tasks inside one experiment

## Troubleshooting

If evaluation fails:
1. Check that the KernelBench dataset exists at `tasks/kernel_bench/KernelBench/`
2. Verify CUDA is available: `nvidia-smi`
3. Check the generated code in `optimize.py` for syntax errors
4. Look at evaluation output for specific error messages

## Next Steps

1. Start with the `test` preset to verify everything works
2. Run the `quick` preset for initial results
3. Progress through levels systematically
4. Focus on specific categories based on your needs
