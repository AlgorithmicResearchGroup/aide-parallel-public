# Example Commands for Running AIDE Experiments

## Quick Start Examples

### 1. Basic Run with Default Settings
```bash
# Run attention task with default GPU configuration
./run_distributed.sh --task attention

# Run kernel optimization task
./run_distributed.sh --task kernel
```

### 2. Using Fractional GPUs for Higher Throughput
```bash
# Run 32 experiments (2 per GPU) on 16 H100s
./run_distributed.sh --gpu-fraction 0.5 --num-experiments 32

# Run 64 experiments (4 per GPU) - more memory efficient
./run_distributed.sh --gpu-fraction 0.25 --num-experiments 64

# Run 160 experiments (10 per GPU) - for lightweight experiments
./run_distributed.sh --gpu-fraction 0.1 --num-experiments 160
```

### 3. Custom W&B Project Names
```bash
# Set custom project name for organization
./run_distributed.sh --wandb-project aide-attention-v2

# Kernel task with custom project
./run_distributed.sh --task kernel --wandb-project aide-kernel-optimization
```

### 4. Override Model Settings (NEW)
```bash
# Use GPT-4 for both generation and feedback
./run_distributed.sh --model gpt-4 --feedback-model gpt-4

# Use different models for generation and feedback
./run_distributed.sh --model gpt-4 --feedback-model claude-3-opus-20240229

# Use o1-mini for faster experimentation
./run_distributed.sh --model o1-mini --feedback-model o1-mini
```

### 5. Complete Examples with All Settings

#### Attention Task with GPT-4
```bash
./run_distributed.sh \
    --task attention \
    --gpu-fraction 0.5 \
    --num-experiments 32 \
    --num-iterations 4 \
    --steps 10 \
    --wandb-project aide-attention-gpt4 \
    --model gpt-4 \
    --feedback-model gpt-4
```

#### Kernel Optimization with Claude Opus
```bash
./run_distributed.sh \
    --task kernel \
    --gpu-fraction 0.25 \
    --num-experiments 64 \
    --num-iterations 3 \
    --steps 15 \
    --wandb-project aide-kernel-claude \
    --model claude-3-opus-20240229 \
    --feedback-model claude-3-opus-20240229
```

#### Fast Iteration with o1-mini
```bash
./run_distributed.sh \
    --task attention \
    --gpu-fraction 0.1 \
    --num-experiments 160 \
    --steps 5 \
    --wandb-project aide-quick-test \
    --model o1-mini \
    --feedback-model o1-mini
```

### 6. Testing and Development

#### Local Testing (No GPU Cluster)
```bash
# Test with 2 experiments locally
./run_distributed.sh --local --num-experiments 2

# Local test with model override
./run_distributed.sh --local --num-experiments 2 --model gpt-4
```

#### Custom Contract Files
```bash
# Use specific contract configuration
./run_distributed.sh --contract contract_gpu_0.5.yaml

# Combine with other settings
./run_distributed.sh \
    --contract contract_gpu_0.25.yaml \
    --gpu-fraction 0.25 \
    --model gpt-4
```

## Parameter Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--task` | Task type (attention/kernel) | `--task kernel` |
| `--contract` | Custom contract file | `--contract contract_gpu.yaml` |
| `--num-experiments` | Number of parallel experiments | `--num-experiments 32` |
| `--num-iterations` | Iterations per experiment | `--num-iterations 4` |
| `--steps` | Steps per iteration | `--steps 10` |
| `--gpu-fraction` | GPU fraction per experiment | `--gpu-fraction 0.5` |
| `--wandb-project` | W&B project name | `--wandb-project aide-v2` |
| `--model` | Override generation model | `--model gpt-4` |
| `--feedback-model` | Override feedback model | `--feedback-model claude-3-opus-20240229` |
| `--local` | Run locally without cluster | `--local` |

## GPU Fraction Guidelines

| Fraction | Experiments per GPU | Use Case |
|----------|-------------------|----------|
| 1.0 | 1 | Maximum memory per experiment |
| 0.5 | 2 | Balanced throughput/memory |
| 0.25 | 4 | High throughput |
| 0.1 | 10 | Maximum parallelism |

## Available Models

### OpenAI Models
- `gpt-4` - Most capable, slower
- `gpt-4-turbo-preview` - Faster GPT-4
- `gpt-3.5-turbo` - Fast, less capable
- `o1-mini` - Optimized for speed
- `o1-preview` - Preview model

### Anthropic Models
- `claude-3-opus-20240229` - Most capable Claude
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fast, efficient

## Monitoring Experiments

### View Running Experiments
```bash
# Check Ray cluster status
ray status --address ray://172.26.134.141:10001

# Monitor GPU usage
./gpu_status.py

# Watch W&B dashboard
# Navigate to: https://wandb.ai/YOUR_USERNAME/YOUR_PROJECT
```

### Check Logs
```bash
# View Ray logs
ray logs --address ray://172.26.134.141:10001

# Check specific experiment logs
ls /tmp/ray/session_latest/logs/
```

## Troubleshooting

### If experiments fail to start:
```bash
# Restart Ray cluster
./stop_cluster.sh
./start_cluster.sh

# Check GPU availability
nvidia-smi
```

### If W&B isn't logging:
```bash
# Ensure W&B is configured
wandb login

# Check API key
echo $WANDB_API_KEY
```

### If models aren't overriding:
```bash
# Verify environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Check contract is being overridden
cat tasks/$TASK/contract.yaml | grep model
```