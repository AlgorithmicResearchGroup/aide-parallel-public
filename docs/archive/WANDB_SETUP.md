# Weights & Biases Setup for AIDE GPU Parallel

## Quick Setup

### 1. Get your W&B API Key

1. Sign up / Log in at https://wandb.ai
2. Go to https://wandb.ai/settings
3. Copy your API key

### 2. Update the .env file

Edit `/home/ubuntu/aide_parallel/.env` and replace the placeholders:

```bash
WANDB_API_KEY=your_actual_api_key_here
WANDB_PROJECT=aide-attention-h100  # or your project name
WANDB_ENTITY=your_username_or_team  # optional, leave empty for personal
```

### 3. Sync to GPU nodes

```bash
# Copy .env to GPU nodes
for node in arg-node-001 arg-node-002; do
    scp /home/ubuntu/aide_parallel/.env ubuntu@$node:/home/ubuntu/aide_parallel/
done
```

## What Gets Logged

When you run experiments with W&B enabled, the following metrics are tracked:

### Per Experiment (Each GPU runs separately):
- **val_loss**: Validation loss at each step
- **best_val_loss**: Best validation loss achieved
- **training_time**: Time taken for each evaluation
- **aide_step**: Current AIDE optimization step
- **gpu_id**: Which GPU is running the experiment
- **experiment_idx**: Experiment number (0-15 for 16 GPUs)

### Code Artifacts:
- Generated code at each step
- Best code found by each experiment
- Full solution tree structure

### GPU Metrics:
- GPU utilization percentage
- GPU memory usage
- GPU assignment per experiment

## Viewing Results

### Online Dashboard
After running experiments, view results at:
```
https://wandb.ai/YOUR_ENTITY/aide-attention-h100
```

### Features:
- **Parallel Runs View**: Compare all 16 experiments side-by-side
- **Loss Curves**: Track validation loss over time for each GPU
- **Best Performers**: Automatically identifies best solutions
- **Code Diffs**: Compare generated code versions
- **GPU Utilization**: Monitor resource usage

## Running with W&B

Once configured, just run normally:

```bash
# W&B logging happens automatically if API key is set
./run_distributed.sh --contract contract_gpu.yaml
```

## Grouping Experiments

All parallel experiments from one run are grouped together:
- **Group**: "parallel_run" - All 16 experiments in one batch
- **Job Type**: "optimization" - Categorizes as optimization task
- **Run Names**: `exp_0_gpu0`, `exp_1_gpu1`, etc.

## Example W&B Queries

In the W&B dashboard, you can create custom charts:

### Compare GPU Performance:
```python
# Group by gpu_id and show average val_loss
grouped = runs.groupby('config.gpu_id')['val_loss'].mean()
```

### Find Best Solution Across All GPUs:
```python
# Get minimum val_loss across all runs
best_loss = runs['val_loss'].min()
best_run = runs[runs['val_loss'] == best_loss]
```

## Disabling W&B

To run without W&B logging:
- Set `WANDB_API_KEY=` (empty) in .env
- Or remove the WANDB_API_KEY line entirely

## Troubleshooting

### "Warning: W&B API key not configured"
- Update .env with your actual API key
- Sync .env to GPU nodes

### W&B runs not showing up
- Check API key is valid
- Ensure wandb is installed on GPU nodes
- Check network connectivity to wandb.ai

### Multiple runs with same name
- This is normal - each GPU creates its own run
- They're grouped under the same project