# GPU Parallelization Setup for AIDE

This repository has been enhanced to leverage multiple GPUs across a Ray cluster for parallel AIDE experiments.

## Cluster Configuration

The setup is configured for the following GPU cluster:

- **arg-node-001** (172.26.134.141): 8x H100 GPUs - Ray head node
- **arg-node-002** (172.26.134.74): 8x H100 GPUs - Ray worker node
- **Total**: 16 H100 GPUs available for parallel experiments

## Quick Start

### 1. Start the Ray Cluster

```bash
./start_cluster.sh
```

This will:
- Initialize Ray on both GPU nodes
- Configure 16 GPUs for parallel use
- Start the Ray dashboard on port 8265
- Verify cluster connectivity

### 2. Run Distributed Experiments

```bash
# Run with GPU-optimized configuration (16 parallel experiments)
./run_distributed.sh --contract contract_gpu.yaml

# Or use custom settings
./run_distributed.sh --num-experiments 8 --steps 10

# Run locally for testing
./run_distributed.sh --local --num-experiments 2
```

### 3. Monitor GPU Usage

```bash
# Real-time GPU monitoring
python gpu_status.py --head-node-ip 172.26.134.141

# One-time status check
python gpu_status.py --once --verbose

# JSON output for programmatic use
python gpu_status.py --json --once
```

### 4. Stop the Cluster

```bash
./stop_cluster.sh
```

## Key Files

### Core Scripts
- **`run_ray_gpu.py`**: GPU-aware version of the Ray experiment runner
- **`start_cluster.sh`**: Initialize Ray cluster on GPU nodes
- **`stop_cluster.sh`**: Shutdown Ray cluster
- **`run_distributed.sh`**: Launch distributed experiments with GPU support
- **`gpu_status.py`**: Monitor GPU utilization across the cluster
- **`test_gpu_setup.py`**: Verify GPU allocation is working correctly

### Configuration Files
- **`ray_cluster_config.yaml`**: Ray cluster configuration
- **`tasks/attention/contract_gpu.yaml`**: GPU-optimized task configuration (16 experiments)
- **`tasks/attention/evaluate_gpu.py`**: GPU-aware evaluation wrapper

## How It Works

1. **GPU Allocation**: Each AIDE experiment is assigned exactly 1 GPU through Ray's resource management
2. **Isolation**: Ray sets `CUDA_VISIBLE_DEVICES` for each actor, ensuring GPU isolation
3. **Parallel Execution**: Up to 16 experiments run simultaneously (one per GPU)
4. **Automatic Scheduling**: Ray queues experiments if more are requested than GPUs available

## Testing the Setup

```bash
# Test with 4 actors on the cluster
python test_gpu_setup.py --num-actors 4 --head-node-ip 172.26.134.141

# Test locally
python test_gpu_setup.py --num-actors 2
```

## Monitoring

The Ray dashboard is available at: `http://172.26.134.141:8265`

This provides:
- Real-time cluster utilization
- Job and task status
- Resource allocation visualization
- Performance metrics

## Troubleshooting

### Cannot connect to cluster
```bash
# Check if Ray is running on the nodes
ssh ubuntu@arg-node-001 "ray status"
ssh ubuntu@arg-node-002 "ray status"

# Restart the cluster
./stop_cluster.sh
./start_cluster.sh
```

### GPU not being used
```bash
# Verify GPUs are visible
ssh ubuntu@arg-node-001 "nvidia-smi"
ssh ubuntu@arg-node-002 "nvidia-smi"

# Check Ray sees the GPUs
python -c "import ray; ray.init(address='ray://172.26.134.141:10001'); print(ray.cluster_resources()); ray.shutdown()"
```

### Experiments running on CPU
- Ensure you're using `evaluate_gpu.py` or that `evaluate.py` has been updated
- Check that `CUDA_VISIBLE_DEVICES` is being set correctly
- Verify PyTorch has CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

## Performance Optimization

### Current Configuration
- **1 GPU per experiment**: Maximum parallelism, good for memory-intensive tasks
- **16 concurrent experiments**: Fully utilizes all available GPUs
- **3 iterations**: Balanced between exploration and refinement

### Alternative Configurations

For lighter tasks, you could modify to:
- Use fractional GPUs (0.5 per experiment = 32 concurrent)
- Use multiple GPUs per experiment for faster individual training
- Adjust in `run_ray_gpu.py` by changing the `@ray.remote(num_gpus=X)` decorator

## Advanced Usage

### Custom Ray Configuration
Edit `ray_cluster_config.yaml` to adjust:
- Object store memory
- Number of CPUs/GPUs per node
- Dashboard settings

### Programmatic Access
```python
import ray
from run_ray_gpu import Experiment

# Connect to cluster
ray.init(address="ray://172.26.134.141:10001")

# Launch experiments programmatically
actor = Experiment.remote(data_dir, goal, model, feedback_model, eval_metric)
result = ray.get(actor.run.remote(steps=10))
```

## Notes

- The system automatically handles GPU memory management through PyTorch
- Each experiment runs in isolation with its own GPU
- Ray provides automatic fault tolerance and retry mechanisms
- Logs are stored in `/tmp/ray/session_*/logs/` on each node