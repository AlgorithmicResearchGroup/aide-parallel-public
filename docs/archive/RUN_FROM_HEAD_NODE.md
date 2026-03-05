# Running GPU Jobs from Head Node (arg-head-001)

Since you're on the **head node (arg-head-001)** which has **no GPUs**, you need to connect to the Ray cluster running on the GPU nodes to utilize the GPUs.

## Architecture Overview

```
arg-head-001 (You are here)
├── CPU only, no GPUs
├── Runs Ray client code
└── Submits jobs to GPU nodes

arg-node-001 (GPU Node 1)
├── 8x H100 GPUs
├── Runs Ray head node
└── Executes GPU workloads

arg-node-002 (GPU Node 2)
├── 8x H100 GPUs
├── Runs Ray worker node
└── Executes GPU workloads
```

## Step-by-Step Guide

### 1. Start the Ray Cluster on GPU Nodes

From the head node (where you are now):
```bash
./start_cluster.sh
```

This will:
- SSH to arg-node-001 and start Ray head node with 8 GPUs
- SSH to arg-node-002 and start Ray worker node with 8 GPUs
- Create a cluster with 16 total GPUs

### 2. Verify Cluster is Running

```bash
# Check cluster status
python3 -c "
import ray
ray.init(address='ray://172.26.134.141:10001')
print(f'Total GPUs in cluster: {ray.cluster_resources().get(\"GPU\", 0)}')
ray.shutdown()
"
```

Expected output:
```
Total GPUs in cluster: 16.0
```

### 3. Run GPU Tests from Head Node

```bash
# Test with cluster connection
./test_gpu_from_head.sh

# Or directly:
python test_gpu_setup.py --head-node-ip 172.26.134.141 --num-actors 4
```

### 4. Run Distributed Experiments

```bash
# Run AIDE experiments on GPU cluster
./run_distributed.sh --contract contract_gpu.yaml

# Or with custom settings
python run_ray_gpu.py --task attention --head-node-ip 172.26.134.141
```

### 5. Monitor GPU Usage

In a separate terminal on the head node:
```bash
# Real-time GPU monitoring
python gpu_status.py --head-node-ip 172.26.134.141

# One-time check
python gpu_status.py --head-node-ip 172.26.134.141 --once --verbose
```

## Important Notes

### ❌ What DOESN'T Work on Head Node:

```python
# This will fail - no local GPUs
ray.init()  # Local mode - NO GPUs!
torch.cuda.is_available()  # Returns False
```

### ✅ What DOES Work on Head Node:

```python
# Connect to GPU cluster
ray.init(address='ray://172.26.134.141:10001')  # Connected to 16 GPUs!

# Submit jobs that run on GPU nodes
@ray.remote(num_gpus=1)
class GPUActor:
    def run(self):
        # This code runs on GPU nodes, not head node
        return torch.cuda.is_available()  # Returns True on GPU nodes
```

## Troubleshooting

### "No GPUs available" Error
- **Cause**: Running Ray in local mode on head node
- **Fix**: Connect to cluster with `--head-node-ip 172.26.134.141`

### "Cannot connect to Ray cluster"
- **Cause**: Cluster not started
- **Fix**: Run `./start_cluster.sh` first

### "Ray actors queuing forever"
- **Cause**: Requesting more GPUs than available (16 max)
- **Fix**: Reduce `num_experiments` or wait for GPUs to free up

## Quick Reference

```bash
# From head node (arg-head-001):

# Start cluster
./start_cluster.sh

# Run experiments (connects to GPU cluster automatically)
./run_distributed.sh --contract contract_gpu.yaml

# Monitor GPUs
python gpu_status.py --head-node-ip 172.26.134.141

# Stop cluster
./stop_cluster.sh
```

## Alternative: Run Directly on GPU Node

If you want to test directly on a GPU node:
```bash
# SSH to GPU node
ssh arg-node-001

# Run local test
cd /home/ubuntu/aide_parallel
python test_gpu_setup.py --num-actors 2
```

But for production runs, always use the cluster mode from the head node for proper distributed execution!