# Quick Start Guide - GPU Parallel AIDE

## First Time Setup (One-time only)

### 1. Install Dependencies on Head Node
```bash
# Install AIDE and dependencies locally
cd /home/ubuntu/aide_parallel/aideml
pip install -e .
cd ..

# Install Ray and other requirements
pip install ray[default]==2.10.0 python-dotenv backoff
```

### 2. Setup GPU Nodes
```bash
# Install dependencies on all GPU nodes (required!)
./setup_gpu_nodes.sh
```

This installs AIDE, Ray, and all dependencies on arg-node-001 and arg-node-002.

## Running Experiments

### 1. Start the Ray Cluster
```bash
./start_cluster.sh
```
This initializes Ray on both GPU nodes with 16 total GPUs.

### 2. Run Distributed Experiments
```bash
# Full 16-GPU parallel run
./run_distributed.sh --contract contract_gpu.yaml

# Test with fewer experiments
./run_distributed.sh --num-experiments 4 --steps 2 --num-iterations 1
```

### 3. Monitor GPU Usage
In a separate terminal:
```bash
python gpu_status.py --head-node-ip 172.26.134.141
```

### 4. Stop the Cluster
```bash
./stop_cluster.sh
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'aide'"
The GPU nodes don't have AIDE installed. Run:
```bash
./setup_gpu_nodes.sh
```

### "No GPUs available"
You're running locally on the head node. Make sure to:
1. Start the cluster: `./start_cluster.sh`
2. Use the cluster IP: `--head-node-ip 172.26.134.141`

### "Cannot connect to Ray cluster"
The cluster isn't running. Start it with:
```bash
./start_cluster.sh
```

## Quick Test
```bash
# Minimal test (1 experiment, 1 step)
./run_distributed.sh --num-experiments 1 --steps 1 --num-iterations 1
```

## Important Notes
- Always run from the **head node** (arg-head-001)
- The head node has **no GPUs** - it only coordinates
- All GPU work happens on **arg-node-001** and **arg-node-002**
- Each experiment gets **1 dedicated H100 GPU**