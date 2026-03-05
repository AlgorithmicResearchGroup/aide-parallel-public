# AIDE Parallel

AIDE Parallel is a Ray-based runner for AIDE experiments, with built-in support for attention and KernelBench tasks.

## 1. Prerequisites

- Python 3.10+
- `pip`
- GPU + CUDA for GPU tasks
- API keys for your configured model provider(s)

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e ./aideml
```

## 2. Configuration

Copy environment template:

```bash
cp .env.example .env
```

Set at least the keys you use (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GROQ_API_KEY`).

## 3. Quick Start (Local)

Run a small local test:

```bash
./cli/aide-run --local --task attention --num-experiments 1 --num-iterations 1 --steps 1
```

Run KernelBench locally:

```bash
./cli/aide-run --local --task kernelbench --kb-task 1_19 --num-experiments 1 --num-iterations 1 --steps 1
```

## 4. Distributed Run (Optional)

Set cluster env values (example):

```bash
export AIDE_HEAD_NODE_IP=10.0.0.5
```

Then run:

```bash
./cli/aide-run --task kernelbench --kb-task 1_19 --num-experiments 16 --gpu-fraction 1.0
```

If you manage Ray via provided scripts:

```bash
./cli/aide-cluster-up
./cli/aide-cluster-down
```

Cluster scripts are env-driven (`AIDE_HEAD_HOST`, `AIDE_WORKER_HOSTS`, `AIDE_SSH_USER`, etc.).

## 5. Main Commands

- `./cli/aide-run`: primary experiment launcher
- `python src/aide_runner.py ...`: Python entrypoint equivalent
- `python src/cluster_gpu_status.py --once`: cluster GPU snapshot
- `./cli/run-kb-sequence quick`: run preset KernelBench task sequence

## 6. Outputs

- `logs/`: run logs and best solutions
- `workspaces/`: per-run workspaces

## 7. Compatibility Wrappers

Deprecated names are still available for one release:

- `compat/run_distributed.sh` -> `cli/aide-run`
- `compat/run_ray_gpu.py` -> `src/aide_runner.py`
- `compat/start_cluster.sh` -> `cli/aide-cluster-up`
- `compat/stop_cluster.sh` -> `cli/aide-cluster-down`
- `compat/gpu_status.py` -> `src/cluster_gpu_status.py`

## 8. Repository Layout

- `cli/`: primary executable scripts
- `src/`: Python runtime entrypoints and integrations
- `tasks/`: task contracts and task-specific code
- `compat/`: temporary wrappers for old command names
