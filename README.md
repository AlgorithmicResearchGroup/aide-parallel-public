# AIDE Parallel

AIDE Parallel runs AIDE experiments locally or on a Ray cluster. The simplest first run is the attention task on CPU. KernelBench is available, but it requires a GPU environment and is not the recommended first-run path.

## Quick Start

Use Python 3.10 or newer. `aideml` will not install on older interpreters.
The commands below use `python3.12` as an example; replace it with any installed Python 3.10+ binary.

Create a virtual environment and install the repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ./aideml
```

Copy the environment template:

```bash
cp .env.example .env
```

Set one provider and one model pair in `.env`.

Groq example:

```bash
GROQ_API_KEY=...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
AIDE_MODEL=llama-3.3-70b-versatile
AIDE_FEEDBACK_MODEL=llama-3.3-70b-versatile
```

Anthropic example:

```bash
ANTHROPIC_API_KEY=...
AIDE_MODEL=claude-sonnet-4-20250514
AIDE_FEEDBACK_MODEL=claude-sonnet-4-20250514
```

Optional MLflow tracking:

```bash
AIDE_ENABLE_MLFLOW=1
MLFLOW_EXPERIMENT_NAME=aide-public
```

If `MLFLOW_TRACKING_URI` is unset, runs are stored locally under `mlruns/`.
For longer-lived tracking, prefer setting `MLFLOW_TRACKING_URI` to a real MLflow server or database-backed deployment.

Run the deterministic setup check:

```bash
./cli/aide-check
```

Run one local optimization step:

```bash
AIDE_ATTENTION_FAST_EVAL=1 ./cli/aide-run --local --task attention --num-experiments 1 --num-iterations 1 --steps 1
```

## What Works Best

- Use `./cli/aide-check` first. It verifies imports and runs a baseline attention evaluation on CPU.
- Use the attention task for first run. It now auto-prepares a tiny Shakespeare dataset if the wiki dataset is missing.
- Set `AIDE_MODEL` and `AIDE_FEEDBACK_MODEL` explicitly. Do not rely on provider-specific default model availability.
- For the current repo state, Groq and Anthropic are the most reliable provider paths.
- Enable `AIDE_ENABLE_MLFLOW=1` if you want experiment tracking. Local MLflow works without a server.

## KernelBench

KernelBench needs a CUDA-capable GPU setup.

Run a local KernelBench job:

```bash
./cli/aide-run --local --task kernelbench --kb-task 1_19 --num-experiments 1 --num-iterations 1 --steps 1
```

For Linux GPU nodes, install CUDA-specific PyTorch wheels separately, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## AlgoTune

AlgoTune is available as an optional advanced benchmark and is not part of the default first-run path.

Use a separate Python 3.10 environment for it:

```bash
python3.10 -m venv .venv-algotune
source .venv-algotune/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-algotune.txt
python -m pip install -e ./aideml
```

Then run a small local smoke test:

```bash
./cli/aide-run --local --task algotune --at-task kmeans --num-experiments 1 --num-iterations 1 --steps 1 --cpus-per-experiment 2
```

Validate the strict benchmark environment before any publishable AlgoTune run:

```bash
./cli/aide-algotune-validate-env
```

Fetch the local Hugging Face snapshot explicitly before strict benchmark runs:

```bash
export ALGOTUNE_HF_REVISION='fc3744ffd7eebaa9e9b55427e2cda440955fdd2d'
./cli/aide-algotune-fetch-dataset --task kmeans
```

Run a strict held-out benchmark task:

```bash
./cli/aide-run --local --task algotune --at-task kmeans --num-experiments 1 --num-iterations 1 --steps 1 --cpus-per-experiment 2
```

The repo now exposes only the publishable AlgoTune path. It validates the Python 3.10 environment up front, searches on the train split, runs one held-out test evaluation at the end, and rejects repo-side timeout/compatibility shortcuts.
If you source datasets from Hugging Face for strict runs, pin `ALGOTUNE_HF_REVISION` to a non-`main` revision and prefetch the local snapshot so the benchmark run itself does not perform network downloads.
The shared AlgoTune env is only valid if `./cli/aide-algotune-validate-env` still passes after `aideml` is installed.

Run a resumable sweep across the full AlgoTune inventory:

```bash
./cli/run-at-sequence all --local --profile coverage
```

Control per-task attempts and concurrent tasks independently:

```bash
./cli/run-at-sequence all --local --profile coverage --attempts-per-task 4 --max-concurrent-tasks 4
```

More details: `tasks/algotune/README.md`

## Optional Dependencies

If you want notebooks, tracing, or extra benchmarking tools:

```bash
python -m pip install -r requirements-optional.txt
```

For the optional AlgoTune benchmark:

```bash
python -m pip install -r requirements-algotune.txt
python -m pip install -e ./aideml
```

## Main Commands

- `./cli/aide-check`: validate the local install with a deterministic CPU run
- `./cli/aide-algotune-validate-env`: validate the strict AlgoTune benchmark environment
- `./cli/aide-run`: run AIDE experiments
- `./cli/aide-run --task algotune --at-task <task>`: run an AlgoTune task locally or on Ray
- `./cli/run-at-sequence all --profile coverage`: run a resumable AlgoTune sweep
- `./cli/run-at-sequence all --attempts-per-task N --max-concurrent-tasks M`: control AlgoTune search depth and sweep concurrency
- `./cli/aide-cluster-up`: start a Ray cluster from env vars
- `./cli/aide-cluster-down`: stop the Ray cluster
- `./cli/run-kb-sequence quick`: run a preset KernelBench sequence
