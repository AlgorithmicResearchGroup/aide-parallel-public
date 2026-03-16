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

OpenAI example:

```bash
OPENAI_API_KEY=...
AIDE_MODEL=gpt-4o-mini
AIDE_FEEDBACK_MODEL=gpt-4o-mini
```

Groq example:

```bash
GROQ_API_KEY=...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
AIDE_MODEL=llama-3.3-70b-versatile
AIDE_FEEDBACK_MODEL=llama-3.3-70b-versatile
```

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

More details: [tasks/algotune/README.md](/Users/arg/Desktop/PUBLIC/aide_parallel/tasks/algotune/README.md)

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
- `./cli/aide-run`: run AIDE experiments
- `./cli/aide-run --task algotune --at-task <task>`: run an AlgoTune task locally or on Ray
- `./cli/aide-cluster-up`: start a Ray cluster from env vars
- `./cli/aide-cluster-down`: stop the Ray cluster
- `./cli/run-kb-sequence quick`: run a preset KernelBench sequence
