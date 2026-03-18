# AlgoTune Task

AlgoTune is available as an optional advanced benchmark inside this repo.

The vendored runtime lives under `tasks/algotune/vendor/AlgoTune` and preserves the upstream MIT license in `tasks/algotune/vendor/AlgoTune/LICENSE`.

## Install

Use a separate Python 3.10 virtual environment for AlgoTune. The base repo install stays unchanged.

```bash
python3.10 -m venv .venv-algotune
source .venv-algotune/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-algotune.txt
python -m pip install -e ./aideml
```

If you keep AlgoTune somewhere else, set `ALGOTUNE_PATH=/absolute/path/to/AlgoTune`.
Strict runs no longer download the Hugging Face snapshot implicitly. Pin `ALGOTUNE_HF_REVISION` to a non-`main` revision and fetch the local snapshot first:

```bash
export ALGOTUNE_HF_REVISION='fc3744ffd7eebaa9e9b55427e2cda440955fdd2d'
./cli/aide-algotune-fetch-dataset --task base64_encoding
```

## Run

A small local smoke test:

```bash
./cli/aide-run --local --task algotune --at-task kmeans --num-experiments 1 --num-iterations 1 --steps 1 --cpus-per-experiment 2
```

Validate the strict benchmark environment before any publishable run:

```bash
./cli/aide-algotune-validate-env
```

Run a strict single-task benchmark search plus held-out final test evaluation:

```bash
./cli/aide-run --local --task algotune --at-task kmeans --num-experiments 1 --num-iterations 1 --steps 1 --cpus-per-experiment 2
```

A full resumable coverage sweep:

```bash
./cli/run-at-sequence all --local --profile coverage
```

Control task-level and per-task parallelism independently:

```bash
./cli/run-at-sequence all --local --profile coverage --attempts-per-task 8 --max-concurrent-tasks 4
```

Sweep outputs are written under `runs/algotune/<campaign-id>/` with per-task logs, result JSON, copied best solver code, and aggregate summaries.

If you enable `AIDE_ENABLE_MLFLOW=1`, the sweep also logs campaign and per-attempt tracking data to the configured MLflow experiment.

## Notes

- AlgoTune is CPU-oriented in this integration. Use `--cpus-per-experiment` when running locally or on CPU Ray workers.
- This repo now exposes only the publication-grade AlgoTune path. It searches on the train split and reports one final held-out test evaluation.
- Strict runs require the dataset snapshot to already exist locally. Use `./cli/aide-algotune-fetch-dataset` before benchmarking.
- Repo-side outer task timeouts, compatibility skips, local AlgoTune config overrides, and custom fast/fallback evaluators have been removed.
- Sweep summaries now include the benchmark-style harmonic-mean score with a 1.0 mercy floor for failed or slower-than-baseline tasks.
- `--attempts-per-task` controls how many AIDE attempts each AlgoTune task gets.
- `--max-concurrent-tasks` controls how many different AlgoTune tasks are active at once.
- Approximate CPU demand is `attempts-per-task * max-concurrent-tasks * cpus-per-experiment` when the sweep is fully loaded.
- The shared AlgoTune env is only valid if `./cli/aide-algotune-validate-env` passes after `aideml` is installed. `aideml` should not downgrade AlgoTune's scientific stack.
- Official AlgoTune support is best validated on Python 3.10 because that matches the upstream project.
