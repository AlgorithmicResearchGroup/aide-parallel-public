# MLflow Setup For AIDE

This repo now uses MLflow for optional experiment tracking.

## Local Tracking

Set these values in `.env`:

```bash
AIDE_ENABLE_MLFLOW=1
MLFLOW_EXPERIMENT_NAME=aide-public
```

If `MLFLOW_TRACKING_URI` is unset, AIDE writes runs to a local `mlruns/` directory in the repo.

## Remote Tracking

Point MLflow at an existing tracking server:

```bash
AIDE_ENABLE_MLFLOW=1
MLFLOW_TRACKING_URI=http://your-mlflow-host:5000
MLFLOW_EXPERIMENT_NAME=aide-public
```

You can also set `MLFLOW_ARTIFACT_ROOT` when creating a new experiment on a fresh tracking backend.

## What Gets Logged

- Per-attempt metrics such as `speedup`, `val_loss`, `training_time`, and iteration counters
- Generated code artifacts for intermediate and best solutions
- Run config snapshots and high-level campaign summaries

## Notes

- The runner creates one parent campaign run and one child run per experiment actor.
- AlgoTune sweeps can also forward tracking into the same MLflow experiment with `--tracking-experiment`.
