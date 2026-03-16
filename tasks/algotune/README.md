# AlgoTune Task

AlgoTune is available as an optional advanced benchmark inside this repo.

The vendored runtime lives under `tasks/algotune/vendor/AlgoTune` and preserves the upstream MIT license in [LICENSE](/Users/arg/Desktop/PUBLIC/aide_parallel/tasks/algotune/vendor/AlgoTune/LICENSE).

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

## Run

A small local smoke test:

```bash
./cli/aide-run --local --task algotune --at-task kmeans --num-experiments 1 --num-iterations 1 --steps 1 --cpus-per-experiment 2
```

## Notes

- AlgoTune is CPU-oriented in this integration. Use `--cpus-per-experiment` when running locally or on CPU Ray workers.
- If you install AlgoTune into the same environment as AIDE, install `requirements-algotune.txt` before `pip install -e ./aideml` so AIDE's pinned versions win the final resolution.
- Some cryptography/compression tasks use byte payloads and may require reduced subprocess isolation. The evaluator can disable isolation automatically for known affected tasks, and you can also set `ALGOTUNE_DISABLE_SUBPROCESS_ISOLATION=1`.
- Official AlgoTune support is best validated on Python 3.10 because that matches the upstream project.
