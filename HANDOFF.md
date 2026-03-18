# Session Handoff

We are on branch `cleanup` in repo `/home/matt/Desktop/ray/aide-parallel-public`.

Main goal has been to make AlgoTune and KernelBench publication-grade, strict-only benchmark integrations with low cognitive overhead between them.

## Current State

### 1. AlgoTune
- AlgoTune is now strict-only from the repo wrapper perspective.
- Removed repo-owned dev/loose benchmark paths, fallback data generation, compatibility skips, outer task timeouts for benchmark runs, and relaxed evaluator behavior.
- Strict path now does:
  - search on train only
  - final evaluation on test only
  - explicit environment validation
  - pinned HF revision via `ALGOTUNE_HF_REVISION`
- Added `.venv-algotune` with Python 3.10 and fixed dependency conflicts.
- `./cli/aide-algotune-validate-env --json` passes in `.venv-algotune`.
- Added explicit dataset fetch flow so benchmark runtime no longer downloads from HF:
  - `cli/aide-algotune-fetch-dataset`
- Fixed HF download issues caused by `huggingface_hub` / `tqdm` incompatibility.
- Added baseline reuse across AIDE steps in strict AlgoTune.
- Added detailed progress logging across:
  - `src/aide_runner.py`
  - `aideml/aide/agent.py`
  - `aideml/aide/backend/__init__.py`
  - `aideml/aide/interpreter.py`
  - vendored AlgoTune evaluator files
- MLflow logging/traces are working, including:
  - model-call traces
  - task outcome traces
  - explicit `task_id` / `experiment_idx`
  - token/cost fields
  - sweep summary runs
- Strict AlgoTune Ray runs are now working end to end.
- Real strict result observed:
  - `base64_encoding` child runs produced valid `search_mean_speedup` around `0.89-0.94`, i.e. correct but slower than baseline.
- Important fix:
  - strict benchmark wrapper was initially misclassifying valid slow runs as invalid; this is fixed now.

### 2. KernelBench
- KernelBench has been refactored toward the same strict-only philosophy and interface shape as AlgoTune.
- Added strict evaluator:
  - `tasks/kernelbench/strict_benchmark.py`
- Replaced old wrapper:
  - `tasks/kernelbench/evaluate_gpu.py` is now a thin strict wrapper
- Added new CLIs:
  - `cli/aide-kernelbench-validate-env`
  - `cli/aide-kernelbench-generate-baseline`
- Replaced legacy `cli/run-kb-sequence` with an AlgoTune-style resumable runner supporting:
  - `--num-experiments`
  - `--max-concurrent-tasks`
  - resume/rerun
  - manifests
  - MLflow summary sync
- Added campaign summary module:
  - `src/kernelbench_sweep.py`
  - computes `compiled_rate`, `correct_rate`, geometric mean speedup, and `fast_p` variants
- KernelBench now requires an explicit baseline contract:
  - `--kb-baseline-path`
  - or `--kb-reference-baseline`
- Removed silent hardcoded H100 fallback behavior from active strict path.
- Prompt/task metadata updated to avoid H100-specific wording and match official tolerance semantics.
- Vendored KernelBench optional provider imports were made optional so local evaluation does not fail on missing cloud SDKs.
- Docs/contracts updated:
  - `README.md`
  - `tasks/kernelbench/README.md`
  - `tasks/kernelbench/contract.yaml`
  - `tasks/kernelbench/contract_gpu.yaml`

### 3. Verification Status
- All major modified Python files compile.
- Help works for the new CLIs.
- KernelBench strict validator now fails correctly on this machine because CUDA is unavailable:
  - `torch.cuda.is_available() is false`
- Strict KernelBench `aide-run` now fails fast and writes structured failure JSON instead of drifting into loose behavior.
- KernelBench summary helper was smoke-tested on synthetic manifest data.
- No real end-to-end strict KernelBench GPU run has been done yet because this machine lacks usable CUDA in the active runtime.

### 4. Important Files Changed
- `src/aide_runner.py`
- `aideml/aide/interpreter.py`
- `aideml/aide/agent.py`
- `aideml/aide/backend/__init__.py`
- `tasks/algotune/strict_benchmark.py`
- `tasks/algotune/evaluate_algotune.py`
- `tasks/algotune/vendor/AlgoTune/...` multiple evaluator and HF helper files
- `cli/aide-run`
- `cli/run-at-sequence`
- `cli/aide-algotune-validate-env`
- `cli/aide-algotune-fetch-dataset`
- `scripts/run_algotune_strict_ray.sh`
- `src/algotune_sweep.py`
- `src/mlflow_integration.py`
- `tasks/kernelbench/strict_benchmark.py`
- `tasks/kernelbench/evaluate_gpu.py`
- `cli/aide-kernelbench-validate-env`
- `cli/aide-kernelbench-generate-baseline`
- `cli/run-kb-sequence`
- `src/kernelbench_sweep.py`
- `tasks/kernelbench/kb_tasks.py`
- `tasks/kernelbench/kernelbench_prompts.py`
- `tasks/kernelbench/prepare_kernelbench_task.py`
- `tasks/kernel_bench/KernelBench/src/utils.py`

### 5. Current Likely Next Steps
- On a CUDA machine, run:
  - `./cli/aide-kernelbench-validate-env --kb-reference-baseline <baseline>`
  - or generate a local baseline with `./cli/aide-kernelbench-generate-baseline`
  - then run one strict KernelBench task end to end
- Verify KernelBench MLflow logging and final campaign summary on real GPU runs
- Optionally commit and push all outstanding changes if not already committed

### 6. Repo State Note
- Work has been happening directly in the working tree on branch `cleanup`.
- Check `git status` immediately to see modified/untracked files before continuing.

## Suggested Resume Prompt

Start by reading `git status`, then inspect the strict KernelBench path and continue verification from there.
