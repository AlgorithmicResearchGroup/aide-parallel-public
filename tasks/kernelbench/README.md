# KernelBench Strict Benchmark Integration

This repo now runs KernelBench only in a strict, publication-grade mode.

## Strict Workflow

1. Validate the environment and baseline contract:

```bash
./cli/aide-kernelbench-validate-env \
  --kb-reference-baseline H100_PCIe_LambdaLabs
```

2. Or generate a local eager baseline artifact on your hardware:

```bash
./cli/aide-kernelbench-generate-baseline --hardware-name MY_GPU
```

3. Run a strict single task:

```bash
./cli/aide-run \
  --local \
  --task kernelbench \
  --kb-task 1_23 \
  --kb-reference-baseline H100_PCIe_LambdaLabs \
  --num-experiments 4 \
  --steps 15
```

4. Run a strict campaign with the same interface shape as AlgoTune:

```bash
./cli/run-kb-sequence all \
  --local \
  --kb-reference-baseline H100_PCIe_LambdaLabs \
  --num-experiments 4 \
  --max-concurrent-tasks 4 \
  --tracking-experiment aide-kernelbench
```

## Baselines

Strict KernelBench requires exactly one explicit baseline source:

- `--kb-baseline-path <file>` for a locally generated baseline JSON
- `--kb-reference-baseline <name>` for a vendored reference baseline

The wrapper does not silently guess a baseline anymore. Vendored reference baselines are allowed, but they must be selected explicitly and should match the hardware used for the benchmark run.

## Metrics

Per-task search still uses correctness + speedup for agent feedback, but campaign summaries now report benchmark-style metrics:

- compilation rate
- correctness rate
- mean / median speedup over successful tasks
- geometric mean speedup over correct tasks
- `fast_p` metrics at multiple thresholds

These summary metrics are written to `runs/kernelbench/<campaign-id>/summary.json` and synced to MLflow when tracking is enabled.

## Notes

- Strict KernelBench requires CUDA.
- The vendored eager baseline generator currently covers levels 1-3. Level 4 requires a baseline artifact that contains level 4 timings.
- The official KernelBench correctness tolerance is `rtol=1e-2, atol=1e-2`, and the prompt layer in this repo now matches that.
