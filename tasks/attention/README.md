# Attention-Agent Task

Optimise the causal attention mechanism used by nanoGPT on a Wikipedia subset. The agent edits
`optimize.py`, which must expose a `build_attention(config)` factory returning an `nn.Module` compatible with the
transformer blocks in `nanoGPT/model.py`.

## Files

- `optimize.py` – baseline attention implementation. Modify this file.
- `evaluate.py` – training + evaluation harness. It loads the candidate attention block, trains a compact nanoGPT model
  on a tokenized Wikipedia slice for a short schedule, and prints `val_loss` (lower is better).
- `contract.yaml` – configuration consumed by `run_ray.py` when launching batches.
- `nanoGPT/` – snapshot of the nanoGPT project, including the preprocessed Wiki corpus under `nanoGPT/data/wiki`.

## Manual evaluation

```bash
cd tasks/attention
python evaluate.py --solution-path optimize.py --device cpu
```

This command trains for a handful of gradient steps on the wiki subset and reports validation loss. Ensure your attention
module keeps interface and shape compatibility, avoids NaNs/Infs, and remains efficient.
