"""Evaluation harness for the Attention-Agent task.

This script trains a compact nanoGPT model on a subset of OpenWebText using the attention module defined in
``optimize.py``. It prints the validation loss after a short training schedule. Lower ``val_loss`` is better.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import math

try:
    import wandb
except ImportError:
    wandb = None

# Paths ---------------------------------------------------------------------
TASK_ROOT = Path(__file__).resolve().parent
NANOGPT_ROOT = TASK_ROOT / "nanoGPT"
_DATA_CACHE: dict[str, torch.Tensor] = {}


class _NullWandb:
    run = None

    @staticmethod
    def init(*args, **kwargs):
        return None

    @staticmethod
    def log(*args, **kwargs):
        return None


WANDB = wandb if wandb is not None else _NullWandb()
WANDB_ENABLED = (
    wandb is not None
    and os.environ.get("AIDE_ENABLE_WANDB", "0") == "1"
    and bool(os.environ.get("WANDB_API_KEY"))
)


def _repo_attention_data_root() -> Path | None:
    project_root = os.environ.get("AIDE_PROJECT_ROOT")
    if not project_root:
        return None
    candidate = Path(project_root).expanduser().resolve() / "tasks" / "attention" / "nanoGPT" / "data"
    return candidate if candidate.exists() else None


def _prepare_shakespeare_char_dataset(data_dir: Path) -> None:
    prepare_script = data_dir / "prepare.py"
    if not prepare_script.exists():
        raise FileNotFoundError(f"Missing dataset preparation script: {prepare_script}")

    print(f"Preparing bundled dataset in {data_dir} ...", flush=True)
    result = os.system(f'"{sys.executable}" "{prepare_script}"')
    if result != 0:
        raise RuntimeError(f"Dataset preparation failed for {data_dir}")


def _resolve_data_dir() -> Path:
    env_override = os.environ.get("AIDE_ATTENTION_DATA_DIR")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())

    local_data_root = NANOGPT_ROOT / "data"
    repo_data_root = _repo_attention_data_root()
    data_roots = [local_data_root]
    if repo_data_root is not None and repo_data_root != local_data_root:
        data_roots.append(repo_data_root)

    for data_root in data_roots:
        candidates.extend(
            [
                data_root / "wiki",
                data_root / "shakespeare_char",
            ]
        )

    for candidate in candidates:
        if (candidate / "train.bin").exists() and (candidate / "val.bin").exists():
            return candidate

    for data_root in data_roots:
        shakespeare_char_dir = data_root / "shakespeare_char"
        if shakespeare_char_dir.exists():
            _prepare_shakespeare_char_dataset(shakespeare_char_dir)
            if (shakespeare_char_dir / "train.bin").exists() and (shakespeare_char_dir / "val.bin").exists():
                return shakespeare_char_dir

    raise FileNotFoundError("Could not locate or prepare an attention dataset.")


DATA_DIR = _resolve_data_dir()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def prepare_environment() -> None:
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _use_fast_eval(device: torch.device) -> bool:
    if os.environ.get("AIDE_ATTENTION_FULL_EVAL") == "1":
        return False
    if os.environ.get("AIDE_ATTENTION_FAST_EVAL") == "0":
        return False
    return device.type != "cuda" or DATA_DIR.name != "wiki"


def load_meta() -> dict:
    import pickle

    meta_path = DATA_DIR / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            "OpenWebText metadata not found. Please ensure dataset preprocessing has been run in nanoGPT."
        )
    with meta_path.open("rb") as f:
        return pickle.load(f)


def _load_sequence(split: str) -> torch.Tensor:
    if split in _DATA_CACHE:
        return _DATA_CACHE[split]

    bin_path = DATA_DIR / ("train.bin" if split == "train" else "val.bin")
    limit = 2_000_000 if split == "train" else 200_000
    with open(bin_path, "rb") as f:
        trimmed = np.fromfile(f, dtype=np.uint16, count=limit).astype(np.int64)
    tensor = torch.from_numpy(trimmed)
    _DATA_CACHE[split] = tensor
    return tensor


def get_batch(split: str, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = _load_sequence(split)
    if data.size(0) < block_size + 1:
        raise ValueError("Dataset is too small for the requested block size.")
    idx = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    windows = torch.stack([data[i : i + block_size] for i in idx])
    targets = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
    x = windows
    y = targets
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def inject_attention(model: nn.Module, builder: Callable[[object], nn.Module], device: torch.device) -> None:
    for block in model.transformer.h:  # type: ignore[attr-defined]
        attn = builder(model.config)
        if not isinstance(attn, nn.Module):
            raise TypeError("build_attention must return an nn.Module")
        block.attn = attn.to(device)

# Import nanoGPT components --------------------------------------------------
import sys
if str(NANOGPT_ROOT) not in sys.path:
    sys.path.append(str(NANOGPT_ROOT))

from model import GPT, GPTConfig


def train_and_eval(builder: Callable[[object], nn.Module], device_str: str) -> float:
    print("\n" + "="*60, flush=True)
    print("[ATTENTION TRAINING] Starting train_and_eval function", flush=True)
    print(f"Device: {device_str}", flush=True)
    print("="*60 + "\n", flush=True)

    prepare_environment()
    device = torch.device(device_str)
    fast_eval = _use_fast_eval(device)

    if fast_eval:
        print(f"Using fast local evaluation profile with dataset: {DATA_DIR.name}", flush=True)
    else:
        print(f"Using full evaluation profile with dataset: {DATA_DIR.name}", flush=True)

    # Initialize W&B if not already initialized
    if WANDB_ENABLED and WANDB.run is None:
        WANDB.init(project="attention-training", name="nanoGPT-training")

    meta = load_meta()
    vocab_size = meta.get("vocab_size", 50304)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=128 if fast_eval else 512,
        n_layer=2 if fast_eval else 12,
        n_head=2 if fast_eval else 12,
        n_embd=128 if fast_eval else 768,
        dropout=0.1,
        bias=False,
    )

    model = GPT(config).to(device)
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
    inject_attention(model, builder, device)

    base_lr = 4e-4
    min_lr = 4e-5
    warmup_steps = 2 if fast_eval else 100
    total_optimizer_steps = 5 if fast_eval else 900

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=1e-2)

    model.train()
    micro_batch_size = 4 if fast_eval else 8
    grad_accum_steps = 1 if fast_eval else 6
    block_size = config.block_size

    optimizer_step = 0
    total_tokens_per_step = block_size * micro_batch_size * grad_accum_steps
    print(
        f"Training with block_size={block_size}, model={config.n_layer}x{config.n_embd}, "
        f"tokens/step≈{total_tokens_per_step:,}, optimizer steps={total_optimizer_steps}"
    )
    print(f"Starting training loop... This should take several minutes.", flush=True)
    import time
    start_time = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    while optimizer_step < total_optimizer_steps:
        optimizer.zero_grad(set_to_none=True)
        for micro_idx in range(grad_accum_steps):
            xb, yb = get_batch("train", block_size, micro_batch_size, device)
            ctx = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                if device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                logits, loss = model(xb, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced non-finite loss")
            loss = loss / grad_accum_steps

            # Log loss to console and W&B (before gradient accumulation division)
            if micro_idx == 0:
                actual_loss = loss.item() * grad_accum_steps
                # Print progress every 10 steps or at key milestones
                if optimizer_step % 10 == 0 or optimizer_step in [1, 5, 50, 100, 200, 500, 800]:
                    elapsed = time.time() - start_time
                    print(f"[TRAINING] Step {optimizer_step}/{total_optimizer_steps}: loss = {actual_loss:.4f} | Elapsed: {elapsed:.1f}s", flush=True)
                if optimizer_step % 10 == 0 and WANDB_ENABLED:
                    WANDB.log({"train_loss": actual_loss, "step": optimizer_step})
            if device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer_step += 1

        if optimizer_step <= warmup_steps:
            lr = base_lr * optimizer_step / warmup_steps
        elif optimizer_step > total_optimizer_steps:
            lr = min_lr
        else:
            progress = (optimizer_step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cosine
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    total_tokens = optimizer_step * total_tokens_per_step
    elapsed_time = time.time() - start_time
    print(f"Completed {optimizer_step} optimizer steps (~{total_tokens:,} tokens processed).")
    print(f"Training took {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)", flush=True)

    model.eval()
    eval_iters = 3 if fast_eval else 50
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch("val", block_size, micro_batch_size, device)
            ctx = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                if device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                logits, loss = model(xb, yb)
            losses.append(loss.item())

    val_loss = float(sum(losses) / len(losses))
    if WANDB_ENABLED:
        WANDB.log({"val_loss": val_loss})
    return val_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Override device based on CUDA availability if cuda is requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda":
        # Log GPU information for debugging
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    print(f"Attention dataset: {DATA_DIR}")

    module = load_module(Path(args.solution_path))
    if hasattr(module, "build_attention"):
        builder = module.build_attention
    elif hasattr(module, "AttentionBlock"):
        builder = lambda cfg: module.AttentionBlock(cfg)  # type: ignore[arg-type]
    else:
        raise AttributeError("Module must expose build_attention(config) or AttentionBlock class")

    val_loss = train_and_eval(builder, args.device)
    print(f"val_loss: {val_loss:.6f}")


if __name__ == "__main__":
    main()
