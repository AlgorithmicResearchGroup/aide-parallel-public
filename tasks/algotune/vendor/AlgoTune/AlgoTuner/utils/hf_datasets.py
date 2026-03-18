"""Helpers for pulling AlgoTune datasets from Hugging Face."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


def _raw_snapshot_download(**kwargs: Any) -> Any:
    """Call the raw huggingface_hub snapshot helper directly.

    This intentionally bypasses the public ``huggingface_hub.snapshot_download``
    symbol because external instrumentation may wrap that symbol and inject
    unsupported kwargs such as ``name='huggingface_hub.snapshot_download'``.
    """
    try:
        from huggingface_hub import _snapshot_download as snapshot_module
    except Exception as exc:
        raise RuntimeError(
            "Strict AlgoTune dataset download requires "
            "huggingface_hub._snapshot_download.snapshot_download"
        ) from exc

    snapshot_download = snapshot_module.snapshot_download
    logging.info(
        "Using raw HF snapshot_download implementation from %s",
        getattr(snapshot_module, "__file__", "<unknown>"),
    )
    return snapshot_download(**kwargs)


def _hf_compatible_tqdm():
    """Return a tqdm callable that tolerates HF-specific kwargs like `name`.

    huggingface_hub 1.7.x passes `name=...` to the provided tqdm class. Plain
    `tqdm.tqdm` in our environment rejects that kwarg, so wrap it and drop the
    unsupported field.
    """
    try:
        from tqdm import tqdm as base_tqdm
    except ImportError:
        return None

    class _HFTqdm(base_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("name", None)
            super().__init__(*args, **kwargs)

    return _HFTqdm


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv_token() -> str | None:
    # Try HF_TOKEN first (official HuggingFace env var), then fall back to old names
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Backward compatibility
    token = os.environ.get("HUGGING_FACE_TOKEN")
    if token:
        return token

    dotenv_path = _project_root() / ".env"
    if not dotenv_path.exists():
        return None

    try:
        for raw in dotenv_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            # Check for HF_TOKEN first, then HUGGING_FACE_TOKEN for backward compat
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_TOKEN="):
                _, value = line.split("=", 1)
                return value.strip().strip('"').strip("'")
    except Exception:
        return None

    return None


def _get_repo_id() -> str:
    return os.environ.get("ALGOTUNE_HF_DATASET", "oripress/AlgoTune")


def _get_revision() -> str:
    return os.environ.get("ALGOTUNE_HF_REVISION", "main")


def _get_cache_dir() -> Path:
    cache_dir = os.environ.get("ALGOTUNE_HF_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return _project_root() / ".hf_datasets"


def _get_local_snapshot_root(repo_id: str | None = None) -> Path:
    resolved_repo_id = repo_id or _get_repo_id()
    return _get_cache_dir() / resolved_repo_id.replace("/", "__")


def _is_lazy_mode() -> bool:
    """Check if lazy download mode is enabled (only download .jsonl initially)."""
    return os.environ.get("ALGOTUNE_HF_LAZY") == "1"


def _data_dir_has_task_dataset(data_dir: Path, task_name: str) -> bool:
    """Return True if data_dir appears to contain a dataset for task_name."""
    import glob

    pattern = f"{task_name}_T*ms_n*_size*_train.jsonl"
    for base in (data_dir / task_name, data_dir):
        if base.is_dir() and glob.glob(str(base / pattern)):
            return True
    return False


def _maybe_set_data_dir_for_hf(data_dir: Path, task_name: str | None) -> None:
    """Point DATA_DIR at HF cache when it is the only place with the dataset."""
    current = os.environ.get("DATA_DIR")
    if not current:
        os.environ["DATA_DIR"] = str(data_dir)
        logging.info("DATA_DIR not set; using HF dataset dir %s", data_dir)
        return

    if task_name:
        try:
            current_path = Path(current)
        except Exception:
            current_path = None
        if current_path is None or not _data_dir_has_task_dataset(current_path, task_name):
            os.environ["DATA_DIR"] = str(data_dir)
            logging.info(
                "DATA_DIR has no dataset for %s; switching to HF dataset dir %s",
                task_name,
                data_dir,
            )


def _resolve_cached_data_dir(task_name: str | None = None) -> Path | None:
    data_dir = _get_local_snapshot_root() / "data"
    if not data_dir.is_dir():
        return None

    if task_name:
        task_dir = data_dir / task_name
        if task_dir.is_dir():
            return task_dir
        if not _data_dir_has_task_dataset(data_dir, task_name):
            return None

    return data_dir


def ensure_hf_dataset(task_name: str | None = None) -> Path | None:
    """
    Ensure HF datasets are available locally and return the data directory to use.

    Returns:
        Path to the local "data" directory (or the task subdirectory), or None.
    """
    if os.environ.get("ALGOTUNE_HF_DISABLE") == "1":
        return None
    cached = _resolve_cached_data_dir(task_name)
    if cached is None:
        logging.warning(
            "HF dataset cache not present locally for repo=%s task=%s at %s. "
            "Strict runs require a pre-fetched local snapshot.",
            _get_repo_id(),
            task_name or "all",
            _get_local_snapshot_root(),
        )
        return None

    _maybe_set_data_dir_for_hf(cached.parent if task_name else cached, task_name)
    logging.info("HF dataset available locally at %s", cached)
    return cached


def fetch_hf_dataset(task_name: str | None = None, *, force: bool = False) -> Path:
    repo_id = _get_repo_id()
    revision = _get_revision()
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_dir = _get_local_snapshot_root(repo_id)
    allow_patterns = None
    lazy_mode = _is_lazy_mode()

    if lazy_mode:
        if task_name:
            allow_patterns = [f"data/{task_name}/**/*.jsonl"]
        else:
            allow_patterns = ["data/**/*.jsonl"]
    elif task_name:
        allow_patterns = [f"data/{task_name}/**"]

    token = _load_dotenv_token()
    mode_str = "metadata-only (lazy)" if lazy_mode else "full"
    logging.info(
        "HF dataset fetch requested (repo=%s, revision=%s, task=%s, force=%s, mode=%s)",
        repo_id,
        revision,
        task_name or "all",
        force,
        mode_str,
    )

    if lazy_mode:
        print(f"📥 Downloading {task_name or 'all tasks'} metadata from HuggingFace ({repo_id})...")
        print("   (Lazy mode: .npy files will be downloaded on-demand)")
    else:
        print(f"📥 Downloading {task_name or 'all tasks'} dataset from HuggingFace ({repo_id})...")
        print("   This may take several minutes for large datasets...")

    tqdm_class = _hf_compatible_tqdm()
    if tqdm_class is None:
        print("   (Install tqdm for progress bars: pip install tqdm)")

    hf_cache_dir = local_dir / ".hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _raw_snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        cache_dir=str(hf_cache_dir),
        allow_patterns=allow_patterns,
        token=token,
        force_download=force,
        tqdm_class=tqdm_class,
    )
    print(f"✓ Download complete! Dataset cached at {local_dir}")
    return Path(snapshot_path)


def download_npy_files(task_name: str) -> bool:
    """
    Download .npy files for a specific task (for lazy loading mode).
    Call this after ensure_hf_dataset() in lazy mode to get the actual data files.

    Args:
        task_name: The task name to download .npy files for

    Returns:
        True if successful, False otherwise
    """
    if not _is_lazy_mode():
        logging.info("Not in lazy mode, .npy files should already be downloaded")
        return True

    repo_id = _get_repo_id()
    revision = _get_revision()
    local_dir = _get_local_snapshot_root(repo_id)

    allow_patterns = [f"data/{task_name}/**/*.npy"]
    token = _load_dotenv_token()

    print(f"📥 Downloading {task_name} .npy files from HuggingFace...")
    print("   This may take several minutes for large datasets...")

    try:
        # Import tqdm for progress display if available
        tqdm_class = _hf_compatible_tqdm()
        if tqdm_class is None:
            tqdm_class = None

        # Set cache_dir to be within local_dir to avoid duplicating space.
        # Recent huggingface_hub versions removed local_dir_use_symlinks, so
        # rely on the default local_dir behavior here.
        hf_cache_dir = local_dir / ".hf_cache"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

        _raw_snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            cache_dir=str(hf_cache_dir),
            allow_patterns=allow_patterns,
            token=token,
            force_download=False,  # Use cached files if available
            tqdm_class=tqdm_class,
        )
        print(f"✓ Download complete for {task_name} .npy files!")
        return True
    except Exception as exc:
        logging.warning("HF .npy download failed for %s: %s", task_name, exc)
        print(f"❌ .npy download failed: {exc}")
        return False


def cleanup_npy_files(task_name: str | None = None) -> None:
    """
    Clean up .npy files to free disk space (manual cleanup utility).

    This keeps .jsonl files but removes large .npy files to free disk space.
    Use this when you need to free up disk space manually.

    Note: The HF cache is normally kept and reused across jobs. Only clean up
    when you need to free disk space and won't be running the same task again soon.

    Args:
        task_name: If provided, only clean up files for this task.
                   If None, clean up all .npy files.
    """
    cache_dir = _get_cache_dir()
    repo_id = _get_repo_id()
    local_dir = cache_dir / repo_id.replace("/", "__") / "data"

    if not local_dir.exists():
        logging.info("No HF cache directory found, nothing to clean up")
        return

    cleaned_count = 0
    freed_bytes = 0

    if task_name:
        # Clean up specific task
        task_dir = local_dir / task_name
        if task_dir.exists():
            for npy_file in task_dir.glob("**/*.npy"):
                size = npy_file.stat().st_size
                npy_file.unlink()
                cleaned_count += 1
                freed_bytes += size
            logging.info(
                "Cleaned up %d .npy files for task %s (freed %.2f GB)",
                cleaned_count,
                task_name,
                freed_bytes / 1e9,
            )
    else:
        # Clean up all tasks
        for npy_file in local_dir.glob("**/*.npy"):
            size = npy_file.stat().st_size
            npy_file.unlink()
            cleaned_count += 1
            freed_bytes += size
        logging.info("Cleaned up %d .npy files (freed %.2f GB)", cleaned_count, freed_bytes / 1e9)

    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} .npy files (freed {freed_bytes / 1e9:.2f} GB)")
    else:
        print("🧹 No .npy files found to clean up")
