"""
Download + tokenize the EleutherAI Wikipedia dataset for nanoGPT-style training.
Outputs train.bin / val.bin (uint16 token IDs) and a meta.pkl with vocab size.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def iter_token_ids(split, tokenizer: GPT2TokenizerFast, text_key: str):
    """Yield lists of token IDs for each example in the split."""
    for example in split:
        text = example[text_key]
        if not text:
            continue
        yield tokenizer.encode(text, add_special_tokens=False)


def write_tokens(path: Path, token_iter):
    """Stream token IDs to a binary file as uint16."""
    with path.open("wb") as f:
        for token_ids in token_iter:
            arr = np.array(token_ids, dtype=np.uint16)
            arr.tofile(f)


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-v1")

    # Original dataset only has 'train'; create val split
    split = dataset["train"].train_test_split(
        train_size=args.train_fraction,
        test_size=args.val_fraction,
        seed=args.seed,
        shuffle=True,
    )
    split["val"] = split.pop("test")

    # Tokenize + write
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    write_tokens(train_path, iter_token_ids(split["train"], tokenizer, args.text_key))
    write_tokens(val_path, iter_token_ids(split["val"], tokenizer, args.text_key))

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_name": "gpt2",
        "text_key": args.text_key,
    }
    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    print(f"Wrote {train_path} and {val_path}")
    print(f"Meta saved to {out_dir / 'meta.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/wiki", help="Destination directory")
    parser.add_argument("--text-key", default="page", help="Column containing text")
    parser.add_argument("--train-fraction", type=float, default=0.92)
    parser.add_argument("--val-fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=2357)
    main(parser.parse_args())
