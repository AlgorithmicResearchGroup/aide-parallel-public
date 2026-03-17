#!/usr/bin/env python3
"""Generate summary artifacts for an AlgoTune sweep campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algotune_sweep import write_campaign_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an AlgoTune campaign")
    parser.add_argument("campaign_dir", help="Path to runs/algotune/<campaign-id>")
    args = parser.parse_args()

    summary = write_campaign_summary(args.campaign_dir)
    print(f"Campaign: {summary['campaign_id']}")
    print(
        "Status counts: "
        f"succeeded={summary['status_counts'].get('succeeded', 0)} "
        f"failed={summary['status_counts'].get('failed', 0)} "
        f"skipped={summary['status_counts'].get('skipped', 0)}"
    )
    print(f"Success rate: {summary['success_rate']:.2%}")
    if summary.get("median_speedup") is not None:
        print(f"Median speedup: {summary['median_speedup']:.4f}")
    if summary.get("mean_speedup") is not None:
        print(f"Mean speedup: {summary['mean_speedup']:.4f}")
    print(f"Summary JSON: {Path(args.campaign_dir) / 'summary.json'}")
    print(f"Summary CSV: {Path(args.campaign_dir) / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
