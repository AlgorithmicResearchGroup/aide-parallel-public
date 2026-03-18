"""Helpers for strict KernelBench sweep orchestration and summary generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "coverage": {
        "num_experiments": 1,
        "num_iterations": 1,
        "cpus_per_experiment": 1,
        "gpu_fraction": 1.0,
    },
    "promotion": {
        "num_experiments": 2,
        "num_iterations": 2,
        "cpus_per_experiment": 1,
        "gpu_fraction": 1.0,
    },
    "final": {
        "num_experiments": 4,
        "num_iterations": 3,
        "cpus_per_experiment": 1,
        "gpu_fraction": 1.0,
    },
}

FAST_P_THRESHOLDS = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)


def sanitize_slug(value: str) -> str:
    slug = "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"


def default_campaign_id(profile: str, model: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{profile}_{sanitize_slug(model or 'default-model')}"


def compute_config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def load_manifest_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_effective_records_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    executed: dict[str, dict[str, Any]] = {}
    for record in records:
        task_name = record.get("task_name")
        if not task_name:
            continue
        latest[task_name] = record
        if record.get("status") != "skipped":
            executed[task_name] = record
    return {task_name: executed.get(task_name, record) for task_name, record in latest.items()}


def geometric_mean(values: list[float]) -> float | None:
    positives = [float(value) for value in values if isinstance(value, (int, float)) and float(value) > 0.0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def fast_p_score(rows: list[dict[str, Any]], threshold: float) -> float:
    if not rows:
        return 0.0
    hit_count = 0
    for row in rows:
        metric = row.get("best_metric")
        if row.get("correct") is True and isinstance(metric, (int, float)) and float(metric) > threshold:
            hit_count += 1
    return hit_count / len(rows)


def summarize_campaign(campaign_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign_path = Path(campaign_dir)
    manifest_records = load_manifest_records(campaign_path / "manifest.jsonl")
    latest = latest_effective_records_by_task(manifest_records)
    latest_rows = sorted(latest.values(), key=lambda row: row["task_name"])

    campaign_config = {}
    campaign_config_path = campaign_path / "campaign_config.json"
    if campaign_config_path.exists():
        campaign_config = json.loads(campaign_config_path.read_text(encoding="utf-8"))

    status_counts = Counter(row.get("status", "unknown") for row in latest_rows)
    succeeded = [row for row in latest_rows if row.get("status") == "succeeded"]
    executable = [row for row in latest_rows if row.get("status") != "skipped"]
    compiled_count = sum(1 for row in executable if row.get("compiled") is True)
    correct_count = sum(1 for row in executable if row.get("correct") is True)
    speedups = [float(row["best_metric"]) for row in succeeded if isinstance(row.get("best_metric"), (int, float))]

    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latest_rows:
        by_level[f"level{row.get('level', 'unknown')}"].append(row)

    level_summary = []
    for level_name, rows in sorted(by_level.items()):
        level_speedups = [
            float(row["best_metric"])
            for row in rows
            if row.get("status") == "succeeded" and isinstance(row.get("best_metric"), (int, float))
        ]
        level_summary.append(
            {
                "level": level_name,
                "total": len(rows),
                "succeeded": sum(1 for row in rows if row.get("status") == "succeeded"),
                "failed": sum(1 for row in rows if row.get("status") == "failed"),
                "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
                "geometric_mean_speedup": geometric_mean(level_speedups),
                "fast_p": {str(p): fast_p_score(rows, p) for p in FAST_P_THRESHOLDS},
            }
        )

    fast_p_metrics = {f"fast_{str(p).replace('.', '_')}": fast_p_score(latest_rows, p) for p in FAST_P_THRESHOLDS}

    summary = {
        "campaign_id": campaign_config.get("campaign_id", campaign_path.name),
        "profile": campaign_config.get("profile"),
        "selected_task_count": campaign_config.get("selected_task_count", len(latest_rows)),
        "latest_task_count": len(latest_rows),
        "attempt_count": len(manifest_records),
        "status_counts": dict(status_counts),
        "succeeded_count": len(succeeded),
        "failed_count": status_counts.get("failed", 0),
        "skipped_count": status_counts.get("skipped", 0),
        "success_rate": (len(succeeded) / len(latest_rows)) if latest_rows else 0.0,
        "compiled_rate": (compiled_count / len(executable)) if executable else 0.0,
        "correct_rate": (correct_count / len(executable)) if executable else 0.0,
        "mean_speedup": (sum(speedups) / len(speedups)) if speedups else None,
        "median_speedup": (
            sorted(speedups)[len(speedups) // 2]
            if speedups and len(speedups) % 2 == 1
            else (
                (sorted(speedups)[len(speedups) // 2 - 1] + sorted(speedups)[len(speedups) // 2]) / 2
                if speedups
                else None
            )
        ),
        "geometric_mean_speedup_correct": geometric_mean(speedups),
        "level_summary": level_summary,
        "top_tasks": sorted(
            succeeded,
            key=lambda row: float(row.get("best_metric", float("-inf"))),
            reverse=True,
        )[:20],
        "generated_at": datetime.now().isoformat(),
    }
    summary.update(fast_p_metrics)
    return summary, latest_rows


def write_campaign_summary(campaign_dir: str | Path) -> dict[str, Any]:
    campaign_path = Path(campaign_dir)
    campaign_path.mkdir(parents=True, exist_ok=True)
    summary, latest_rows = summarize_campaign(campaign_path)
    (campaign_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fieldnames = [
        "task_name",
        "level",
        "category",
        "status",
        "best_metric",
        "compiled",
        "correct",
        "failure_reason",
        "artifact_dir",
        "config_hash",
    ]
    with (campaign_path / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in latest_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return summary


def sync_campaign_summary_to_mlflow(
    campaign_dir: str | Path,
    *,
    tracking_experiment: str | None = None,
) -> dict[str, Any] | None:
    if os.getenv("AIDE_ENABLE_MLFLOW", "0") != "1":
        return None

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        return None

    from src.mlflow_integration import resolve_experiment_name, resolve_tracking_uri

    campaign_path = Path(campaign_dir)
    summary, _ = summarize_campaign(campaign_path)
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = resolve_experiment_name(tracking_experiment, "kernelbench")

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
        experiment_id = (
            client.create_experiment(experiment_name, artifact_location=artifact_root)
            if artifact_root
            else client.create_experiment(experiment_name)
        )
    else:
        experiment_id = experiment.experiment_id

    state_path = campaign_path / ".mlflow_summary_run.json"
    run_id = None
    if state_path.exists():
        try:
            run_id = json.loads(state_path.read_text(encoding="utf-8")).get("run_id")
        except Exception:
            run_id = None

    if run_id:
        try:
            run = client.get_run(run_id)
            if run.info.experiment_id != experiment_id:
                run_id = None
        except Exception:
            run_id = None

    if not run_id:
        run = client.create_run(
            experiment_id=experiment_id,
            tags={
                "mlflow.runName": f"kernelbench_sweep_{summary['campaign_id']}",
                "task_type": "kernelbench",
                "run_kind": "kernelbench_sweep",
                "campaign_id": summary["campaign_id"],
                "profile": str(summary.get("profile")),
            },
        )
        run_id = run.info.run_id
        state_path.write_text(json.dumps({"run_id": run_id}, indent=2), encoding="utf-8")

    timestamp_ms = int(datetime.now().timestamp() * 1000)
    step = int(summary.get("latest_task_count") or 0)
    metric_keys = [
        "selected_task_count",
        "latest_task_count",
        "attempt_count",
        "succeeded_count",
        "failed_count",
        "skipped_count",
        "success_rate",
        "compiled_rate",
        "correct_rate",
        "mean_speedup",
        "median_speedup",
        "geometric_mean_speedup_correct",
        *[f"fast_{str(p).replace('.', '_')}" for p in FAST_P_THRESHOLDS],
    ]
    for key in metric_keys:
        value = summary.get(key)
        if isinstance(value, bool):
            metric_value = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            metric_value = float(value)
        else:
            continue
        client.log_metric(run_id, key, metric_value, timestamp_ms, step)

    client.set_tag(run_id, "campaign_id", summary["campaign_id"])
    client.set_tag(run_id, "summary.generated_at", summary.get("generated_at", ""))

    for artifact_name in ("summary.json", "summary.csv"):
        artifact_path = campaign_path / artifact_name
        if artifact_path.exists():
            client.log_artifact(run_id, str(artifact_path))
    return summary


def reconcile_campaign_runs_in_mlflow(
    campaign_dir: str | Path,
    *,
    tracking_experiment: str | None = None,
    finalize_incomplete: bool = False,
) -> dict[str, int] | None:
    if os.getenv("AIDE_ENABLE_MLFLOW", "0") != "1":
        return None

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        return None

    from src.mlflow_integration import (
        build_task_outcome_payload,
        resolve_experiment_name,
        resolve_tracking_uri,
    )

    manifest_records = load_manifest_records(Path(campaign_dir) / "manifest.jsonl")
    latest_records = latest_effective_records_by_task(manifest_records)

    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(resolve_experiment_name(tracking_experiment, "kernelbench"))
    if experiment is None:
        return {"closed_from_manifest": 0, "closed_incomplete": 0}

    runs = client.search_runs([experiment.experiment_id], max_results=5000)
    timestamp_ms = int(datetime.now().timestamp() * 1000)
    closed_from_manifest = 0
    closed_incomplete = 0

    for run in runs:
        if run.data.tags.get("run_kind") == "kernelbench_sweep":
            continue
        task_id = run.data.tags.get("task_id")
        if not task_id:
            continue
        record = latest_records.get(task_id)
        if record is None and not finalize_incomplete:
            continue

        if record is not None:
            metric_value = record.get("best_metric")
            metric_value = float(metric_value) if isinstance(metric_value, (int, float)) else None
            status = record.get("status", "failed")
            compiled = record.get("compiled")
            correct = record.get("correct")
            error = record.get("failure_reason")
        else:
            metric_value = None
            status = "failed"
            compiled = None
            correct = None
            error = "Campaign ended before task produced a manifest record"

        _, metrics, outcome_tags = build_task_outcome_payload(
            task_type="kernelbench",
            status=status,
            metric_value=metric_value,
            compiled=compiled,
            correct=correct,
            error=error,
        )
        outcome_tags["task_id"] = task_id
        for key, value in metrics.items():
            if isinstance(value, bool):
                numeric = float(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric = float(value)
            else:
                continue
            client.log_metric(run.info.run_id, key, numeric, timestamp_ms, int(record.get("attempt") or 0) if record else 0)
        for key, value in outcome_tags.items():
            if value is not None:
                client.set_tag(run.info.run_id, key, str(value)[:5000])
        if run.info.status == "RUNNING":
            client.set_terminated(run.info.run_id, status="FINISHED" if status == "succeeded" else "FAILED", end_time=timestamp_ms)
            if record is not None:
                closed_from_manifest += 1
            else:
                closed_incomplete += 1

    return {
        "closed_from_manifest": closed_from_manifest,
        "closed_incomplete": closed_incomplete,
    }
