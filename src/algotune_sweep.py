"""Helpers for AlgoTune sweep orchestration and summary generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "coverage": {
        "num_experiments": 1,
        "num_iterations": 1,
        "steps": 4,
        "cpus_per_experiment": 2,
    },
    "promotion": {
        "num_experiments": 3,
        "num_iterations": 2,
        "steps": 8,
        "cpus_per_experiment": 2,
    },
    "final": {
        "num_experiments": 3,
        "num_iterations": 2,
        "steps": 10,
        "cpus_per_experiment": 2,
    },
}


def sanitize_slug(value: str) -> str:
    allowed = []
    for char in value.lower():
        allowed.append(char if char.isalnum() else "-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"


def default_campaign_id(profile: str, model: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = sanitize_slug(model or "default-model")
    return f"{timestamp}_{profile}_{model_slug}"


def compute_config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def load_manifest_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def latest_records_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        task_name = record.get("task_name")
        if task_name:
            latest[task_name] = record
    return latest


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
    effective: dict[str, dict[str, Any]] = {}
    for task_name, record in latest.items():
        effective[task_name] = executed.get(task_name, record)
    return effective


def harmonic_mean(values: list[float]) -> float | None:
    positives = [float(value) for value in values if isinstance(value, (int, float)) and float(value) > 0.0]
    if not positives:
        return None
    return len(positives) / sum(1.0 / value for value in positives)


def summarize_campaign(campaign_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign_path = Path(campaign_dir)
    manifest_path = campaign_path / "manifest.jsonl"
    records = load_manifest_records(manifest_path)
    latest = latest_effective_records_by_task(records)

    campaign_config_path = campaign_path / "campaign_config.json"
    campaign_config = {}
    if campaign_config_path.exists():
        campaign_config = json.loads(campaign_config_path.read_text(encoding="utf-8"))

    latest_rows = sorted(latest.values(), key=lambda row: row["task_name"])
    status_counts = Counter(row.get("status", "unknown") for row in latest_rows)
    succeeded = [row for row in latest_rows if row.get("status") == "succeeded"]
    failed = [row for row in latest_rows if row.get("status") == "failed"]
    executable_rows = [row for row in latest_rows if row.get("status") != "skipped"]

    metrics = [
        float(row["best_metric"])
        for row in succeeded
        if isinstance(row.get("best_metric"), (int, float))
    ]
    paper_scores = [
        max(1.0, float(row["best_metric"]))
        if row.get("status") == "succeeded" and isinstance(row.get("best_metric"), (int, float))
        else 1.0
        for row in latest_rows
    ]
    speedup_gt_1_count = sum(1 for value in metrics if value > 1.0)
    speedup_positive_count = sum(1 for value in metrics if value > 0.0)
    correct_count = sum(1 for row in executable_rows if row.get("correct") is True)
    compiled_count = sum(1 for row in executable_rows if row.get("compiled") is True)
    failure_reasons = Counter(row.get("failure_reason") or "unknown" for row in failed)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latest_rows:
        by_category[row.get("category", "unknown")].append(row)

    category_summary = []
    for category, rows in sorted(by_category.items()):
        category_success = [
            float(row["best_metric"])
            for row in rows
            if row.get("status") == "succeeded"
            and isinstance(row.get("best_metric"), (int, float))
        ]
        category_paper_scores = [
            max(1.0, float(row["best_metric"]))
            if row.get("status") == "succeeded" and isinstance(row.get("best_metric"), (int, float))
            else 1.0
            for row in rows
        ]
        category_summary.append(
            {
                "category": category,
                "total": len(rows),
                "succeeded": sum(1 for row in rows if row.get("status") == "succeeded"),
                "failed": sum(1 for row in rows if row.get("status") == "failed"),
                "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
                "mean_speedup": mean(category_success) if category_success else None,
                "median_speedup": median(category_success) if category_success else None,
                "paper_harmonic_mean_score": harmonic_mean(category_paper_scores),
            }
        )

    successful_ranked = sorted(
        succeeded,
        key=lambda row: float(row.get("best_metric", float("-inf"))),
        reverse=True,
    )

    summary = {
        "campaign_id": campaign_config.get("campaign_id", campaign_path.name),
        "profile": campaign_config.get("profile"),
        "at_mode": campaign_config.get("config", {}).get("at_mode"),
        "selected_task_count": campaign_config.get("selected_task_count", len(latest_rows)),
        "latest_task_count": len(latest_rows),
        "attempt_count": len(records),
        "status_counts": dict(status_counts),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "skipped_count": status_counts.get("skipped", 0),
        "success_rate": (len(succeeded) / len(latest_rows)) if latest_rows else 0.0,
        "correct_rate": (correct_count / len(executable_rows)) if executable_rows else 0.0,
        "compiled_rate": (compiled_count / len(executable_rows)) if executable_rows else 0.0,
        "mean_speedup": mean(metrics) if metrics else None,
        "median_speedup": median(metrics) if metrics else None,
        "paper_harmonic_mean_score": harmonic_mean(paper_scores),
        "speedup_gt_1_count": speedup_gt_1_count,
        "speedup_gt_1_rate": (speedup_gt_1_count / len(succeeded)) if succeeded else 0.0,
        "positive_speedup_count": speedup_positive_count,
        "positive_speedup_rate": (speedup_positive_count / len(succeeded)) if succeeded else 0.0,
        "category_summary": category_summary,
        "top_tasks": successful_ranked[:20],
        "bottom_tasks": list(reversed(successful_ranked[-20:])),
        "failure_reasons": dict(failure_reasons),
        "generated_at": datetime.now().isoformat(),
    }
    if summary["at_mode"] == "benchmark_strict":
        summary["final_success_rate"] = summary["success_rate"]
        summary["final_correct_rate"] = summary["correct_rate"]
        summary["final_speedup_gt_1_count"] = summary["speedup_gt_1_count"]
        summary["final_speedup_gt_1_rate"] = summary["speedup_gt_1_rate"]
        summary["final_paper_harmonic_mean_score"] = summary["paper_harmonic_mean_score"]
    return summary, latest_rows


def write_campaign_summary(campaign_dir: str | Path) -> dict[str, Any]:
    campaign_path = Path(campaign_dir)
    campaign_path.mkdir(parents=True, exist_ok=True)
    summary, latest_rows = summarize_campaign(campaign_path)

    summary_path = campaign_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = campaign_path / "summary.csv"
    fieldnames = [
        "task_name",
        "category",
        "status",
        "at_mode",
        "report_phase",
        "best_metric",
        "search_metric",
        "final_metric",
        "search_status",
        "final_status",
        "correct",
        "compiled",
        "attempt",
        "failure_reason",
        "artifact_dir",
        "config_hash",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
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
    experiment_name = resolve_experiment_name(tracking_experiment, "algotune")

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
        if artifact_root:
            experiment_id = client.create_experiment(
                experiment_name,
                artifact_location=artifact_root,
            )
        else:
            experiment_id = client.create_experiment(experiment_name)
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
                "mlflow.runName": f"algotune_sweep_{summary['campaign_id']}",
                "task_type": "algotune",
                "run_kind": "algotune_sweep",
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
        "correct_rate",
        "compiled_rate",
        "mean_speedup",
        "median_speedup",
        "paper_harmonic_mean_score",
        "speedup_gt_1_count",
        "speedup_gt_1_rate",
        "positive_speedup_count",
        "positive_speedup_rate",
        "final_success_rate",
        "final_correct_rate",
        "final_speedup_gt_1_count",
        "final_speedup_gt_1_rate",
        "final_paper_harmonic_mean_score",
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

    tags = {
        "campaign_id": summary["campaign_id"],
        "profile": str(summary.get("profile")),
        "summary.generated_at": summary.get("generated_at"),
        "summary.top_success_task": (
            summary["top_tasks"][0]["task_name"] if summary.get("top_tasks") else ""
        ),
        "summary.top_success_metric": (
            str(summary["top_tasks"][0].get("best_metric"))
            if summary.get("top_tasks")
            else ""
        ),
    }
    for key, value in tags.items():
        if value is not None:
            client.set_tag(run_id, key, str(value)[:5000])

    summary_path = campaign_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = campaign_path / "summary.csv"
    if summary_path.exists():
        client.log_artifact(run_id, str(summary_path))
    if csv_path.exists():
        client.log_artifact(run_id, str(csv_path))
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

    campaign_path = Path(campaign_dir)
    manifest_records = load_manifest_records(campaign_path / "manifest.jsonl")
    latest_records = latest_effective_records_by_task(manifest_records)

    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = resolve_experiment_name(tracking_experiment, "algotune")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {
            "closed_from_manifest": 0,
            "closed_incomplete": 0,
        }

    runs = client.search_runs([experiment.experiment_id], max_results=5000)
    closed_from_manifest = 0
    closed_incomplete = 0
    timestamp_ms = int(datetime.now().timestamp() * 1000)

    for run in runs:
        if run.data.tags.get("run_kind") == "algotune_sweep":
            continue

        task_id = run.data.tags.get("task_id")
        if not task_id:
            continue

        needs_backfill = (
            run.data.tags.get("task_status") is None
            or run.data.tags.get("outcome") is None
            or run.data.tags.get("metric_available") is None
            or (
                run.data.tags.get("metric_name") == "final_speedup"
                and "final_speedup_or_zero" not in run.data.metrics
            )
            or (
                run.data.tags.get("metric_name") == "final_speedup"
                and run.data.tags.get("task_status") == "failed"
                and "metric_available" not in run.data.metrics
            )
        )
        if run.info.status != "RUNNING" and not needs_backfill:
            continue

        record = latest_records.get(task_id)
        if record is None and not finalize_incomplete:
            continue

        if record is not None:
            metric_value = record.get("best_metric")
            if not isinstance(metric_value, (int, float)):
                metric_value = None
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

        metric_key, metrics, outcome_tags = build_task_outcome_payload(
            task_type="algotune",
            status=status,
            metric_value=metric_value,
            compiled=compiled,
            correct=correct,
            error=error,
        )
        outcome_tags["task_id"] = task_id
        outcome_tags["cleanup_reason"] = (
            "reconciled_from_manifest" if record is not None else "finalized_incomplete_running_run"
        )

        step = 0
        if record is not None:
            try:
                step = int(record.get("attempt") or 0)
            except Exception:
                step = 0

        for key, value in metrics.items():
            if isinstance(value, bool):
                metric_numeric = float(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                metric_numeric = float(value)
            else:
                continue
            client.log_metric(run.info.run_id, key, metric_numeric, timestamp_ms, step)

        for key, value in outcome_tags.items():
            if value is None:
                continue
            client.set_tag(run.info.run_id, key, str(value)[:5000])

        if run.info.status == "RUNNING":
            terminate_status = "FINISHED" if status == "succeeded" else "FAILED"
            client.set_terminated(run.info.run_id, status=terminate_status, end_time=timestamp_ms)
            if record is not None:
                closed_from_manifest += 1
            else:
                closed_incomplete += 1

    return {
        "closed_from_manifest": closed_from_manifest,
        "closed_incomplete": closed_incomplete,
    }
