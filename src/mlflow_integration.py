"""MLflow integration for AIDE experiments."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    root = os.environ.get("AIDE_PROJECT_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def default_tracking_uri() -> str:
    return (_project_root() / "mlruns").resolve().as_uri()


def resolve_tracking_uri(override: str | None = None) -> str:
    if override:
        return override
    return os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri()


def resolve_experiment_name(override: str | None = None, task_type: str | None = None) -> str:
    if override:
        return override
    if os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        return os.environ["MLFLOW_EXPERIMENT_NAME"]
    if task_type:
        return f"aide-{task_type}"
    return "aide-public"


RUN_OVERVIEW_TAG_KEYS = {
    "task_type",
    "task_id",
    "experiment_idx",
    "iteration",
    "model",
    "feedback_model",
    "gpu_id",
    "gpu_name",
    "run_kind",
}


def _slugify_run_value(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-_.")
    return text[:80] or "unknown"


def _stringify_trace_metadata(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:5000]
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)[:5000]


def build_task_outcome_payload(
    *,
    task_type: str,
    status: str,
    metric_value: float | None,
    compiled: bool | None = None,
    correct: bool | None = None,
    error: str | None = None,
    phase: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if phase == "search":
        metric_key = (
            "search_mean_speedup" if task_type in {"kernel", "kernelbench", "algotune"} else "search_val_loss"
        )
    elif phase == "final_test":
        metric_key = (
            "final_test_mean_speedup"
            if task_type in {"kernel", "kernelbench", "algotune"}
            else "final_test_val_loss"
        )
    else:
        metric_key = "final_speedup" if task_type in {"kernel", "kernelbench", "algotune"} else "final_val_loss"
    metric_available = metric_value is not None
    prefix = f"{phase}_" if phase else ""
    metrics: dict[str, Any] = {
        f"{prefix}task_passed" if phase else "task_passed": 1.0 if status == "succeeded" else 0.0,
        f"{prefix}task_failed" if phase else "task_failed": 1.0 if status == "failed" else 0.0,
        f"{prefix}metric_available" if phase else "metric_available": 1.0 if metric_available else 0.0,
        f"{prefix}metric_missing" if phase else "metric_missing": 0.0 if metric_available else 1.0,
        f"{prefix}has_evaluation_error" if phase else "has_evaluation_error": 1.0 if error is not None else 0.0,
    }
    if metric_value is not None:
        metrics[metric_key] = metric_value
    if metric_key in {"final_speedup", "search_mean_speedup", "final_test_mean_speedup"}:
        zero_key = "final_speedup_or_zero"
        if phase == "search":
            zero_key = "search_mean_speedup_or_zero"
        elif phase == "final_test":
            zero_key = "final_test_mean_speedup_or_zero"
        metrics[zero_key] = float(metric_value) if metric_value is not None else 0.0
    if compiled is not None:
        metrics[f"{prefix}compiled" if phase else "compiled"] = 1.0 if compiled else 0.0
    if correct is not None:
        metrics[f"{prefix}correct" if phase else "correct"] = 1.0 if correct else 0.0

    outcome_tags = {
        f"{prefix}status" if phase else "task_status": status,
        "outcome": status,
        f"{prefix}metric_name" if phase else "metric_name": metric_key,
        f"{prefix}evaluation_error" if phase else "evaluation_error": error,
        f"{prefix}compiled_status" if phase else "compiled_status": compiled,
        f"{prefix}correct_status" if phase else "correct_status": correct,
        f"{prefix}metric_available" if phase else "metric_available": str(metric_available),
    }
    return metric_key, metrics, outcome_tags


class AIDEMLflowLogger:
    """Persistent MLflow logger for a single experiment actor."""

    def __init__(
        self,
        run_name: str | None = None,
        gpu_id: int | None = None,
        experiment_config: dict[str, Any] | None = None,
        tracking_experiment: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        experiment_config = experiment_config or {}
        task_type = experiment_config.get("task_type", "attention")
        self.enabled = False
        self.run_id: str | None = None
        self.active_run = None
        self.run_context = {
            "task_type": experiment_config.get("task_type", task_type),
            "task_id": experiment_config.get("task_id"),
            "experiment_idx": experiment_config.get("experiment_idx"),
            "iteration": experiment_config.get("iteration"),
            "model": experiment_config.get("model"),
            "feedback_model": experiment_config.get("feedback_model"),
            "run_kind": experiment_config.get("task_type", task_type),
        }

        try:
            import mlflow
            from mlflow import MlflowClient
        except ImportError:
            print("MLflow is not installed. Tracking disabled.")
            return

        self.mlflow = mlflow
        self.tracking_uri = resolve_tracking_uri()
        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_name = resolve_experiment_name(tracking_experiment, task_type)
        self.experiment_id = self._ensure_experiment()

        run_name = self._derive_run_name(run_name, gpu_id)
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            device_suffix = f"gpu{gpu_id}" if gpu_id is not None and gpu_id >= 0 else "cpu"
            run_name = f"aide_{task_type}_{device_suffix}_{timestamp}"

        tags = {
            "mlflow.runName": run_name,
            "task_type": str(task_type),
            "run_kind": str(task_type),
            "tracking_uri": self.tracking_uri,
        }
        for key in RUN_OVERVIEW_TAG_KEYS:
            value = self.run_context.get(key)
            if value is not None and value != "":
                tags[key] = str(value)
        if parent_run_id:
            tags["mlflow.parentRunId"] = parent_run_id
        if gpu_id is not None and gpu_id >= 0:
            tags["gpu_id"] = str(gpu_id)
        elif "gpu_id" in experiment_config:
            tags["gpu_id"] = str(experiment_config["gpu_id"])

        try:
            run = self.client.create_run(experiment_id=self.experiment_id, tags=tags)
        except Exception as exc:
            print(f"Failed to initialize MLflow run: {exc}")
            return

        self.run_id = run.info.run_id
        self.enabled = True
        self._activate_run()
        self._log_initial_config(experiment_config)

    def _derive_run_name(self, run_name: str | None, gpu_id: int | None) -> str | None:
        base_name = run_name
        if base_name is None:
            return None
        suffixes: list[str] = []
        task_id = self.run_context.get("task_id")
        if task_id:
            suffixes.append(_slugify_run_value(task_id))
        experiment_idx = self.run_context.get("experiment_idx")
        if experiment_idx is not None and f"exp{experiment_idx}" not in base_name:
            suffixes.append(f"idx{experiment_idx}")
        if suffixes:
            base_name = "_".join([base_name, *suffixes])
        if gpu_id is not None and gpu_id >= 0 and f"gpu{gpu_id}" not in base_name:
            base_name = f"{base_name}_gpu{gpu_id}"
        return base_name

    def _activate_run(self) -> None:
        if not self.enabled or not self.run_id:
            return
        active_run = self.mlflow.active_run()
        if active_run and active_run.info.run_id == self.run_id:
            self.active_run = active_run
            return
        if active_run and active_run.info.run_id != self.run_id:
            self.mlflow.end_run(status="FINISHED")
        self.active_run = self.mlflow.start_run(run_id=self.run_id)

    def _ensure_experiment(self) -> str:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id

        artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
        try:
            if artifact_root:
                return self.client.create_experiment(
                    self.experiment_name,
                    artifact_location=artifact_root,
                )
            return self.client.create_experiment(self.experiment_name)
        except Exception:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise
            return experiment.experiment_id

    def _log_initial_config(self, experiment_config: dict[str, Any]) -> None:
        if not self.enabled or not self.run_id:
            return

        serializable_config = {
            "timestamp": datetime.now().isoformat(),
            **experiment_config,
        }
        params: dict[str, str] = {}
        tags: dict[str, str] = {}
        for key, value in serializable_config.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                params[key] = str(value)
                if key in RUN_OVERVIEW_TAG_KEYS:
                    tags[key] = str(value)
            else:
                tags[key] = json.dumps(value, sort_keys=True)

        for key, value in params.items():
            try:
                self.client.log_param(self.run_id, key, value[:500])
            except Exception:
                tags[key] = value

        for key, value in tags.items():
            self.client.set_tag(self.run_id, key, value[:5000])

        self._log_text_artifact(
            json.dumps(serializable_config, indent=2, sort_keys=True),
            "config/config.json",
        )

    def _log_text_artifact(self, text: str, artifact_path: str) -> None:
        if not self.enabled or not self.run_id:
            return

        with tempfile.TemporaryDirectory(prefix="aide-mlflow-") as temp_dir:
            output_path = Path(temp_dir) / Path(artifact_path).name
            output_path.write_text(text, encoding="utf-8")
            artifact_dir = str(Path(artifact_path).parent)
            self.client.log_artifact(self.run_id, str(output_path), artifact_path=artifact_dir)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or not self.run_id:
            return

        timestamp_ms = int(time.time() * 1000)
        for key, value in metrics.items():
            if isinstance(value, bool):
                metric_value = float(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                metric_value = float(value)
            else:
                self.client.set_tag(self.run_id, f"latest.{key}", str(value))
                continue
            self.client.log_metric(self.run_id, key, metric_value, timestamp_ms, step or 0)

    def log_code(self, code: str, name: str = "generated_code", step: int = 0) -> None:
        self._log_text_artifact(code, f"code/{name}_step{step}.py")

    def set_tags(self, tags: dict[str, Any]) -> None:
        if not self.enabled or not self.run_id:
            return
        for key, value in tags.items():
            if value is None:
                continue
            self.client.set_tag(self.run_id, key, str(value)[:5000])

    def log_experiment_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled or not self.run_id:
            return

        numeric_summary = {
            f"summary.{key}": value
            for key, value in summary.items()
            if isinstance(value, (int, float, bool)) and math.isfinite(float(value))
        }
        if numeric_summary:
            self.log_metrics(numeric_summary)

        for key, value in summary.items():
            if key == "best_code" and value:
                self._log_text_artifact(str(value), "code/best_code.py")
            elif not isinstance(value, (int, float, bool)) or not math.isfinite(float(value)):
                self.client.set_tag(self.run_id, f"summary.{key}", str(value))

    def log_task_outcome(
        self,
        *,
        task_type: str,
        status: str,
        metric_value: float | None,
        compiled: bool | None = None,
        correct: bool | None = None,
        error: str | None = None,
        step: int | None = None,
        phase: str | None = None,
    ) -> None:
        metric_key, metrics, outcome_tags = build_task_outcome_payload(
            task_type=task_type,
            status=status,
            metric_value=metric_value,
            compiled=compiled,
            correct=correct,
            error=error,
            phase=phase,
        )
        self.log_metrics(metrics, step=step)
        if self.run_context.get("task_id") is not None:
            outcome_tags["task_id"] = self.run_context["task_id"]
        if self.run_context.get("experiment_idx") is not None:
            outcome_tags["experiment_idx"] = self.run_context["experiment_idx"]
        self.set_tags(outcome_tags)
        self._log_task_outcome_trace(
            task_type=task_type,
            status=status,
            metric_key=metric_key,
            metric_value=metric_value,
            compiled=compiled,
            correct=correct,
            error=error,
            step=step,
            phase=phase,
        )

    def _log_task_outcome_trace(
        self,
        *,
        task_type: str,
        status: str,
        metric_key: str,
        metric_value: float | None,
        compiled: bool | None,
        correct: bool | None,
        error: str | None,
        step: int | None,
        phase: str | None,
    ) -> None:
        if not self.enabled or not self.run_id:
            return

        self._activate_run()
        _, trace_metrics, _ = build_task_outcome_payload(
            task_type=task_type,
            status=status,
            metric_value=metric_value,
            compiled=compiled,
            correct=correct,
            error=error,
            phase=phase,
        )
        speedup_zero_key = "final_speedup_or_zero"
        if phase == "search":
            speedup_zero_key = "search_mean_speedup_or_zero"
        elif phase == "final_test":
            speedup_zero_key = "final_test_mean_speedup_or_zero"

        trace_tags = {
            key: str(value)
            for key, value in self.run_context.items()
            if key in {"task_type", "task_id", "experiment_idx", "iteration"} and value is not None
        }
        trace_tags["task_status"] = status
        if phase:
            trace_tags["phase"] = phase

        outcome = {
            "task_type": task_type,
            "phase": phase,
            "status": status,
            "metric_name": metric_key,
            "metric_value": metric_value,
            "metric_available": trace_metrics.get(f"{phase}_metric_available" if phase else "metric_available"),
            "metric_missing": trace_metrics.get(f"{phase}_metric_missing" if phase else "metric_missing"),
            speedup_zero_key: trace_metrics.get(speedup_zero_key),
            "compiled": compiled,
            "correct": correct,
            "error": error,
            "step": step,
        }
        response_preview = status
        if metric_value is not None:
            response_preview = f"{status} | {metric_key}={metric_value:.4f}"

        try:
            @self.mlflow.trace(
                name="aide.task.outcome",
                attributes={
                    **trace_tags,
                    "metric_name": metric_key,
                    "task_type": task_type,
                },
            )
            def _emit_task_outcome(payload: dict[str, Any], active_step: int | None) -> dict[str, Any]:
                self.mlflow.update_current_trace(
                    tags=trace_tags,
                    metadata={
                        "task_type": _stringify_trace_metadata(task_type),
                        "task_status": _stringify_trace_metadata(status),
                        "metric_name": _stringify_trace_metadata(metric_key),
                        "metric_value": _stringify_trace_metadata(metric_value),
                        "metric_available": _stringify_trace_metadata(
                            trace_metrics.get(f"{phase}_metric_available" if phase else "metric_available")
                        ),
                        speedup_zero_key: _stringify_trace_metadata(trace_metrics.get(speedup_zero_key)),
                        "compiled": _stringify_trace_metadata(compiled),
                        "correct": _stringify_trace_metadata(correct),
                        "error": _stringify_trace_metadata(error),
                        "step": _stringify_trace_metadata(step),
                        **{
                            key: _stringify_trace_metadata(value)
                            for key, value in trace_tags.items()
                        },
                    },
                    response_preview=response_preview,
                    state="OK" if status == "succeeded" else "ERROR",
                )
                return {
                    "run_id": self.run_id,
                    "step": active_step,
                    **payload,
                }

            _emit_task_outcome(outcome, step)
        except Exception:
            # Trace visibility should never break experiment execution.
            return

    def log_evaluation(
        self,
        *,
        speedup: float | None = None,
        execution_time: float | None = None,
        val_loss: float | None = None,
        training_time: float | None = None,
        eval_status: str | None = None,
        step: int | None = None,
    ) -> None:
        metrics: dict[str, Any] = {}
        if speedup is not None:
            metrics["speedup"] = speedup
            metrics["best_speedup_so_far"] = speedup
        if execution_time is not None:
            metrics["kernel_execution_time"] = execution_time
        if val_loss is not None:
            metrics["val_loss"] = val_loss
            metrics["best_val_loss_so_far"] = val_loss
        if training_time is not None:
            metrics["training_time"] = training_time
        if eval_status is not None:
            metrics["eval_status"] = eval_status
        self.log_metrics(metrics, step=step)

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled or not self.run_id:
            return
        try:
            active_run = self.mlflow.active_run()
            if active_run and active_run.info.run_id == self.run_id:
                self.mlflow.end_run(status=status)
            else:
                self.client.set_terminated(self.run_id, status=status)
        except Exception as exc:
            print(f"Failed to finish MLflow run: {exc}")
        finally:
            self.active_run = None


class MlflowCallback:
    """Callback for integrating MLflow with AIDE experiments."""

    def __init__(self, logger: AIDEMLflowLogger):
        self.logger = logger
        self.step_count = 0

    def on_step_start(self, step: int) -> None:
        self.step_count = step
        self.logger.log_metrics({"aide_step": step}, step=step)

    def on_code_generated(self, code: str) -> None:
        self.logger.log_code(code, step=self.step_count)

    def on_evaluation_complete(self, val_loss: float, exec_time: float | None = None) -> None:
        self.logger.log_evaluation(val_loss=val_loss, training_time=exec_time, step=self.step_count)


def create_mlflow_logger_for_experiment(
    experiment_name: str,
    gpu_id: int,
    config: dict[str, Any],
    tracking_experiment: str | None = None,
    parent_run_id: str | None = None,
) -> AIDEMLflowLogger:
    full_config = {
        "experiment_name": experiment_name,
        "gpu_id": gpu_id,
        "timestamp": datetime.now().isoformat(),
        **config,
    }
    return AIDEMLflowLogger(
        run_name=experiment_name,
        gpu_id=gpu_id,
        experiment_config=full_config,
        tracking_experiment=tracking_experiment,
        parent_run_id=parent_run_id,
    )
