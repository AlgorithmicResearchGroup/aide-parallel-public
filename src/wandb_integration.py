"""
Weights & Biases integration for AIDE parallel experiments.
Tracks metrics across all GPU experiments in a single W&B project.
Includes Weave support for LLM call tracing.
"""

import wandb
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Try to import weave for LLM tracing
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    print("Weave not available. Install with 'pip install weave' for LLM call tracing.")


class AIDEWandbLogger:
    """Centralized W&B logger for AIDE experiments"""

    def __init__(
        self,
        run_name: str = None,
        gpu_id: int = None,
        experiment_config: Dict[str, Any] = None
    ):
        """Initialize W&B run for an experiment"""

        # Get W&B config from environment
        api_key = os.environ.get("WANDB_API_KEY", "")

        # Use task-specific project if provided in config
        task_type = experiment_config.get("task_type", "attention") if experiment_config else "attention"
        default_project = f"aide-{task_type}-h100"
        project = os.environ.get("WANDB_PROJECT", default_project)
        entity = os.environ.get("WANDB_ENTITY", None)

        if not api_key or api_key == "YOUR_WANDB_KEY_HERE":
            print("Warning: W&B API key not configured. Logging disabled.")
            self.enabled = False
            return

        self.enabled = True

        # Create unique run name
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gpu_suffix = f"gpu{gpu_id}" if gpu_id is not None else "cpu"
            run_name = f"aide_{gpu_suffix}_{timestamp}"

        # Initialize W&B
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=experiment_config or {},
                group="parallel_run",  # Group all parallel experiments
                job_type="optimization",
                reinit=True
            )

            # Initialize Weave for LLM tracing if available
            if WEAVE_AVAILABLE:
                try:
                    # Initialize weave with the same project
                    weave.init(f"{entity}/{project}" if entity else project)
                    self.weave_enabled = True
                    print(f"Weave initialized for LLM tracing in project: {project}")
                except Exception as e:
                    print(f"Warning: Failed to initialize Weave: {e}")
                    self.weave_enabled = False
            else:
                self.weave_enabled = False

            # Add GPU info to config
            if gpu_id is not None:
                wandb.config.update({"gpu_id": gpu_id})

            print(f"W&B run initialized: {self.run.url}")

        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            self.enabled = False
            self.weave_enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B"""
        if not self.enabled:
            return

        try:
            # Add timestamp
            metrics["timestamp"] = time.time()

            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        except Exception as e:
            print(f"Failed to log metrics to W&B: {e}")

    def log_code(self, code: str, name: str = "generated_code", step: int = 0):
        """Log generated code as W&B artifact"""
        if not self.enabled:
            return

        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=f"{name}_step{step}",
                type="code",
                description=f"AIDE generated code at step {step}"
            )

            # Save code to temporary file
            temp_path = Path(f"/tmp/{name}_step{step}.py")
            temp_path.write_text(code)

            # Add to artifact
            artifact.add_file(str(temp_path))

            # Log artifact
            self.run.log_artifact(artifact)

            # Clean up
            temp_path.unlink()

        except Exception as e:
            print(f"Failed to log code artifact: {e}")

    def log_tree_node(self, node_info: Dict[str, Any], step: int):
        """Log information about a tree search node"""
        if not self.enabled:
            return

        metrics = {
            "tree/node_id": node_info.get("id", ""),
            "tree/is_buggy": node_info.get("is_buggy", False),
            "tree/has_valid_metric": node_info.get("has_metric", False),
            "tree/depth": node_info.get("depth", 0),
            "tree/step": step
        }

        # Add metric value if available
        if "metric_value" in node_info:
            metrics["tree/metric_value"] = node_info["metric_value"]

        self.log_metrics(metrics, step=step)

    def log_evaluation(self, val_loss: float, training_time: float = None, eval_status: str = None, step: int = None):
        """Log evaluation metrics"""
        if not self.enabled:
            return

        metrics = {
            "val_loss": val_loss,
            "best_val_loss": val_loss,  # W&B will track the minimum automatically
        }

        if training_time is not None:
            metrics["training_time"] = training_time

        if eval_status is not None:
            metrics["eval_status"] = eval_status

        self.log_metrics(metrics, step=step)

    def log_gpu_metrics(self, gpu_utilization: float, memory_used_mb: float):
        """Log GPU utilization metrics"""
        if not self.enabled:
            return

        metrics = {
            "gpu/utilization_percent": gpu_utilization,
            "gpu/memory_used_mb": memory_used_mb,
        }

        self.log_metrics(metrics)

    def log_experiment_summary(self, summary: Dict[str, Any]):
        """Log final experiment summary"""
        if not self.enabled:
            return

        try:
            # Log summary metrics
            wandb.run.summary.update(summary)

            # Log final code as artifact if provided
            if "best_code" in summary:
                self.log_code(summary["best_code"], name="best_code")

        except Exception as e:
            print(f"Failed to log summary: {e}")

    def finish(self):
        """Close W&B run"""
        if not self.enabled:
            return

        try:
            # Close Weave if it was initialized
            if hasattr(self, 'weave_enabled') and self.weave_enabled and WEAVE_AVAILABLE:
                try:
                    weave.finish()
                    print("Weave tracing finished")
                except Exception as e:
                    print(f"Warning: Failed to close Weave: {e}")

            wandb.finish()
        except Exception as e:
            print(f"Failed to finish W&B run: {e}")


class KernelWandbLogger(AIDEWandbLogger):
    """W&B logger specifically for kernel optimization experiments"""

    def log_evaluation(self, speedup: float, execution_time: float = None, eval_status: str = None, step: int = None):
        """Log kernel evaluation metrics"""
        if not self.enabled:
            return

        metrics = {
            "speedup": speedup,
            "best_speedup": speedup,  # W&B will track the maximum automatically
        }

        if execution_time is not None:
            metrics["kernel_execution_time"] = execution_time

        if eval_status is not None:
            metrics["eval_status"] = eval_status

        if step is not None:
            metrics["step"] = step

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"Failed to log metrics to W&B: {e}")

    def log_experiment_summary(self, summary: Dict[str, Any]):
        """Log final experiment summary with kernel-specific metrics"""
        if not self.enabled:
            return

        # Extract kernel-specific metrics
        summary_metrics = {
            "final_speedup": summary.get("best_speedup"),
            "total_steps": summary.get("total_steps"),
            "gpu_id": summary.get("gpu_id"),
            "experiment_idx": summary.get("experiment_idx")
        }

        # Filter out None values
        summary_metrics = {k: v for k, v in summary_metrics.items() if v is not None}

        try:
            wandb.log(summary_metrics)
            wandb.summary.update(summary_metrics)

            # Log best code if available
            if "best_code" in summary and summary["best_code"]:
                artifact = wandb.Artifact(
                    name=f"optimized_kernel_{self.run.id}",
                    type="code",
                    description="Best kernel optimization code"
                )

                with artifact.new_file("optimize.py") as f:
                    f.write(summary["best_code"])

                self.run.log_artifact(artifact)

        except Exception as e:
            print(f"Failed to log summary to W&B: {e}")


class WandbCallback:
    """Callback for integrating W&B with AIDE experiments"""

    def __init__(self, logger: AIDEWandbLogger):
        self.logger = logger
        self.step_count = 0

    def on_step_start(self, step: int):
        """Called at the start of each AIDE step"""
        self.step_count = step
        self.logger.log_metrics({"aide_step": step}, step=step)

    def on_code_generated(self, code: str):
        """Called when new code is generated"""
        self.logger.log_code(code, step=self.step_count)

    def on_evaluation_complete(self, val_loss: float, exec_time: float = None):
        """Called after evaluation completes"""
        self.logger.log_evaluation(val_loss, exec_time, step=self.step_count)

    def on_node_created(self, node_info: Dict[str, Any]):
        """Called when a new tree node is created"""
        self.logger.log_tree_node(node_info, step=self.step_count)


def create_wandb_logger_for_experiment(
    experiment_name: str,
    gpu_id: int,
    config: Dict[str, Any]
) -> AIDEWandbLogger:
    """Factory function to create W&B logger for an experiment"""

    # Add experiment metadata to config
    full_config = {
        "experiment_name": experiment_name,
        "gpu_id": gpu_id,
        "timestamp": datetime.now().isoformat(),
        **config
    }

    # Choose logger based on task type
    task_type = config.get("task_type", "attention")

    if task_type in ["kernel", "kernelbench"]:
        return KernelWandbLogger(
            run_name=experiment_name,
            gpu_id=gpu_id,
            experiment_config=full_config
        )
    else:
        return AIDEWandbLogger(
            run_name=experiment_name,
            gpu_id=gpu_id,
            experiment_config=full_config
        )