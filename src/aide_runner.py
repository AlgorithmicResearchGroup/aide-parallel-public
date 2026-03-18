"""
Ray-based parallel experiment runner with GPU support.
Distributes experiments across multiple GPUs on the cluster.
"""

from pathlib import Path
from typing import Any, Optional
import json
import math
import os
import re
import sys
import tempfile
import traceback

# Note: Path manipulation is now handled by Ray runtime_env
# The runtime_env will ensure all required paths are available on worker nodes

from dotenv import load_dotenv
import ray
import yaml
import logging
import importlib

# Import AIDE conditionally - it will be imported in the actor
try:
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "aideml")))
    import aide
    AIDE_AVAILABLE = True
except ImportError:
    AIDE_AVAILABLE = False
    print("AIDE will be imported inside Ray actors")

if os.getenv("AIDE_ENABLE_MLFLOW", "0") == "1":
    try:
        from mlflow_integration import create_mlflow_logger_for_experiment
        MLFLOW_AVAILABLE = True
    except ImportError:
        MLFLOW_AVAILABLE = False
else:
    MLFLOW_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALGOTUNE_PATH = PROJECT_ROOT / "tasks" / "algotune" / "vendor" / "AlgoTune"
STRICT_ALGOTUNE_MODE = "benchmark_strict"
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
RUNTIME_ENV_PREFIXES = (
    "AIDE_",
    "MLFLOW_",
    "OPENAI_",
    "ANTHROPIC_",
    "GROQ_",
    "KERNELBENCH_",
    "ALGOTUNE_",
)
TRACE_CONTEXT_ENV_MAP = {
    "task_type": "AIDE_TRACE_TASK_TYPE",
    "task_id": "AIDE_TRACE_TASK_ID",
    "experiment_idx": "AIDE_TRACE_EXPERIMENT_IDX",
    "iteration": "AIDE_TRACE_ITERATION",
}


def _progress(message: str) -> None:
    print(f"[aide.run] {message}", flush=True)


def _validate_model_credentials(model: str, feedback_model: str) -> None:
    backend = _import_local_aide().backend

    def required_env(provider: str) -> str | None:
        if provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        if provider == "gemini":
            return "GEMINI_API_KEY"
        if provider == "openrouter":
            return "OPENROUTER_API_KEY"
        if provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "")
            if "groq.com" in base_url or not base_url:
                return "GROQ_API_KEY"
            return "OPENAI_API_KEY"
        return None

    checked: set[tuple[str, str | None]] = set()
    for candidate in (model, feedback_model):
        provider = backend.determine_provider(candidate)
        env_name = required_env(provider)
        marker = (provider, env_name)
        if marker in checked:
            continue
        checked.add(marker)
        if env_name and not os.getenv(env_name):
            detail = f"Selected model '{candidate}' uses provider '{provider}'"
            if provider == "openai" and (os.getenv("OPENAI_BASE_URL", "") == "" or "groq.com" in os.getenv("OPENAI_BASE_URL", "")):
                detail += " via the current OpenAI-compatible endpoint configuration"
            raise RuntimeError(
                f"{detail}, but required credential {env_name} is not set. "
                f"Set {env_name} or pass --model/--feedback-model for a provider with configured credentials."
            )


def _import_local_aide():
    aide_root = str(PROJECT_ROOT / "aideml")
    if sys.path[0] != aide_root:
        sys.path.insert(0, aide_root)
    module = sys.modules.get("aide")
    if module is not None:
        module_path = getattr(module, "__file__", "") or ""
        if module_path.startswith(str(PROJECT_ROOT / "aideml")):
            return module
        del sys.modules["aide"]
    return importlib.import_module("aide")


def _extract_eval_metadata(term_out: str | None) -> dict[str, Any]:
    """Parse structured evaluation hints from speedup-task terminal output."""
    metadata: dict[str, Any] = {
        "compiled": None,
        "correct": None,
        "error": None,
    }
    if not term_out:
        return metadata

    if "Compiled: ✓" in term_out or "Compilation: ✓" in term_out:
        metadata["compiled"] = True
    elif "Compiled: ✗" in term_out or "Compilation: ✗" in term_out:
        metadata["compiled"] = False

    if "Correct: ✓" in term_out or "Correctness: ✓" in term_out:
        metadata["correct"] = True
    elif "Correct: ✗" in term_out or "Correctness: ✗" in term_out:
        metadata["correct"] = False

    eval_error = re.findall(r"\[Eval error\]:\s*(.+)", term_out)
    if eval_error:
        metadata["error"] = eval_error[-1].strip()
    return metadata


def _is_strict_algotune_mode(mode: str | None) -> bool:
    return (mode or STRICT_ALGOTUNE_MODE) == STRICT_ALGOTUNE_MODE


def _is_non_reportable_run(task_type: str, at_mode: str | None) -> bool:
    return False


def _select_reportable_result(task_type: str, at_mode: str | None, best_result: dict[str, Any], final_result: dict[str, Any] | None = None) -> dict[str, Any]:
    if task_type == "algotune" and _is_strict_algotune_mode(at_mode):
        if isinstance(final_result, dict) and final_result:
            return final_result
    return best_result


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


MAX_RESULT_CODE_CHARS = int(os.getenv("AIDE_MAX_RESULT_CODE_CHARS", "50000"))
MAX_RESULT_TEXT_CHARS = int(os.getenv("AIDE_MAX_RESULT_TEXT_CHARS", "20000"))
MAX_ERROR_TEXT_CHARS = int(os.getenv("AIDE_MAX_ERROR_TEXT_CHARS", "12000"))
MAX_RETURN_CODE_CHARS = int(os.getenv("AIDE_MAX_RETURN_CODE_CHARS", "2000"))
MAX_RETURN_TEXT_CHARS = int(os.getenv("AIDE_MAX_RETURN_TEXT_CHARS", "1200"))
MAX_RETURN_ERROR_CHARS = int(os.getenv("AIDE_MAX_RETURN_ERROR_CHARS", "3000"))
MAX_ERROR_STACK_LINES = int(os.getenv("AIDE_MAX_ERROR_STACK_LINES", "12"))
MAX_RAY_RESULT_BYTES = int(os.getenv("AIDE_MAX_RAY_RESULT_BYTES", "65536"))


def _truncate_text(value: str | None, limit: int) -> str | None:
    if value is None or len(value) <= limit:
        return value
    clipped = len(value) - limit
    return f"{value[:limit]}\n... [{clipped} characters truncated]"


def _compact_exception(exc: BaseException) -> str:
    detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    if exc.__traceback__:
        extracted = traceback.extract_tb(exc.__traceback__)
        extracted = extracted[-MAX_ERROR_STACK_LINES:]
        tb = "".join(traceback.format_list(extracted))
    else:
        tb = ""
    text = f"{detail}\n{tb}".strip() if tb else detail
    return _truncate_text(text, MAX_RETURN_ERROR_CHARS) or detail


def _load_best_code(result: dict[str, Any]) -> str | None:
    best_solution_path = result.get("best_solution_path")
    if best_solution_path:
        try:
            return Path(best_solution_path).read_text(encoding="utf-8")
        except Exception:
            pass
    return result.get("code")


def _coerce_ray_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            scalar = value.item()
            if isinstance(scalar, (bool, int, str)):
                return scalar
            if isinstance(scalar, float):
                return scalar if math.isfinite(scalar) else None
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _coerce_ray_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_ray_value(item) for item in value]
    return _truncate_text(repr(value), MAX_RETURN_TEXT_CHARS)


def _ray_result_fallback(result: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    fallback = {
        "valid_metric": result.get("valid_metric"),
        "code": None,
        "compiled": result.get("compiled"),
        "correct": result.get("correct"),
        "error": _truncate_text(error or result.get("error"), MAX_RETURN_ERROR_CHARS),
        "term_out": None,
        "gpu_id": result.get("gpu_id"),
        "gpu_name": result.get("gpu_name"),
        "experiment_idx": result.get("experiment_idx"),
        "log_dir": result.get("log_dir"),
        "workspace_dir": result.get("workspace_dir"),
        "best_solution_path": result.get("best_solution_path"),
        "config_path": result.get("config_path"),
    }
    return fallback


def _ensure_ray_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    sanitized = _coerce_ray_value(result)
    if not isinstance(sanitized, dict):
        return _ray_result_fallback({}, error="Unexpected non-dict Ray result payload")

    try:
        import msgpack

        packed = msgpack.packb(sanitized, use_bin_type=True)
        if len(packed) <= MAX_RAY_RESULT_BYTES:
            return sanitized

        compact = dict(sanitized)
        compact["code"] = None
        compact["term_out"] = None
        packed = msgpack.packb(compact, use_bin_type=True)
        if len(packed) <= MAX_RAY_RESULT_BYTES:
            compact["error"] = _truncate_text(
                compact.get("error") or "Ray payload compacted to fit serialization budget",
                MAX_RETURN_ERROR_CHARS,
            )
            return compact

        return _ray_result_fallback(
            compact,
            error=(
                "Ray result payload exceeded serialization budget and was reduced "
                "to metadata only"
            ),
        )
    except Exception as exc:
        return _ray_result_fallback(
            sanitized,
            error=f"Ray result payload sanitization fallback: {_compact_exception(exc)}",
        )


def _release_experiment_memory(exp: Any) -> None:
    if exp is None:
        return
    try:
        journal = getattr(exp, "journal", None)
        nodes = getattr(journal, "nodes", None)
        if isinstance(nodes, list):
            nodes.clear()
    except Exception:
        pass
    for attr in ("agent", "interpreter", "journal"):
        try:
            setattr(exp, attr, None)
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def _ray_failure_result(
    *,
    experiment_idx: int,
    error: str,
) -> dict[str, Any]:
    return _ensure_ray_safe_result({
        "valid_metric": None,
        "code": None,
        "compiled": None,
        "correct": None,
        "error": _truncate_text(error, MAX_RETURN_ERROR_CHARS),
        "term_out": None,
        "gpu_id": None,
        "gpu_name": None,
        "experiment_idx": experiment_idx,
        "log_dir": None,
        "workspace_dir": None,
        "best_solution_path": None,
        "config_path": None,
    })


# Note: num_gpus will be set dynamically in run_experiments()
class Experiment:
    def __init__(
        self,
        data_dir: str,
        goal: str,
        model: str,
        feedback_model: str,
        eval_metric: str | None,
        tracking_enabled: bool = False,
        task_type: str = "attention",
        tracking_experiment: str = None,
        parent_run_id: str | None = None,
        task_id: str = None,
        algotune_n_problems: int | None = None,
        algotune_n_runs: int | None = None,
        algotune_mode: str = STRICT_ALGOTUNE_MODE,
    ):
        # Load environment variables from .env file
        import os
        import sys
        from pathlib import Path

        # Environment variables and paths are now handled by Ray runtime_env
        # No need to manually load .env or manipulate sys.path

        self.data_dir = data_dir
        self.goal = goal
        self.model = model
        self.feedback_model = feedback_model
        self.eval_metric = eval_metric
        self.tracking_enabled = tracking_enabled
        self.task_type = task_type
        self.task_id = task_id
        self.algotune_n_problems = None
        self.algotune_n_runs = None
        self.algotune_mode = STRICT_ALGOTUNE_MODE if self.task_type == "algotune" else algotune_mode
        self.tracking_experiment = tracking_experiment
        self.parent_run_id = parent_run_id
        self.mlflow_logger = None  # Persistent MLflow logger across iterations
        self.total_steps_completed = 0  # Track total steps across all iterations

        if self.task_type == "algotune":
            # Keep parent actor processes single-threaded for local sweep stability.
            for env_name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                os.environ.setdefault(env_name, "1")

        if tracking_experiment:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = tracking_experiment
            logger.info(f"Set MLflow experiment to: {tracking_experiment}")

        # Get GPU info for this actor
        import torch
        if torch.cuda.is_available():
            self.gpu_id = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.gpu_id)
            logger.info(f"Experiment initialized on GPU {self.gpu_id}: {self.gpu_name}")
        else:
            self.gpu_id = None
            self.gpu_name = "CPU"
            logger.warning("No GPU available, running on CPU")

    def _set_trace_context(self, experiment_idx: int, iteration: int) -> None:
        trace_context = {
            "task_type": self.task_type,
            "task_id": self.task_id,
            "experiment_idx": str(experiment_idx),
            "iteration": str(iteration),
        }
        for key, env_name in TRACE_CONTEXT_ENV_MAP.items():
            value = trace_context.get(key)
            if value:
                os.environ[env_name] = value
            else:
                os.environ.pop(env_name, None)

    def _failure_result(
        self,
        *,
        experiment_idx: int,
        error: str,
        exp: Any = None,
    ) -> dict[str, Any]:
        return {
            "valid_metric": None,
            "code": None,
            "compiled": None,
            "correct": None,
            "error": _truncate_text(error, MAX_RETURN_ERROR_CHARS),
            "term_out": None,
            "gpu_id": self.gpu_id,
            "gpu_name": self.gpu_name,
            "experiment_idx": experiment_idx,
            "log_dir": str(exp.cfg.log_dir) if exp is not None else None,
            "workspace_dir": str(exp.cfg.workspace_dir) if exp is not None else None,
            "best_solution_path": str(exp.cfg.log_dir / "best_solution.py") if exp is not None else None,
            "config_path": str(exp.cfg.log_dir / "config.yaml") if exp is not None else None,
        }

    def _log_tracking_outcome(
        self,
        tracking_logger: Any,
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
        if not tracking_logger:
            return
        tracking_logger.log_task_outcome(
            task_type=task_type,
            status=status,
            metric_value=metric_value,
            compiled=compiled,
            correct=correct,
            error=error,
            step=step,
            phase=phase,
        )

    def run(self, steps: int = 2, experiment_idx: int = 0, iteration: int = 1, previous_step_count: int = 0) -> dict[str, Any]:
        """Run the AIDE experiment with GPU device properly configured and MLflow logging.

        Args:
            steps: Number of AIDE steps to run
            experiment_idx: Index of this experiment
            iteration: Current iteration number (1-based)
            previous_step_count: Cumulative steps from previous iterations
        """

        # Memory monitoring for fractional GPU usage
        import torch
        import gc

        # Clear any cached memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if self.tracking_enabled:
            try:
                from mlflow_integration import create_mlflow_logger_for_experiment
                tracking_available = True
            except ImportError:
                tracking_available = False
                logger.warning("MLflow integration not available on this node")
        else:
            tracking_available = False

        # Set CUDA_VISIBLE_DEVICES to ensure subprocess uses correct GPU
        if self.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            logger.info(f"Set CUDA_VISIBLE_DEVICES={self.gpu_id} for experiment")
        self._set_trace_context(experiment_idx=experiment_idx, iteration=iteration)
        _progress(
            f"experiment={experiment_idx} iteration={iteration} steps={steps} "
            f"task={self.task_type} model={self.model}"
        )

        if tracking_available and self.mlflow_logger is None:
            self.mlflow_logger = create_mlflow_logger_for_experiment(
                experiment_name=f"{self.task_type}_exp{experiment_idx}_gpu{self.gpu_id if self.gpu_id is not None else 'cpu'}",
                gpu_id=self.gpu_id if self.gpu_id is not None else -1,
                config={
                    "task_type": self.task_type,
                    "task_id": self.task_id,
                    "non_reportable": _is_non_reportable_run(self.task_type, self.algotune_mode),
                    "model": self.model,
                    "feedback_model": self.feedback_model,
                    "eval_metric": self.eval_metric,
                    "gpu_name": self.gpu_name,
                    "data_dir": str(self.data_dir),
                    "total_iterations": iteration,  # Will be updated in each iteration
                    "steps_per_iteration": steps,
                    "experiment_idx": experiment_idx,
                    "at_mode": self.algotune_mode,
                },
                tracking_experiment=self.tracking_experiment,
                parent_run_id=self.parent_run_id,
            )

        tracking_logger = self.mlflow_logger

        if tracking_logger:
            tracking_logger.log_metrics({
                "iteration": iteration,
                "iteration_started": 1
            }, step=self.total_steps_completed)

        # Import AIDE here where it's needed
        # Add aideml to path since runtime_env distributes files
        aide = _import_local_aide()

        exp = None
        try:
            exp = aide.Experiment(
                data_dir=self.data_dir,
                goal=self.goal,
                eval=self.eval_metric,
                task_type=self.task_type,
                task_id=self.task_id,
                algotune_n_problems=self.algotune_n_problems,
                algotune_n_runs=self.algotune_n_runs,
                algotune_mode=self.algotune_mode,
            )

            # Configure the experiment
            exp.cfg.agent.code.model = self.model
            exp.cfg.agent.feedback.model = self.feedback_model
            exp.cfg.report.model = self.model
            logger.info(f"Starting experiment {experiment_idx} on GPU {self.gpu_id} - Iteration {iteration}")
            _progress(f"created experiment workspace={exp.cfg.workspace_dir} log_dir={exp.cfg.log_dir}")

            for step in range(steps):
                global_step = self.total_steps_completed + step
                _progress(
                    f"experiment={experiment_idx} iteration={iteration} step={step + 1}/{steps} starting"
                )
                if tracking_logger:
                    tracking_logger.log_metrics({
                        "aide_step": step + 1,
                        "global_step": global_step + 1,
                        "iteration": iteration
                    }, step=global_step)

                try:
                    exp.agent.step(exec_callback=exp.interpreter.run)
                    _progress(
                        f"experiment={experiment_idx} iteration={iteration} step={step + 1}/{steps} finished"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                        logger.error(f"GPU OOM on experiment {experiment_idx}, step {step}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        continue
                    raise
                except Exception as exc:
                    compact_error = _compact_exception(exc)
                    logger.error(
                        "Experiment %s failed during step %s: %s",
                        experiment_idx,
                        step,
                        compact_error,
                    )
                    _progress(
                        f"experiment={experiment_idx} iteration={iteration} step={step + 1} failed: {compact_error}"
                    )
                    if tracking_logger:
                        tracking_logger.log_metrics(
                            {
                                "iteration": iteration,
                                "aide_step": step + 1,
                            },
                            step=global_step,
                        )
                        tracking_logger.log_experiment_summary(
                            {
                                "final_status": "failed",
                                "error": compact_error,
                            }
                        )
                        self._log_tracking_outcome(
                            tracking_logger,
                            task_type=self.task_type,
                            status="failed",
                            metric_value=None,
                            error=compact_error,
                            step=global_step,
                            phase="search" if self.task_type == "algotune" and _is_strict_algotune_mode(self.algotune_mode) else None,
                        )
                    best_node = exp.journal.get_best_node(only_good=False)
                    failure_result = self._failure_result(
                        experiment_idx=experiment_idx,
                        error=compact_error,
                        exp=exp,
                    )
                    _release_experiment_memory(exp)
                    exp = None
                    return _ensure_ray_safe_result(failure_result)

                best_node = exp.journal.get_best_node(only_good=False)
                if tracking_logger and best_node:
                    eval_metadata = _extract_eval_metadata(best_node.term_out if best_node else None)
                    if best_node.metric and best_node.metric.value is not None:
                        metric_value = float(best_node.metric.value)
                        eval_status = "success"
                    else:
                        metric_value = 0.0
                        eval_status = "failed"
                        logger.info(f"Step {step}: Evaluation failed, logging with speedup=0.0")

                    if self.task_type in ["kernel", "kernelbench", "algotune"] and not (
                        self.task_type == "algotune" and _is_strict_algotune_mode(self.algotune_mode)
                    ):
                        tracking_logger.log_evaluation(
                            speedup=metric_value,
                            execution_time=best_node.exec_time if best_node.exec_time else 0.0,
                            eval_status=eval_status,
                            step=global_step
                        )
                    elif self.task_type == "algotune" and _is_strict_algotune_mode(self.algotune_mode):
                        tracking_logger.log_metrics(
                            {
                                "search_step_mean_speedup": metric_value,
                                "search_step_execution_time": best_node.exec_time if best_node.exec_time else 0.0,
                                "search_eval_status": eval_status,
                            },
                            step=global_step,
                        )
                    else:
                        if eval_status == "failed":
                            metric_value = float('inf')
                        tracking_logger.log_evaluation(
                            val_loss=metric_value,
                            training_time=best_node.exec_time if best_node.exec_time else 0.0,
                            eval_status=eval_status,
                            step=global_step
                        )

                    step_metrics = {
                        "step_has_evaluation_error": 1.0 if eval_metadata["error"] else 0.0,
                    }
                    if eval_metadata["compiled"] is not None:
                        step_metrics["step_compiled"] = 1.0 if eval_metadata["compiled"] else 0.0
                    if eval_metadata["correct"] is not None:
                        step_metrics["step_correct"] = 1.0 if eval_metadata["correct"] else 0.0
                    tracking_logger.log_metrics(step_metrics, step=global_step)
                    tracking_logger.set_tags(
                        {
                            "latest.compiled_status": eval_metadata["compiled"],
                            "latest.correct_status": eval_metadata["correct"],
                            "latest.evaluation_error": eval_metadata["error"],
                        }
                    )

                    tracking_logger.log_code(best_node.code, name=f"code_iter{iteration}_step{step}", step=global_step)
                if best_node:
                    _progress(
                        f"experiment={experiment_idx} best metric after step={step + 1}: "
                        f"{best_node.metric.value if best_node.metric else None}"
                    )

                if torch.cuda.is_available() and (step + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.debug(f"Cleared GPU cache at step {step + 1}")

            best_solution = exp.journal.get_best_node(only_good=False)
            self.total_steps_completed += steps

            if tracking_logger:
                if best_solution and best_solution.metric and best_solution.metric.value is not None:
                    metric_value = float(best_solution.metric.value)
                    eval_status = "success"
                else:
                    if self.task_type in ["kernel", "kernelbench", "algotune"]:
                        metric_value = 0.0
                    else:
                        metric_value = float('inf')
                    eval_status = "failed"

                iteration_summary = {
                    f"iteration_{iteration}_best": metric_value,
                    f"iteration_{iteration}_steps": steps,
                    f"iteration_{iteration}_status": eval_status,
                    "cumulative_steps": self.total_steps_completed,
                    "current_iteration": iteration,
                }

                if self.task_type in ["kernel", "kernelbench", "algotune"]:
                    iteration_summary["best_speedup_so_far"] = metric_value
                else:
                    iteration_summary["best_val_loss_so_far"] = metric_value
                if self.task_type == "algotune" and _is_strict_algotune_mode(self.algotune_mode):
                    iteration_summary["search_best_speedup_so_far"] = metric_value

                tracking_logger.log_metrics(iteration_summary, step=self.total_steps_completed - 1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            from datetime import datetime
            completion_time = datetime.now().strftime("%H:%M:%S")
            metric_str = f"{best_solution.metric.value:.4f}" if (best_solution and best_solution.metric and best_solution.metric.value is not None) else "None"
            print(f"\n{'='*60}")
            print(f"✅ EXPERIMENT {experiment_idx} COMPLETED at {completion_time}")
            print(f"   Task: {self.task_type}")
            print(f"   GPU: {self.gpu_name} (ID: {self.gpu_id})")
            print(f"   Final Metric: {metric_str}")
            print(f"   Total Steps: {steps}")
            print(f"{'='*60}\n")

            term_out = best_solution.term_out if best_solution else None
            eval_metadata = _extract_eval_metadata(term_out)
            result_payload = {
                "valid_metric": best_solution.metric.value if (best_solution and best_solution.metric and best_solution.metric.value is not None) else None,
                "code": _truncate_text(best_solution.code if best_solution else None, MAX_RETURN_CODE_CHARS),
                "compiled": eval_metadata["compiled"],
                "correct": eval_metadata["correct"],
                "error": _truncate_text(eval_metadata["error"], MAX_RETURN_ERROR_CHARS),
                "term_out": _truncate_text(term_out, MAX_RETURN_TEXT_CHARS),
                "gpu_id": self.gpu_id,
                "gpu_name": self.gpu_name,
                "experiment_idx": experiment_idx,
                "log_dir": str(exp.cfg.log_dir),
                "workspace_dir": str(exp.cfg.workspace_dir),
                "best_solution_path": str(exp.cfg.log_dir / "best_solution.py"),
                "config_path": str(exp.cfg.log_dir / "config.yaml"),
            }
            result_status = _result_status(self.task_type, result_payload)
            _progress(f"experiment={experiment_idx} completed status={result_status}")
            if tracking_logger:
                self._log_tracking_outcome(
                    tracking_logger,
                    task_type=self.task_type,
                    status=result_status,
                    metric_value=result_payload["valid_metric"],
                    compiled=result_payload["compiled"],
                    correct=result_payload["correct"],
                    error=result_payload["error"],
                    step=self.total_steps_completed - 1,
                    phase="search" if self.task_type == "algotune" and _is_strict_algotune_mode(self.algotune_mode) else None,
                )
            ray_result = _ensure_ray_safe_result(result_payload)
            del best_solution
            _release_experiment_memory(exp)
            exp = None
            return ray_result
        except Exception as exc:
            compact_error = _compact_exception(exc)
            logger.error("Experiment %s failed before completion: %s", experiment_idx, compact_error)
            _progress(f"experiment={experiment_idx} failed before completion: {compact_error}")
            if tracking_logger:
                tracking_logger.log_experiment_summary(
                    {
                        "final_status": "failed",
                        "error": compact_error,
                    }
                )
                self._log_tracking_outcome(
                    tracking_logger,
                    task_type=self.task_type,
                    status="failed",
                    metric_value=None,
                    error=compact_error,
                    phase="search" if self.task_type == "algotune" and _is_strict_algotune_mode(self.algotune_mode) else None,
                )
            failure_result = _ensure_ray_safe_result(self._failure_result(
                experiment_idx=experiment_idx,
                error=compact_error,
                exp=exp,
            ))
            _release_experiment_memory(exp)
            exp = None
            return failure_result

    def finish_all_iterations(self):
        """Finish the MLflow run after all iterations are complete."""
        if self.mlflow_logger:
            self.mlflow_logger.log_experiment_summary({
                "total_steps_completed": self.total_steps_completed,
                "final_status": "completed"
            })
            self.mlflow_logger.finish()
            self.mlflow_logger = None


def _metric_score(result: dict[str, Any]) -> float:
    """Score the result based on the metric."""
    metric = result.get("valid_metric")
    if isinstance(metric, (int, float)):
        return float(metric)
    if metric == "failed":
        return float("-inf")
    try:
        return float(metric)
    except (ValueError, TypeError):
        return float("-inf")


def _result_status(task_type: str, result: dict[str, Any]) -> str:
    metric = result.get("valid_metric")
    if metric is None:
        return "failed"
    if task_type == "algotune" and (
        result.get("correct") is not True or result.get("error")
    ):
        return "failed"
    return "succeeded"


def _validate_algotune_strict_environment() -> None:
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from tasks.algotune.strict_benchmark import assert_benchmark_environment

    assert_benchmark_environment(check_all_tasks=True)


def _run_algotune_strict_final_evaluation(*, task_id: str, best_result: dict[str, Any]) -> dict[str, Any]:
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from tasks.algotune.strict_benchmark import evaluate_solver_split

    best_solution_path = best_result.get("best_solution_path")
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    solver_path: str
    if best_solution_path and Path(best_solution_path).exists():
        solver_path = str(best_solution_path)
    else:
        code = _load_best_code(best_result)
        if not code:
            return {
                "mode": "benchmark_strict",
                "split": "test",
                "status": "failed",
                "valid_metric": None,
                "compiled": None,
                "correct": None,
                "error": "No best solution artifact available for strict final evaluation",
            }
        temp_dir = tempfile.TemporaryDirectory(prefix="aide-algotune-final-")
        solver_file = Path(temp_dir.name) / "solver.py"
        solver_file.write_text(code, encoding="utf-8")
        solver_path = str(solver_file)

    try:
        return evaluate_solver_split(
            task_name=task_id,
            solver_path=solver_path,
            split="test",
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def get_cluster_resources() -> dict:
    """Get information about available resources in the Ray cluster."""
    resources = ray.available_resources()
    return {
        "total_cpus": resources.get("CPU", 0),
        "total_gpus": resources.get("GPU", 0),
        "available_cpus": ray.available_resources().get("CPU", 0),
        "available_gpus": ray.available_resources().get("GPU", 0),
    }


def _use_direct_single_experiment_mode(
    *,
    task_type: str,
    num_experiments: int,
    head_node_ip: Optional[str],
) -> bool:
    if os.getenv("AIDE_ALGOTUNE_DIRECT_SINGLE_EXPERIMENT", "1") in {"0", "false", "False"}:
        return False
    return task_type == "algotune" and num_experiments == 1 and head_node_ip is None


def run_experiments(
    data_dir: str,
    goal: str,
    model: str,
    feedback_model: str,
    eval_metric: str | None,
    num_experiments: int,
    steps_per_experiment: int = 2,
    task_type: str = "attention",
    gpu_fraction: float = 1.0,
    cpus_per_experiment: int | None = None,
    tracking_experiment: str = None,
    parent_run_id: str | None = None,
    iteration: int = 1,
    experiment_actors: list = None,
    task_id: str = None,
    algotune_n_problems: int | None = None,
    algotune_n_runs: int | None = None,
    algotune_mode: str = STRICT_ALGOTUNE_MODE,
    use_direct_execution: bool = False,
) -> list[dict[str, Any]]:
    """Launch Ray actors with GPU or CPU allocation, wait for all results, and return ranked outputs."""

    if use_direct_execution:
        if experiment_actors is None:
            actors = [
                Experiment(
                    data_dir,
                    goal,
                    model,
                    feedback_model,
                    eval_metric,
                    tracking_enabled=MLFLOW_AVAILABLE,
                    task_type=task_type,
                    tracking_experiment=tracking_experiment,
                    parent_run_id=parent_run_id,
                    task_id=task_id,
                    algotune_n_problems=algotune_n_problems,
                    algotune_n_runs=algotune_n_runs,
                    algotune_mode=algotune_mode,
                )
            ]
        else:
            actors = experiment_actors

        print("\n🚀 EXPERIMENTS STARTED: 1 agent running directly")
        print("   Running in-process (no Ray actor)")
        print(f"   Task: {task_type} | Steps per experiment: {steps_per_experiment}")
        print("   Waiting for experiment to complete...\n")

        result = actors[0].run(
            steps=steps_per_experiment,
            experiment_idx=0,
            iteration=iteration,
        )
        completion_time = __import__("datetime").datetime.now().strftime("%H:%M:%S")
        metric_str = f"{result.get('valid_metric'):.4f}" if result.get("valid_metric") is not None else "None"
        print(f"[{completion_time}] ✅ [1/1] Experiment 0 done | GPU {result.get('gpu_id', 'N/A')} | Metric: {metric_str}")
        print("\n🎉 ALL 1 EXPERIMENTS COMPLETED!")
        return [result], actors

    # Check available resources
    resources = get_cluster_resources()
    prefer_cpu = task_type == "algotune" or cpus_per_experiment is not None
    has_gpus = resources["total_gpus"] > 0 and not prefer_cpu
    if has_gpus:
        logger.info(
            "Cluster resources - Total GPUs: %s, Available GPUs: %s",
            resources["total_gpus"],
            resources["available_gpus"],
        )
        if resources["available_gpus"] < num_experiments * gpu_fraction:
            logger.warning(
                "Requested %s experiments with gpu_fraction=%s but only %s GPUs are currently available",
                num_experiments,
                gpu_fraction,
                resources["available_gpus"],
            )
            logger.info("Experiments will queue and run as GPUs become available")
    else:
        logger.info(
            "Cluster resources - Total CPUs: %s, Available CPUs: %s",
            resources["total_cpus"],
            resources["available_cpus"],
        )
        if resources["total_gpus"] > 0 and prefer_cpu:
            logger.info("Scheduling experiments on CPU workers by task/resource preference.")
        else:
            logger.info("No GPUs detected; scheduling experiments on CPU workers.")

    actor_gpu_fraction = gpu_fraction if has_gpus else 0.0

    # Create or reuse actors with GPU requirements
    if experiment_actors is None:
        if has_gpus:
            ExperimentRemote = ray.remote(num_gpus=actor_gpu_fraction)(Experiment)
            logger.info(f"Creating {num_experiments} actors with {actor_gpu_fraction} GPU fraction each")
            logger.info(f"Total GPU requirement: {num_experiments * actor_gpu_fraction:.1f} GPUs")
            logger.info(f"Experiments per GPU: {1/actor_gpu_fraction:.1f}")
        else:
            actor_gpu_fraction = 0.0
            if cpus_per_experiment is None:
                total_cpus = max(1, int(resources["total_cpus"] or 1))
                cpus_per_experiment = max(1, total_cpus // max(1, num_experiments))
            ExperimentRemote = ray.remote(num_gpus=0, num_cpus=cpus_per_experiment)(Experiment)
            logger.info(f"Creating {num_experiments} actors with {cpus_per_experiment} CPUs each")
            logger.info(f"Total CPU requirement: {num_experiments * cpus_per_experiment}")

        actors = []
        for i in range(num_experiments):
            actor = ExperimentRemote.remote(
                data_dir,
                goal,
                model,
                feedback_model,
                eval_metric,
                tracking_enabled=MLFLOW_AVAILABLE,
                task_type=task_type,
                tracking_experiment=tracking_experiment,
                parent_run_id=parent_run_id,
                task_id=task_id,
                algotune_n_problems=algotune_n_problems,
                algotune_n_runs=algotune_n_runs,
                algotune_mode=algotune_mode,
            )
            actors.append(actor)
            logger.info(f"Created experiment actor {i+1}/{num_experiments}")
    else:
        # Subsequent iterations - reuse existing actors
        actors = experiment_actors
        logger.info(f"Reusing {len(actors)} existing actors for iteration {iteration}")

    logger.info(f"Launching {len(actors)} experiments with {steps_per_experiment} steps each")

    # Start all experiments with unique indices and iteration info
    result_refs = [actor.run.remote(
        steps=steps_per_experiment,
        experiment_idx=i,
        iteration=iteration
    ) for i, actor in enumerate(actors)]

    # Wait for results with progress tracking
    print(f"\n🚀 EXPERIMENTS STARTED: {num_experiments} agents running in parallel")
    if actor_gpu_fraction > 0:
        print(f"   GPU Fraction: {actor_gpu_fraction} | Experiments per GPU: {1/actor_gpu_fraction:.1f}")
    else:
        print("   Running on CPU workers")
    print(f"   Task: {task_type} | Steps per experiment: {steps_per_experiment}")
    print(f"   Waiting for all experiments to complete...\n")

    results = []
    completed_count = 0

    # Use ray.wait to get results as they complete (not in order)
    remaining_refs = list(result_refs)
    experiment_map = {ref: i for i, ref in enumerate(result_refs)}

    while remaining_refs:
        # Wait for at least one experiment to complete
        ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)

        for ref in ready_refs:
            exp_idx = experiment_map[ref]
            try:
                result = ray.get(ref)
            except Exception as exc:
                compact_error = _compact_exception(exc)
                logger.error(
                    "Experiment %s failed while fetching Ray result: %s",
                    exp_idx,
                    compact_error,
                )
                result = _ray_failure_result(
                    experiment_idx=exp_idx,
                    error=f"Ray result retrieval failed: {compact_error}",
                )
            results.append(result)
            completed_count += 1

            # Summary line for completed experiment
            from datetime import datetime
            completion_time = datetime.now().strftime("%H:%M:%S")
            metric_str = f"{result.get('valid_metric'):.4f}" if result.get('valid_metric') is not None else "None"
            print(f"[{completion_time}] ✅ [{completed_count}/{num_experiments}] Experiment {exp_idx} done | "
                  f"GPU {result.get('gpu_id', 'N/A')} | "
                  f"Metric: {metric_str}")

    print(f"\n🎉 ALL {num_experiments} EXPERIMENTS COMPLETED!")

    logger.info("All experiments completed")

    # Rank results by metric score
    ranked_results = sorted(
        results,
        key=_metric_score,
        reverse=True,
    )

    # Return both results and actors for reuse
    return ranked_results, actors


def initialize_ray_cluster(head_node_ip: Optional[str] = None) -> None:
    """Initialize Ray connection to cluster or start local instance with runtime_env."""

    # Define the project root
    project_root = PROJECT_ROOT

    env_vars = {
        "AIDE_PROJECT_ROOT": str(project_root),
        "PYTHONPATH": str(project_root),
        "KERNELBENCH_PATH": str(project_root / "tasks" / "kernel_bench" / "KernelBench"),
        "ALGOTUNE_PATH": os.getenv("ALGOTUNE_PATH", str(DEFAULT_ALGOTUNE_PATH)),
    }
    for key, value in os.environ.items():
        if value and key.startswith(RUNTIME_ENV_PREFIXES):
            env_vars[key] = value

    # Create a comprehensive runtime_env
    runtime_env = {
        # Set working directory - Ray will tar and distribute this entire directory
        "working_dir": str(project_root),

        # Environment variables needed on all nodes
        "env_vars": dict(env_vars),

        # Note: Required packages should be pre-installed on worker nodes
        # Installing via pip in runtime_env can cause virtualenv issues

        # Explicitly exclude large directories to reduce package size
        "excludes": [
            "venv/",
            ".venv/",
            ".venv*/",
            "workspaces/",
            "mlruns/",
            "ray_results/",
            "__pycache__/",
            "*.pyc",
            "*.log",
            ".git/",
            "logs/",
            "test_*.py",
            "*_test.py",
        ],
    }

    # Load local env files and add to runtime_env. .env.local overrides .env.
    from dotenv import dotenv_values

    env_files = [project_root / ".env", project_root / ".env.local"]
    for env_file in env_files:
        if not env_file.exists():
            continue
        loaded_env_vars = dotenv_values(env_file)
        for key, value in loaded_env_vars.items():
            if value:
                runtime_env["env_vars"][key] = value
                env_vars[key] = value
        logger.info(f"Loaded {len(loaded_env_vars)} environment variables from {env_file.name}")

    if head_node_ip:
        # Connect to existing Ray cluster with runtime_env
        ray_address = f"{head_node_ip}:6379"  # Direct GCS connection
        logger.info(f"Connecting to Ray cluster at {ray_address}")
        logger.info(f"Distributing working directory: {project_root}")
        ray.init(address=ray_address, runtime_env=runtime_env)
    else:
        # Local runs use the existing filesystem and interpreter, so avoid
        # packaging the entire repository as a Ray working_dir.
        local_runtime_env = {"env_vars": env_vars}
        try:
            ray.init(address="auto", runtime_env=local_runtime_env)
            logger.info("Connected to existing local Ray instance")
        except:
            logger.info("Starting new local Ray instance")
            ray.init(runtime_env=local_runtime_env)

    # Display cluster information
    resources = get_cluster_resources()
    logger.info(f"Ray cluster initialized with {resources['total_cpus']} CPUs and {resources['total_gpus']} GPUs")

    # Get node information
    nodes = ray.nodes()
    logger.info(f"Cluster has {len(nodes)} nodes:")
    for node in nodes:
        if node['Alive']:
            node_resources = node['Resources']
            logger.info(f"  - Node {node['NodeID'][:8]}: {node_resources.get('CPU', 0)} CPUs, {node_resources.get('GPU', 0)} GPUs")


def main(
    num_experiments: int,
    model: str,
    feedback_model: str,
    num_iterations: int,
    data_dir: str,
    goal: str,
    eval_metric: str,
    steps_per_experiment: int = 2,
    head_node_ip: Optional[str] = None,
    task_type: str = "attention",
    task_id: str | None = None,
    gpu_fraction: float = 1.0,
    cpus_per_experiment: int | None = None,
    tracking_experiment: str | None = None,
    algotune_n_problems: int | None = None,
    algotune_n_runs: int | None = None,
    algotune_mode: str = STRICT_ALGOTUNE_MODE,
) -> dict[str, Any]:
    """Main execution function with GPU-aware Ray cluster support."""

    if task_type == "algotune" and _is_strict_algotune_mode(algotune_mode):
        _validate_algotune_strict_environment()

    use_direct_execution = _use_direct_single_experiment_mode(
        task_type=task_type,
        num_experiments=num_experiments,
        head_node_ip=head_node_ip,
    )
    ray_initialized = False
    if not use_direct_execution:
        initialize_ray_cluster(head_node_ip)
        ray_initialized = True
    parent_logger = None

    try:
        parent_run_id = None
        if MLFLOW_AVAILABLE:
            from mlflow_integration import create_mlflow_logger_for_experiment

            parent_logger = create_mlflow_logger_for_experiment(
                experiment_name=f"{task_type}_campaign",
                gpu_id=-1,
                config={
                    "task_type": task_type,
                    "task_id": task_id,
                    "at_mode": algotune_mode,
                    "non_reportable": _is_non_reportable_run(task_type, algotune_mode),
                    "model": model,
                    "feedback_model": feedback_model,
                    "num_experiments": num_experiments,
                    "num_iterations": num_iterations,
                    "steps_per_experiment": steps_per_experiment,
                    "gpu_fraction": gpu_fraction,
                    "cpus_per_experiment": cpus_per_experiment,
                },
                tracking_experiment=tracking_experiment,
            )
            if parent_logger and parent_logger.enabled:
                parent_run_id = parent_logger.run_id

        # Run initial batch of experiments
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting initial batch of {num_experiments} experiments")
        logger.info(f"{'='*60}")

        ranked_results, experiment_actors = run_experiments(
            data_dir=str(data_dir),
            goal=goal,
            eval_metric=eval_metric,
            model=model,
            feedback_model=feedback_model,
            num_experiments=num_experiments,
            steps_per_experiment=steps_per_experiment,
            task_type=task_type,
            gpu_fraction=gpu_fraction,
            cpus_per_experiment=cpus_per_experiment,
            tracking_experiment=tracking_experiment,
            parent_run_id=parent_run_id,
            task_id=task_id,
            algotune_n_problems=algotune_n_problems,
            algotune_n_runs=algotune_n_runs,
            algotune_mode=algotune_mode,
            use_direct_execution=use_direct_execution,
            iteration=1,
            experiment_actors=None  # Create new actors for first iteration
        )

        logger.info("-" * 60)
        logger.info(f"Initial results: {[result['valid_metric'] for result in ranked_results]}")
        logger.info(f"GPU assignments: {[(r.get('gpu_id', 'CPU')) for r in ranked_results]}")
        logger.info("-" * 60)

        total_completed = len(ranked_results)

        best_result = ranked_results[0]
        best_score = _metric_score(best_result)

        logger.info(f"Best Initial Metric: {best_result['valid_metric']} (GPU: {best_result.get('gpu_id', 'N/A')})")

        # Iterative refinement
        iteration = 1
        while iteration < num_iterations:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")

            try:
                # Create revised goal based on best solution
                revised_goal = f"""{goal}

                You have already built a solution that achieves a metric of {best_result['valid_metric']}
                with the following code:

                {_load_best_code(best_result) or "# No code artifact available."}

                Now, you need to build a solution that improves upon this metric.
                Focus on optimization techniques that could further enhance performance.
                """

                revised_experiments, experiment_actors = run_experiments(
                    data_dir=str(data_dir),
                    goal=revised_goal,
                    eval_metric=eval_metric,
                    model=model,
                    feedback_model=feedback_model,
                    num_experiments=num_experiments,
                    steps_per_experiment=steps_per_experiment,
                    task_type=task_type,
                    gpu_fraction=gpu_fraction,
                    cpus_per_experiment=cpus_per_experiment,
                    tracking_experiment=tracking_experiment,
                    parent_run_id=parent_run_id,
                    task_id=task_id,
                    algotune_n_problems=algotune_n_problems,
                    algotune_n_runs=algotune_n_runs,
                    algotune_mode=algotune_mode,
                    use_direct_execution=use_direct_execution,
                    iteration=iteration + 1,
                    experiment_actors=experiment_actors  # Reuse existing actors
                )

                batch_best = revised_experiments[0]

                logger.info("-" * 60)
                logger.info(f"Iteration {iteration + 1} results: {[r['valid_metric'] for r in revised_experiments]}")
                logger.info(f"Batch Best Metric: {batch_best['valid_metric']} (GPU: {batch_best.get('gpu_id', 'N/A')})")
                logger.info("-" * 60)

                total_completed += len(revised_experiments)

                # Update best result if improved
                candidate = batch_best
                candidate_score = _metric_score(candidate)
                if candidate_score > best_score:
                    best_result = candidate
                    best_score = candidate_score
                    logger.info("✓ New overall best solution found!")
                else:
                    logger.info("No improvement; keeping previous best solution.")

                logger.info(f"Overall Best Metric: {best_result['valid_metric']}")

                iteration += 1

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                iteration += 1

        final_result: dict[str, Any] | None = None
        if task_type == "algotune" and _is_strict_algotune_mode(algotune_mode):
            if _result_status(task_type, best_result) == "succeeded":
                logger.info("Running strict final AlgoTune evaluation on held-out test split...")
                final_result = _run_algotune_strict_final_evaluation(
                    task_id=task_id,
                    best_result=best_result,
                )
            else:
                final_result = {
                    "mode": "benchmark_strict",
                    "split": "test",
                    "status": "failed",
                    "valid_metric": None,
                    "compiled": best_result.get("compiled"),
                    "correct": None,
                    "error": "Search did not produce a valid train-split candidate; skipping final test evaluation",
                }

        report_result = _select_reportable_result(task_type, algotune_mode, best_result, final_result)
        overall_status = _result_status(task_type, report_result)

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Metric: {report_result['valid_metric']}")
        if report_result["valid_metric"] is None:
            logger.warning(
                "No valid metric was produced. The environment ran successfully, "
                "but the selected model did not generate an evaluable solution. "
                "Try a stronger model, more steps, or run ./cli/aide-check to verify setup."
            )
        if final_result is not None:
            logger.info(
                "Strict final test metric: %s",
                final_result.get("valid_metric"),
            )
        logger.info(f"Best GPU: {best_result.get('gpu_id', 'N/A')} ({best_result.get('gpu_name', 'N/A')})")
        logger.info(f"Total Experiments Completed: {total_completed}")
        logger.info("-" * 60)
        logger.info("Best Code:")
        logger.info(_truncate_text(_load_best_code(best_result) or "# No code artifact available.", MAX_RESULT_CODE_CHARS))
        logger.info("=" * 60)

        # Finish all MLflow runs after all iterations complete
        if experiment_actors:
            logger.info("Finalizing MLflow runs for all experiments...")
            try:
                if use_direct_execution:
                    for actor in experiment_actors:
                        actor.finish_all_iterations()
                else:
                    finish_refs = [actor.finish_all_iterations.remote() for actor in experiment_actors]
                    ray.get(finish_refs)  # Wait for all to finish
            except Exception as exc:
                logger.error("Failed to finalize one or more MLflow runs: %s", _compact_exception(exc))
            logger.info("All MLflow runs finalized")

        if parent_logger:
            parent_logger.log_experiment_summary(
                {
                    "best_metric": report_result["valid_metric"],
                    "total_experiments_completed": total_completed,
                    "status": overall_status,
                    "compiled": report_result.get("compiled"),
                    "correct": report_result.get("correct"),
                    "error": report_result.get("error"),
                    "best_code": _load_best_code(best_result),
                    "search_best_metric": best_result.get("valid_metric"),
                    "final_test_metric": final_result.get("valid_metric") if final_result else None,
                }
            )
            if task_type == "algotune" and _is_strict_algotune_mode(algotune_mode):
                parent_logger.log_task_outcome(
                    task_type=task_type,
                    status=_result_status(task_type, best_result),
                    metric_value=best_result.get("valid_metric"),
                    compiled=best_result.get("compiled"),
                    correct=best_result.get("correct"),
                    error=best_result.get("error"),
                    phase="search",
                )
                parent_logger.log_task_outcome(
                    task_type=task_type,
                    status=overall_status,
                    metric_value=report_result.get("valid_metric"),
                    compiled=report_result.get("compiled"),
                    correct=report_result.get("correct"),
                    error=report_result.get("error"),
                    phase="final_test",
                )
            else:
                parent_logger.log_task_outcome(
                    task_type=task_type,
                    status=overall_status,
                    metric_value=report_result.get("valid_metric"),
                    compiled=report_result.get("compiled"),
                    correct=report_result.get("correct"),
                    error=report_result.get("error"),
                )
            parent_logger.finish()
            parent_logger = None

        return {
            "status": overall_status,
            "task_type": task_type,
            "task_id": task_id,
            "num_experiments": num_experiments,
            "num_iterations": num_iterations,
            "steps_per_experiment": steps_per_experiment,
            "model": model,
            "feedback_model": feedback_model,
            "gpu_fraction": gpu_fraction,
            "cpus_per_experiment": cpus_per_experiment,
            "tracking_experiment": tracking_experiment,
            "algotune_eval": {
                "mode": algotune_mode,
                "search_split": "train" if task_type == "algotune" else None,
                "final_split": "test" if task_type == "algotune" else None,
            },
            "at_mode": algotune_mode,
            "non_reportable": _is_non_reportable_run(task_type, algotune_mode),
            "search_result": best_result,
            "final_result": final_result,
            "best_result": best_result,
            "report_result": report_result,
            "total_experiments_completed": total_completed,
        }

    finally:
        if parent_logger:
            try:
                parent_logger.finish(status="FAILED")
            except Exception:
                pass
        if ray_initialized:
            ray.shutdown()
            logger.info("Ray cluster shutdown complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AIDE experiments with Ray-based parallelization")
    parser.add_argument("--task", type=str, default="attention", help="Task name (attention, kernel, kernelbench, or algotune)")
    parser.add_argument(
        "--head-node-ip",
        type=str,
        default=os.getenv("AIDE_HEAD_NODE_IP"),
        help="Ray head-node IP. Defaults to AIDE_HEAD_NODE_IP when set.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override number of steps per experiment")
    parser.add_argument("--num-experiments", type=int, default=None, help="Override number of parallel experiments")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override number of iterations")
    parser.add_argument("--local", action="store_true", help="Run locally instead of on cluster")
    parser.add_argument("--gpu-fraction", type=float, default=1.0,
                       help="Fraction of GPU per experiment (e.g., 0.5 for 2 experiments per GPU, 0.25 for 4, 0.1 for 10)")
    parser.add_argument("--cpus-per-experiment", type=int, default=None,
                       help="CPUs per experiment for CPU-only runs. Defaults to auto-calculated scheduling.")
    parser.add_argument(
        "--tracking-experiment",
        dest="tracking_experiment",
        type=str,
        default=None,
        help="MLflow experiment name override",
    )
    parser.add_argument("--wandb-project", dest="tracking_experiment", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, default=None,
                       help="Override model from contract (e.g., gpt-4, claude-3-opus, o1-mini)")
    parser.add_argument("--feedback-model", type=str, default=None,
                       help="Override feedback model from contract")
    parser.add_argument("--kb-task", type=str, default=None,
                       help="KernelBench task ID (e.g., 1_19, 3_28)")
    parser.add_argument("--at-task", type=str, default=None,
                       help="AlgoTune task name (e.g., kmeans, qr_factorization, svm)")
    parser.add_argument("--result-json", type=str, default=None,
                       help="Write final run metadata to this JSON file")

    args = parser.parse_args()

    # Load task configuration
    base_dir = PROJECT_ROOT
    data_dir = base_dir / "tasks" / args.task

    with open(data_dir / "contract.yaml", "r") as f:
        contract = yaml.safe_load(f)

    # Handle KernelBench task-specific configuration
    if args.task == "kernelbench":
        if not args.kb_task:
            logger.error("--kb-task is required for kernelbench tasks")
            sys.exit(1)

        # Import kb_tasks module and prepare task
        sys.path.insert(0, str(data_dir))

        # First, prepare the task by injecting original code into optimize.py
        from prepare_kernelbench_task import prepare_task
        optimize_path = data_dir / "optimize.py"

        logger.info(f"Preparing KernelBench task {args.kb_task}...")
        prepare_task(args.kb_task, str(optimize_path))

        # Now get enhanced task info with rich goals
        from kb_tasks import get_task_info_with_code
        task_info = get_task_info_with_code(args.kb_task)

        # Update contract with rich task-specific information
        contract["goal"] = task_info["goal"]  # Now includes full code and examples

        # Update eval command with task ID - evaluate_gpu.py is in the input directory
        # Note: eval command runs from working directory, so need ../input to reach sibling input directory
        contract["eval"] = f"python ../input/evaluate_gpu.py --task-id {args.kb_task} --solution-path optimize.py --device cuda"

        # Use task-suggested steps if not overridden
        if not args.steps:
            contract["steps"] = task_info["suggested_steps"]

        logger.info(f"KernelBench task: {task_info['name']} (Level {task_info['level']})")
        logger.info(f"Task prepared with original code and examples")
    elif args.task == "algotune":
        if not args.at_task:
            logger.error("--at-task is required for algotune tasks")
            sys.exit(1)

        sys.path.insert(0, str(data_dir))

        from prepare_algotune_task import prepare_task
        solver_path = data_dir / "solver.py"

        logger.info(f"Preparing AlgoTune task {args.at_task}...")
        prepare_task(args.at_task, str(solver_path))

        from at_tasks import get_task_info_with_code
        task_info = get_task_info_with_code(args.at_task)

        algotune_mode = STRICT_ALGOTUNE_MODE
        algotune_n_problems = None
        algotune_n_runs = None

        contract["goal"] = task_info["goal"]
        contract["eval"] = (
            f"python ../input/evaluate_algotune.py --task {args.at_task} "
            "--solution-path solver.py --split test"
        )

        if not args.steps:
            contract["steps"] = task_info.get("suggested_steps", contract.get("steps", 4))

        logger.info(
            "AlgoTune task: %s (%s)",
            task_info["name"],
            task_info.get("category", "general"),
        )
        logger.info("Task prepared with description and reference implementation")
    else:
        algotune_n_problems = None
        algotune_n_runs = None
        algotune_mode = STRICT_ALGOTUNE_MODE

    logger.info(f"Task configuration: {contract}")

    # Override contract values if specified
    num_experiments = args.num_experiments or contract["num_experiments"]
    num_iterations = args.num_iterations or contract["num_iterations"]
    steps = args.steps or contract.get("steps", 2)

    # Use local mode if specified
    head_node_ip = None if args.local else args.head_node_ip

    # Apply model overrides if specified
    model = args.model if args.model else contract["model"]
    feedback_model = args.feedback_model if args.feedback_model else contract["feedback_model"]

    # Log the models being used
    logger.info(f"Using model: {model}")
    logger.info(f"Using feedback model: {feedback_model}")
    _validate_model_credentials(model, feedback_model)

    try:
        result_payload = main(
            num_experiments=num_experiments,
            model=model,
            feedback_model=feedback_model,
            num_iterations=num_iterations,
            data_dir=data_dir,
            goal=contract["goal"],
            eval_metric=contract["eval_metric"],
            steps_per_experiment=steps,
            head_node_ip=head_node_ip,
            task_type=args.task,
            task_id=args.kb_task if args.task == "kernelbench" else (args.at_task if args.task == "algotune" else None),
            gpu_fraction=args.gpu_fraction,
            cpus_per_experiment=args.cpus_per_experiment,
            tracking_experiment=args.tracking_experiment,
            algotune_n_problems=algotune_n_problems,
            algotune_n_runs=algotune_n_runs,
            algotune_mode=algotune_mode,
        )
    except Exception as exc:
        if args.result_json:
            _write_json(
                args.result_json,
                {
                    "status": "failed",
                    "task_type": args.task,
                    "task_id": args.kb_task if args.task == "kernelbench" else (args.at_task if args.task == "algotune" else None),
                    "at_mode": algotune_mode if args.task == "algotune" else None,
                    "non_reportable": _is_non_reportable_run(args.task, algotune_mode) if args.task == "algotune" else False,
                    "error": _compact_exception(exc),
                },
            )
        raise

    if args.result_json:
        _write_json(args.result_json, result_payload)
