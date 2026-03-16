"""
Ray-based parallel experiment runner with GPU support.
Distributes experiments across multiple GPUs on the cluster.
"""

from pathlib import Path
from typing import Any, Optional
import os
import sys

# Note: Path manipulation is now handled by Ray runtime_env
# The runtime_env will ensure all required paths are available on worker nodes

from dotenv import load_dotenv
import ray
import yaml
import logging

# Import AIDE conditionally - it will be imported in the actor
try:
    import aide
    AIDE_AVAILABLE = True
except ImportError:
    AIDE_AVAILABLE = False
    print("AIDE will be imported inside Ray actors")

if os.getenv("AIDE_ENABLE_WANDB", "0") == "1":
    try:
        from wandb_integration import create_wandb_logger_for_experiment, WandbCallback
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
else:
    WANDB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALGOTUNE_PATH = PROJECT_ROOT / "tasks" / "algotune" / "vendor" / "AlgoTune"
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# Note: num_gpus will be set dynamically in run_experiments()
class Experiment:
    def __init__(self, data_dir: str, goal: str, model: str, feedback_model: str, eval_metric: str | None, wandb_enabled: bool = False, task_type: str = "attention", wandb_project: str = None, task_id: str = None):
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
        self.wandb_enabled = wandb_enabled
        self.task_type = task_type
        self.task_id = task_id
        self.wandb_project = wandb_project
        self.wandb_logger = None  # Persistent W&B logger across iterations
        self.total_steps_completed = 0  # Track total steps across all iterations

        # Set W&B project if specified
        if wandb_project:
            os.environ["WANDB_PROJECT"] = wandb_project
            logger.info(f"Set W&B project to: {wandb_project}")

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

    def run(self, steps: int = 2, experiment_idx: int = 0, iteration: int = 1, previous_step_count: int = 0) -> dict[str, Any]:
        """Run the AIDE experiment with GPU device properly configured and W&B logging.

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

        # Import wandb_integration inside the actor if enabled
        if self.wandb_enabled:
            try:
                from wandb_integration import create_wandb_logger_for_experiment
                wandb_available = True
            except ImportError:
                wandb_available = False
                logger.warning("W&B integration not available on this node")
        else:
            wandb_available = False

        # Set CUDA_VISIBLE_DEVICES to ensure subprocess uses correct GPU
        if self.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            logger.info(f"Set CUDA_VISIBLE_DEVICES={self.gpu_id} for experiment")

        # Initialize W&B logger for this experiment if available (only on first iteration)
        if wandb_available and self.wandb_logger is None:
            self.wandb_logger = create_wandb_logger_for_experiment(
                experiment_name=f"{self.task_type}_exp{experiment_idx}_gpu{self.gpu_id if self.gpu_id is not None else 'cpu'}",
                gpu_id=self.gpu_id if self.gpu_id is not None else -1,
                config={
                    "task_type": self.task_type,
                    "model": self.model,
                    "feedback_model": self.feedback_model,
                    "eval_metric": self.eval_metric,
                    "gpu_name": self.gpu_name,
                    "data_dir": str(self.data_dir),
                    "total_iterations": iteration,  # Will be updated in each iteration
                    "steps_per_iteration": steps,
                    "experiment_idx": experiment_idx
                }
            )

        wandb_logger = self.wandb_logger  # Use persistent logger

        # Log iteration start if we have a logger
        if wandb_logger:
            wandb_logger.log_metrics({
                "iteration": iteration,
                "iteration_started": 1
            }, step=self.total_steps_completed)

        # Import AIDE here where it's needed
        # Add aideml to path since runtime_env distributes files
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "aideml"))
        import aide

        exp = aide.Experiment(
            data_dir=self.data_dir,
            goal=self.goal,
            eval=self.eval_metric,
            task_type=self.task_type,
            task_id=self.task_id,
        )

        # Configure the experiment
        exp.cfg.agent.code.model = self.model
        exp.cfg.agent.feedback.model = self.feedback_model
        exp.cfg.report.model = self.model
        exp.cfg.exec.timeout = 3600  # seconds

        # Run the experiment with W&B tracking
        logger.info(f"Starting experiment {experiment_idx} on GPU {self.gpu_id} - Iteration {iteration}")

        # Track each step with cumulative count
        for step in range(steps):
            global_step = self.total_steps_completed + step
            if wandb_logger:
                wandb_logger.log_metrics({
                    "aide_step": step + 1,
                    "global_step": global_step + 1,
                    "iteration": iteration
                }, step=global_step)

            # Run one step with OOM protection
            try:
                exp.agent.step(exec_callback=exp.interpreter.run)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                    logger.error(f"GPU OOM on experiment {experiment_idx}, step {step}: {e}")
                    # Try to recover by clearing cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    # Skip this step and continue
                    continue
                else:
                    raise  # Re-raise if not OOM error

            # Get current best node for logging
            best_node = exp.journal.get_best_node(only_good=False)
            if wandb_logger and best_node:
                # Determine if evaluation succeeded or failed
                if best_node.metric and best_node.metric.value is not None:
                    # Successful evaluation
                    metric_value = float(best_node.metric.value)
                    eval_status = "success"
                else:
                    # Failed evaluation (compilation error, runtime error, or correctness failure)
                    metric_value = 0.0  # Use 0.0 for failed evaluations
                    eval_status = "failed"
                    logger.info(f"Step {step}: Evaluation failed, logging with speedup=0.0")

                # Log based on task type using global step
                if self.task_type in ["kernel", "kernelbench", "algotune"]:
                    # For speedup tasks, metric is speedup (higher is better)
                    wandb_logger.log_evaluation(
                        speedup=metric_value,
                        execution_time=best_node.exec_time if best_node.exec_time else 0.0,
                        eval_status=eval_status,
                        step=global_step
                    )
                else:
                    # For attention task, metric is validation loss (lower is better)
                    # For failed evaluations, use a large loss value
                    if eval_status == "failed":
                        metric_value = float('inf')  # Use inf for failed attention tasks
                    wandb_logger.log_evaluation(
                        val_loss=metric_value,
                        training_time=best_node.exec_time if best_node.exec_time else 0.0,
                        eval_status=eval_status,
                        step=global_step
                    )

                # Always log code for this step, regardless of evaluation success
                wandb_logger.log_code(best_node.code, name=f"code_iter{iteration}_step{step}", step=global_step)

            # Periodic memory cleanup for fractional GPU usage
            if torch.cuda.is_available() and (step + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug(f"Cleared GPU cache at step {step + 1}")

        # Get final best solution
        best_solution = exp.journal.get_best_node(only_good=False)

        # Update total steps completed
        self.total_steps_completed += steps

        # Log iteration summary if W&B is available
        if wandb_logger:
            # Determine metric value for failed vs successful evaluations
            if best_solution and best_solution.metric and best_solution.metric.value is not None:
                metric_value = float(best_solution.metric.value)
                eval_status = "success"
            else:
                # Failed evaluation
                if self.task_type in ["kernel", "kernelbench", "algotune"]:
                    metric_value = 0.0  # Failed speedup tasks get speedup of 0
                else:
                    metric_value = float('inf')  # Failed attention tasks get infinite loss
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

            wandb_logger.log_metrics(iteration_summary, step=self.total_steps_completed - 1)

            # Don't finish W&B run - keep it alive for next iteration!

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Log completion with timestamp
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

        return {
            "valid_metric": best_solution.metric.value if (best_solution and best_solution.metric and best_solution.metric.value is not None) else None,
            "code": best_solution.code if best_solution else None,
            "gpu_id": self.gpu_id,
            "gpu_name": self.gpu_name,
            "experiment_idx": experiment_idx
        }

    def finish_all_iterations(self):
        """Finish the W&B run after all iterations are complete."""
        if self.wandb_logger:
            self.wandb_logger.log_experiment_summary({
                "total_steps_completed": self.total_steps_completed,
                "final_status": "completed"
            })
            self.wandb_logger.finish()
            self.wandb_logger = None


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


def get_cluster_resources() -> dict:
    """Get information about available resources in the Ray cluster."""
    resources = ray.available_resources()
    return {
        "total_cpus": resources.get("CPU", 0),
        "total_gpus": resources.get("GPU", 0),
        "available_cpus": ray.available_resources().get("CPU", 0),
        "available_gpus": ray.available_resources().get("GPU", 0),
    }


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
    wandb_project: str = None,
    iteration: int = 1,
    experiment_actors: list = None,
    task_id: str = None
) -> list[dict[str, Any]]:
    """Launch Ray actors with GPU or CPU allocation, wait for all results, and return ranked outputs."""

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
            actor = ExperimentRemote.remote(data_dir, goal, model, feedback_model, eval_metric, wandb_enabled=WANDB_AVAILABLE, task_type=task_type, wandb_project=wandb_project, task_id=task_id)
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
            result = ray.get(ref)
            results.append(result)
            completed_count += 1
            exp_idx = experiment_map[ref]

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
            "wandb/",
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

    # Load .env file and add to runtime_env
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import dotenv_values
        loaded_env_vars = dotenv_values(env_file)
        # Add API keys and other secrets to runtime_env
        for key, value in loaded_env_vars.items():
            if value and value != "YOUR_WANDB_KEY_HERE":
                runtime_env["env_vars"][key] = value
                env_vars[key] = value
        logger.info(f"Loaded {len(loaded_env_vars)} environment variables from .env")

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
    wandb_project: str | None = None,
) -> None:
    """Main execution function with GPU-aware Ray cluster support."""

    # Initialize Ray cluster
    initialize_ray_cluster(head_node_ip)

    try:
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
            wandb_project=wandb_project,
            task_id=task_id,
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

                {best_result['code']}

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
                    wandb_project=wandb_project,
                    task_id=task_id,
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

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Metric: {best_result['valid_metric']}")
        if best_result["valid_metric"] is None:
            logger.warning(
                "No valid metric was produced. The environment ran successfully, "
                "but the selected model did not generate an evaluable solution. "
                "Try a stronger model, more steps, or run ./cli/aide-check to verify setup."
            )
        logger.info(f"Best GPU: {best_result.get('gpu_id', 'N/A')} ({best_result.get('gpu_name', 'N/A')})")
        logger.info(f"Total Experiments Completed: {total_completed}")
        logger.info("-" * 60)
        logger.info("Best Code:")
        logger.info(best_result['code'])
        logger.info("=" * 60)

        # Finish all W&B runs after all iterations complete
        if experiment_actors:
            logger.info("Finalizing W&B runs for all experiments...")
            finish_refs = [actor.finish_all_iterations.remote() for actor in experiment_actors]
            ray.get(finish_refs)  # Wait for all to finish
            logger.info("All W&B runs finalized")

    finally:
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
    parser.add_argument("--wandb-project", type=str, default=None,
                       help="W&B project name (default: aide-{task}-h100)")
    parser.add_argument("--model", type=str, default=None,
                       help="Override model from contract (e.g., gpt-4, claude-3-opus, o1-mini)")
    parser.add_argument("--feedback-model", type=str, default=None,
                       help="Override feedback model from contract")
    parser.add_argument("--kb-task", type=str, default=None,
                       help="KernelBench task ID (e.g., 1_19, 3_28)")
    parser.add_argument("--at-task", type=str, default=None,
                       help="AlgoTune task name (e.g., kmeans, qr_factorization, svm)")

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

        contract["goal"] = task_info["goal"]
        contract["eval"] = (
            f"python ../input/evaluate_algotune.py --task {args.at_task} "
            "--solution-path solver.py --n-problems 5 --n-runs 3 --fast"
        )

        if not args.steps:
            contract["steps"] = task_info.get("suggested_steps", contract.get("steps", 4))

        logger.info(
            "AlgoTune task: %s (%s)",
            task_info["name"],
            task_info.get("category", "general"),
        )
        logger.info("Task prepared with description and reference implementation")

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

    main(
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
        wandb_project=args.wandb_project,
    )
