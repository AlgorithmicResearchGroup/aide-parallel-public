from dataclasses import dataclass

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    _load_cfg,
    prep_cfg,
)


STRICT_ALGOTUNE_MODE = "benchmark_strict"


@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:
    def __init__(
        self,
        data_dir: str,
        goal: str,
        eval: str | None = None,
        task_type: str | None = None,
        task_id: str | None = None,
        algotune_n_problems: int | None = None,
        algotune_n_runs: int | None = None,
        algotune_mode: str | None = None,
    ):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
            task_type (str | None, optional): Type of task (e.g., "kernelbench", "kernel", "attention").
            task_id (str | None, optional): Task ID for kernel tasks (e.g., "1_19", "2_1").
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        _cfg.task_type = task_type if task_type else "default"
        _cfg.task_id = task_id  # Store task_id in config
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
            task_type=task_type,
        )
        # Pass task-specific parameters to the interpreter for speedup tasks.
        exec_config = OmegaConf.to_container(self.cfg.exec)
        if task_type in ["kernel", "kernelbench", "algotune"]:
            exec_config["task_type"] = task_type
            exec_config["eval_cmd"] = eval
            exec_config["task_id"] = task_id
            if task_type == "algotune":
                exec_config["algotune_n_problems"] = None
                exec_config["algotune_n_runs"] = None
                exec_config["algotune_mode"] = STRICT_ALGOTUNE_MODE

        self.interpreter = Interpreter(
            self.cfg.workspace_dir,
            **exec_config,  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            self.agent.step(exec_callback=self.interpreter.run)
            save_run(self.cfg, self.journal)
        self.interpreter.cleanup_session()

        best_node = self.journal.get_best_node(only_good=False)
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)
