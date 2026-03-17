"""
Python interpreter for executing code snippets and capturing their output.
    Supports:
- captures stdout and stderr
- captures exceptions and stack traces
"""

import logging
import os
import queue
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("aide")
STRICT_ALGOTUNE_MODE = "benchmark_strict"


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [
                line
                for line in tb_lines
                if "aide/" not in line and "importlib" not in line
            ]
        )
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue, timeout=5):
        self.queue = queue
        self.timeout = timeout

    def write(self, msg):
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            logger.warning("Queue write timed out")

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int | None = None,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        task_type: str = None,
        eval_cmd: str = None,
        task_id: str = None,
        algotune_n_problems: int | None = None,
        algotune_n_runs: int | None = None,
        algotune_mode: str = STRICT_ALGOTUNE_MODE,
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int | None, optional): Optional timeout for each code execution step. Defaults to None.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            task_type (str, optional): Type of task (e.g., "kernel", "kernelbench"). Defaults to None.
            eval_cmd (str, optional): Evaluation command to run after code execution for kernel tasks. Defaults to None.
            task_id (str, optional): Task ID for kernel tasks. Defaults to None.
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        self.task_type = task_type
        self.eval_cmd = eval_cmd
        self.task_id = task_id
        self.algotune_n_problems = None
        self.algotune_n_runs = None
        self.algotune_mode = STRICT_ALGOTUNE_MODE
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()
        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in the data preview)
            os.remove(self.agent_file_name)

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup_session(self):
        if self.process is None:
            return
        try:
            # Reduce grace period from 2 seconds to 0.5
            self.process.terminate()
            self.process.join(timeout=0.5)

            if self.process.exitcode is None:
                logger.warning("Process failed to terminate, killing immediately")
                self.process.kill()
                self.process.join(timeout=0.5)

                if self.process.exitcode is None:
                    logger.error("Process refuses to die, using SIGKILL")
                    os.kill(self.process.pid, signal.SIGKILL)
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")
        finally:
            if self.process is not None:
                self.process.close()
                self.process = None

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # wait for child to actually start execution (we don't want interrupt child setup)
        while True:
            try:
                state = self.event_outq.get(timeout=1)
                break
            except queue.Empty:
                if self.process is None or not self.process.is_alive():
                    msg = "REPL child process failed to start execution"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(f"REPL output queue dump: {self.result_outq.get()}")
                    raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
        # if the child process dies without this flag being set, it's an unexpected termination
        child_in_overtime = False

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    logger.warning(f"Execution exceeded timeout of {self.timeout}s")
                    os.kill(self.process.pid, signal.SIGINT)
                    child_in_overtime = True

                    # terminate if we're overtime by more than 5 seconds
                    if running_time > self.timeout + 5:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()

                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        # read all stdout/stderr from child up to the EOF marker
        # waiting until the queue is empty is not enough since
        # the feeder thread in child might still be adding to the queue
        start_collect = time.time()
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                if self.process is None or not self.process.is_alive():
                    if self.result_outq.empty():
                        break
                continue
        if output and output[-1] == "<|EOF|>":
            output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            if self.timeout is not None:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds."
                )

        # For speedup tasks, run the evaluation directly (not as subprocess).
        if self.task_type in ["kernel", "kernelbench", "algotune"] and self.eval_cmd and e_cls_name is None:
            logger.info(f"Running evaluation for {self.task_type} task")
            import sys
            from pathlib import Path

            working_dir_path = Path(self.working_dir)

            if self.task_type == "algotune":
                solver_path = self.working_dir / "solver.py"
                with open(solver_path, "w") as f:
                    f.write(code)

                try:
                    logger.debug(f"Running AlgoTune evaluation for task {self.task_id}")

                    algotune_path = working_dir_path.parent.parent / "tasks" / "algotune"
                    if algotune_path.exists():
                        sys.path.insert(0, str(algotune_path))

                    from evaluate_algotune import evaluate_task

                    eval_results = evaluate_task(
                        task_name=self.task_id,
                        solver_path=str(solver_path),
                        split="train",
                    )

                    output.append("\n=== AlgoTune Evaluation Output ===\n")
                    output.append(
                        "Eval config: "
                        f"mode={self.algotune_mode}, "
                        "split=train\n"
                    )
                    output.append(f"Compiled: {'✓' if eval_results.get('compiled', False) else '✗'}\n")
                    output.append(f"Correct: {'✓' if eval_results.get('correct', False) else '✗'}\n")
                    output.append(f"Speedup: {eval_results.get('speedup', 0.0):.4f}\n")

                    speedup_value = eval_results.get("speedup", 0.0)
                    output.append(f"\nspeedup: {speedup_value:.4f}\n")

                    if eval_results.get("error"):
                        output.append(f"[Eval error]: {eval_results['error']}\n")

                    logger.info(f"AlgoTune evaluation complete - speedup: {speedup_value}")
                except ImportError as e:
                    output.append(f"[Eval error]: Failed to import AlgoTune evaluation module: {str(e)}")
                    logger.error(f"Import error during AlgoTune evaluation: {e}")
                except Exception as e:
                    output.append(f"[Eval error]: Failed to run AlgoTune evaluation: {str(e)}")
                    logger.error(f"AlgoTune evaluation failed: {e}")
            else:
                optimize_path = self.working_dir / "optimize.py"
                with open(optimize_path, 'w') as f:
                    f.write(code)

                try:
                    logger.debug(f"Running direct evaluation for task {self.task_id}")

                    kernelbench_path = working_dir_path.parent.parent / "tasks" / "kernelbench"
                    kb_library_path = working_dir_path.parent.parent / "tasks" / "kernel_bench" / "KernelBench"

                    if kernelbench_path.exists():
                        sys.path.insert(0, str(kernelbench_path))
                    if kb_library_path.exists():
                        sys.path.insert(0, str(kb_library_path))

                    from evaluate_gpu import evaluate_kernelbench_task

                    eval_results = evaluate_kernelbench_task(
                        task_id=self.task_id,
                        solution_path=str(optimize_path),
                        device="cuda",
                        num_correct_trials=5,
                        num_perf_trials=100,
                        measure_performance=True,
                        verbose=True,
                        build_dir=str(self.working_dir / "cuda_build_cache")
                    )

                    output.append("\n=== Evaluation Output ===\n")
                    output.append(f"Compilation: {'✓' if eval_results.get('compiled', False) else '✗'}\n")
                    output.append(f"Correctness: {'✓' if eval_results.get('correct', False) else '✗'}\n")
                    output.append(f"Speedup: {eval_results.get('speedup', 0.0):.4f}\n")

                    speedup_value = eval_results.get('speedup', 0.0)
                    output.append(f"\nspeedup: {speedup_value:.4f}\n")

                    if eval_results.get('error'):
                        output.append(f"[Eval error]: {eval_results['error']}\n")

                    logger.info(f"Evaluation complete - speedup: {speedup_value}")
                except ImportError as e:
                    output.append(f"[Eval error]: Failed to import evaluation module: {str(e)}")
                    logger.error(f"Import error during evaluation: {e}")
                except Exception as e:
                    output.append(f"[Eval error]: Failed to run evaluation: {str(e)}")
                    logger.error(f"Evaluation failed: {e}")

        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
