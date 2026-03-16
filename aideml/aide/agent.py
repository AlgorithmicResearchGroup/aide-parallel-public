import logging
import random
from typing import Any, Callable, cast

import humanize
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": ["is_bug", "summary", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        task_type: str | None = None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.task_type = task_type if task_type else "default"
        self.is_gpu_task = task_type in ["kernelbench", "kernel"]
        self.is_algotune_task = task_type == "algotune"
        self.is_speedup_task = task_type in ["kernelbench", "kernel", "algotune"]

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            print("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                print("[search policy] debugging")
                return random.choice(debuggable_nodes)
            print("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            print("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        print("[search policy] greedy node selected")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    # GPU-specific prompt methods
    @property
    def _prompt_environment_gpu(self):
        pkgs = [
            "torch",
            "triton",
            "cupy",
            "numba",
            "torch.utils.cpp_extension",
            "torch.jit",
            "torch.compile",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use GPU optimization packages such as: {pkg_str}. "
                                 f"You can write custom CUDA kernels using torch.utils.cpp_extension.load_inline() "
                                 f"or use Triton for kernel fusion. PyTorch JIT compilation and torch.compile are also available."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline_gpu(self):
        impl_guideline = [
            "The code should **implement the optimized ModelNew class**.",
            "Create a class named exactly 'ModelNew' (not 'Model' or anything else).",
            "The ModelNew class MUST have EXACTLY the same interface as the original Model (same __init__ and forward signatures).",
            "The code should be a single-file Python program that is self-contained and can be executed as-is.",
            "Your implementation should focus on GPU performance optimization techniques.",
            "Common optimization patterns: kernel fusion, memory coalescing, shared memory usage, tensor cores.",
            f"Be aware of the running time - kernel compilation may take time but execution should be fast.",
            "The outputs must be numerically equivalent (within tolerance: rtol=1e-2, atol=1e-2).",
            "The evaluation system will automatically benchmark your ModelNew against the original Model.",
        ]

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt_gpu(self):
        return {
            "Response format": (
                "Your response should be a brief optimization strategy (3-5 sentences) explaining which operations "
                "you'll optimize and how (e.g., kernel fusion, custom CUDA kernels, Triton), "
                "followed by a single markdown code block (wrapped in ```) which implements the ModelNew class. "
                "There should be no additional headings or text. Just the strategy followed by the code block."
            )
        }

    @property
    def _prompt_impl_guideline_algotune(self):
        return {
            "Implementation guideline": [
                "Create a class named exactly 'Solver'.",
                "The class must define `def solve(self, problem, **kwargs)`.",
                "Return the same output format as the reference implementation.",
                "Focus on correctness first, then speedup.",
                "Your response should contain a single markdown code block implementing Solver.",
            ]
        }

    @property
    def _prompt_resp_fmt_algotune(self):
        return {
            "Response format": (
                "Respond with a brief optimization strategy in natural language (3-5 sentences), "
                "followed by a single markdown code block that implements the Solver class. "
                "Do not include headings or extra prose."
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                #temperature=self.acfg.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        if self.is_algotune_task:
            prompt: Any = {
                "Introduction": (
                    "You are optimizing a reference algorithm implementation for speed. "
                    "Produce a valid Solver class that preserves outputs while improving runtime."
                ),
                "Task description": self.task_desc,
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
            prompt["Instructions"] |= self._prompt_resp_fmt_algotune
            prompt["Instructions"] |= self._prompt_impl_guideline_algotune
        elif self.is_gpu_task:
            # GPU optimization-specific prompt
            prompt: Any = {
                "Introduction": (
                    "You are a GPU optimization expert working on kernel optimization. "
                    "Your goal is to create an optimized GPU implementation that runs faster while maintaining correctness. "
                    "You will be provided with an original implementation to optimize."
                ),
                "Task description": self.task_desc,
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
            prompt["Instructions"] |= self._prompt_resp_fmt_gpu
            prompt["Instructions"] |= {
                "Optimization strategy guideline": [
                    "Start with a straightforward optimization approach for this first attempt.",
                    "Consider the operations in the model and identify optimization opportunities.",
                    "Focus on one or two key optimizations rather than trying everything at once.",
                    "Take the Memory section into consideration to avoid repeating failed approaches.",
                    "The optimization strategy should be 3-5 sentences.",
                    "Target metric is speedup: Time(Original) / Time(Optimized) - higher is better.",
                ],
            }
            prompt["Instructions"] |= self._prompt_impl_guideline_gpu
            prompt["Instructions"] |= self._prompt_environment_gpu
        else:
            # Original ML competition prompt
            prompt: Any = {
                "Introduction": (
                    "You are an expert in transformer architectures and attention mechanisms. "
                    "Your goal is to optimize the multi-head attention implementation to minimize validation loss. "
                    "You should explore techniques like rotary embeddings, flash attention, gating mechanisms, "
                    "KV-caching, or other attention variants while maintaining compatibility with the existing interface."
                ),
                "Task description": self.task_desc,
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
            prompt["Instructions"] |= self._prompt_resp_fmt
            prompt["Instructions"] |= {
                "Solution sketch guideline": [
                    "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                    "Take the Memory section into consideration when proposing the design,"
                    " don't propose the same modelling solution but keep the evaluation the same.",
                    "The solution sketch should be 3-5 sentences.",
                    "Propose an evaluation metric that is reasonable for this task.",
                    "Don't suggest to do EDA.",
                    "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                ],
            }
            prompt["Instructions"] |= self._prompt_impl_guideline
            prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview and not self.is_gpu_task and not self.is_algotune_task:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        if self.is_algotune_task:
            prompt: Any = {
                "Introduction": (
                    "You are refining a Solver implementation for better speed while preserving correctness."
                ),
                "Task description": self.task_desc,
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
            prompt["Previous solution"] = {
                "Code": wrap_code(parent_node.code),
            }
            prompt["Instructions"] |= self._prompt_resp_fmt_algotune
            prompt["Instructions"] |= self._prompt_impl_guideline_algotune
        elif self.is_gpu_task:
            # GPU optimization-specific improvement prompt
            prompt: Any = {
                "Introduction": (
                    "You are a GPU optimization expert. You are provided with a previous GPU implementation "
                    "that needs further optimization to achieve better speedup. "
                    "Analyze the current implementation and identify specific optimization opportunities, "
                    "then implement an improved version."
                ),
                "Task description": self.task_desc,
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
            prompt["Previous solution"] = {
                "Code": wrap_code(parent_node.code),
            }

            prompt["Instructions"] |= self._prompt_resp_fmt_gpu
            prompt["Instructions"] |= {
                "Optimization improvement guideline": [
                    "Analyze the previous solution and identify a specific optimization opportunity.",
                    "Focus on one key improvement: better kernel fusion, reduced memory transfers, or improved parallelism.",
                    "The improvement should be measurable so we can evaluate the speedup gain.",
                    "Take the Memory section into consideration to avoid repeating failed optimizations.",
                    "The optimization strategy should be 3-5 sentences explaining the specific improvement.",
                ],
            }
            prompt["Instructions"] |= self._prompt_impl_guideline_gpu
        else:
            # Original ML competition improvement prompt
            prompt: Any = {
                "Introduction": (
                    "You are an expert in transformer architectures and attention mechanisms. "
                    "You are provided with a previous attention implementation that needs further optimization. "
                    "Analyze the current approach and identify opportunities for improvement such as reducing memory usage, "
                    "improving computational efficiency, or enhancing the attention pattern's effectiveness. "
                    "First outline your optimization strategy, then implement the improvements."
                ),
                "Task description": self.task_desc,
                "Memory": self.journal.generate_summary(),
                "Instructions": {},
            }
            prompt["Previous solution"] = {
                "Code": wrap_code(parent_node.code),
            }

            prompt["Instructions"] |= self._prompt_resp_fmt
            prompt["Instructions"] |= {
                "Solution improvement sketch guideline": [
                    "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                    "You should be very specific and should only propose a single actionable improvement.",
                    "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                    "Take the Memory section into consideration when proposing the improvement.",
                    "The solution sketch should be 3-5 sentences.",
                    "Don't suggest to do EDA.",
                ],
            }
            prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _debug(self, parent_node: Node) -> Node:
        if self.is_algotune_task:
            prompt: Any = {
                "Introduction": (
                    "Your previous Solver implementation failed. Fix the correctness or runtime issue and return a corrected Solver class."
                ),
                "Task description": self.task_desc,
                "Previous (buggy) implementation": wrap_code(parent_node.code),
                "Execution output": wrap_code(parent_node.term_out, lang=""),
                "Instructions": {},
            }
            prompt["Instructions"] |= self._prompt_resp_fmt_algotune
            prompt["Instructions"] |= self._prompt_impl_guideline_algotune
        elif self.is_gpu_task:
            # GPU-specific debug prompt
            prompt: Any = {
                "Introduction": (
                    "You are a GPU optimization expert. "
                    "Your previous GPU implementation encountered an error (compilation, runtime, or correctness issue). "
                    "Based on the error information below, fix the issue while maintaining optimization goals. "
                    "Provide a brief explanation followed by the corrected implementation."
                ),
                "Task description": self.task_desc,
                "Previous (buggy) implementation": wrap_code(parent_node.code),
                "Execution output": wrap_code(parent_node.term_out, lang=""),
                "Instructions": {},
            }
            prompt["Instructions"] |= self._prompt_resp_fmt_gpu
            prompt["Instructions"] |= {
                "Bugfix strategy guideline": [
                    "Analyze the error message and identify the root cause (e.g., CUDA syntax, memory limits, tensor shapes).",
                    "Describe how to fix the issue while preserving optimization benefits (3-5 sentences).",
                    "Common GPU errors: shared memory limits, grid/block dimensions, tensor type mismatches.",
                ],
            }
            prompt["Instructions"] |= self._prompt_impl_guideline_gpu
        else:
            # Original ML competition debug prompt
            prompt: Any = {
                "Introduction": (
                    "You are an expert in transformer architectures and attention mechanisms. "
                    "Your previous attention implementation had an issue. Based on the error information provided, "
                    "debug and fix the implementation while ensuring tensor shape compatibility and maintaining "
                    "the expected interface for the nanoGPT training loop. Provide an explanation of the fix "
                    "followed by a single markdown code block with the corrected implementation."
                ),
                "Task description": self.task_desc,
                "Previous (buggy) implementation": wrap_code(parent_node.code),
                "Execution output": wrap_code(parent_node.term_out, lang=""),
                "Instructions": {},
            }
            prompt["Instructions"] |= self._prompt_resp_fmt
            prompt["Instructions"] |= {
                "Bugfix improvement sketch guideline": [
                    "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                    "Don't suggest to do EDA.",
                ],
            }
            prompt["Instructions"] |= self._prompt_impl_guideline

            if self.acfg.data_preview:
                prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        print(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        print(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        if self.is_algotune_task:
            prompt = {
                "Introduction": (
                    "You are evaluating an optimized algorithm implementation. "
                    "Look for the speedup metric, correctness errors, or runtime failures."
                ),
                "Task description": self.task_desc,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
                "Metric instructions": (
                    "Look for 'speedup: X.XXXX' in the output. "
                    "Speedup = Time(Reference) / Time(Solver). "
                    "A speedup below 1.0 is slower than baseline but not a bug by itself. "
                    "Only mark the solution as buggy when there is a syntax error, runtime error, or validation failure."
                ),
            }
        elif self.is_gpu_task:
            # GPU-specific evaluation prompt
            prompt = {
                "Introduction": (
                    "You are a GPU optimization expert evaluating kernel performance. "
                    "You have executed optimized GPU code and need to evaluate the results. "
                    "Look for the speedup metric in the output, check for errors, and report findings."
                ),
                "Task description": self.task_desc,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
                "Metric instructions": (
                    "Look for 'speedup: X.XXXX' in the output. "
                    "Speedup = Time(Original) / Time(Optimized). "
                    "A speedup > 1.0 means faster than baseline, < 1.0 means slower. "
                    "Report the speedup value as the metric. "
                    "IMPORTANT: A speedup < 1.0 is NOT a bug - it just means the optimization is slower. "
                    "Only mark as bug if there's a compilation error, runtime error, or correctness failure."
                ),
            }
        else:
            # Original ML competition evaluation prompt
            prompt = {
                "Introduction": (
                    "You are an expert in transformer architectures and attention mechanisms. "
                    "You have implemented an optimized attention module. Evaluate its performance based on validation loss "
                    "and training stability. Consider whether the optimization achieves meaningful improvements "
                    "while maintaining correctness and compatibility with the model architecture."
                ),
                "Task description": self.task_desc,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }

        try:
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=review_func_spec,
                    model=self.acfg.feedback.model,
                    #temperature=self.acfg.feedback.temp,
                ),
            )
        except Exception as e:
            # If the function call fails (e.g., invalid JSON, wrong schema, etc.)
            # treat it as a bug with 0 speedup
            logger.error(f"Function call failed: {e}")
            response = {
                "is_bug": True,
                "summary": f"Function call failed: {str(e)[:200]}",
                "metric": 0.0 if self.is_speedup_task else None,
                "lower_is_better": False
            }

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response.get("metric"), float):
            response["metric"] = None

        # For GPU tasks, ensure lower_is_better is False (higher speedup is better)
        if self.is_speedup_task and response["metric"] is not None:
            response["lower_is_better"] = False

        node.analysis = response["summary"]

        # For GPU tasks, treat slow performance (speedup < 1.0) as valid but not optimal
        # Only mark as buggy if there's an actual error, not just slow performance
        if self.is_speedup_task:
            node.is_buggy = (
                response["is_bug"]
                or node.exc_type is not None
                or (response["metric"] is None or response["metric"] == 0.0)  # 0.0 indicates compilation/runtime failure
            )
        else:
            node.is_buggy = (
                response["is_bug"]
                or node.exc_type is not None
                or response["metric"] is None
            )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
