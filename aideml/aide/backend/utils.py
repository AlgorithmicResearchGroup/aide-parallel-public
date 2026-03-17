import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


logger = logging.getLogger("aide")

TRACE_MAX_DEPTH = int(os.getenv("AIDE_TRACE_MAX_DEPTH", "4"))
TRACE_MAX_ITEMS = int(os.getenv("AIDE_TRACE_MAX_ITEMS", "20"))
TRACE_MAX_STRING_CHARS = int(os.getenv("AIDE_TRACE_MAX_STRING_CHARS", "2000"))
BACKOFF_MAX_TIME_SECONDS = float(os.getenv("AIDE_BACKOFF_MAX_TIME_SECONDS", "180"))
BACKOFF_MAX_TRIES = int(os.getenv("AIDE_BACKOFF_MAX_TRIES", "8"))

TRACE_CONTEXT_ENV_MAP = {
    "task_type": "AIDE_TRACE_TASK_TYPE",
    "task_id": "AIDE_TRACE_TASK_ID",
    "experiment_idx": "AIDE_TRACE_EXPERIMENT_IDX",
    "iteration": "AIDE_TRACE_ITERATION",
}


def _mlflow_module():
    if os.getenv("AIDE_ENABLE_MLFLOW", "0") != "1":
        return None
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _truncate_trace_string(value: str) -> str:
    if len(value) <= TRACE_MAX_STRING_CHARS:
        return value
    clipped = len(value) - TRACE_MAX_STRING_CHARS
    return f"{value[:TRACE_MAX_STRING_CHARS]}\n... [{clipped} characters truncated]"


def _serialize_trace_value(value: Any, depth: int = 0) -> Any:
    if depth > TRACE_MAX_DEPTH:
        return _truncate_trace_string(repr(value))
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _truncate_trace_string(value)
    if isinstance(value, bytes):
        try:
            return _truncate_trace_string(value.decode("utf-8"))
        except UnicodeDecodeError:
            return _truncate_trace_string(value.hex())
    if isinstance(value, dict):
        items = list(value.items())
        serialized = {
            str(key): _serialize_trace_value(item, depth + 1)
            for key, item in items[:TRACE_MAX_ITEMS]
        }
        if len(items) > TRACE_MAX_ITEMS:
            serialized["__truncated_items__"] = len(items) - TRACE_MAX_ITEMS
        return serialized
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        serialized = [_serialize_trace_value(item, depth + 1) for item in items[:TRACE_MAX_ITEMS]]
        if len(items) > TRACE_MAX_ITEMS:
            serialized.append(f"... [{len(items) - TRACE_MAX_ITEMS} items truncated]")
        return serialized
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _serialize_trace_value(method(), depth + 1)
            except Exception:
                pass
    return _truncate_trace_string(repr(value))


def current_trace_context() -> dict[str, str]:
    context: dict[str, str] = {}
    for key, env_name in TRACE_CONTEXT_ENV_MAP.items():
        value = os.getenv(env_name)
        if value:
            context[key] = value
    return context


def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    mlflow = _mlflow_module()
    trace_context = current_trace_context()
    span_name = getattr(create_fn, "__qualname__", getattr(create_fn, "__name__", "llm_call"))
    retry_types = tuple(retry_exceptions)
    total_wait_seconds = 0.0

    for attempt in range(1, BACKOFF_MAX_TRIES + 1):
        span_cm = None
        span = None
        if mlflow is not None:
            attributes = {
                "callable.module": getattr(create_fn, "__module__", ""),
                "callable.name": span_name,
                "retry_exception_types": ",".join(exc.__name__ for exc in retry_exceptions),
                "attempt": attempt,
            }
            attributes.update(trace_context)
            span_cm = mlflow.start_span(name=span_name, span_type="LLM", attributes=attributes)
            span = span_cm.__enter__()
            span.set_inputs(
                {
                    "trace_context": trace_context,
                    "args": _serialize_trace_value(args),
                    "kwargs": _serialize_trace_value(kwargs),
                }
            )
        try:
            response = create_fn(*args, **kwargs)
            if span is not None:
                span.set_outputs(_serialize_trace_value(response))
                span.set_status("OK")
            return response
        except retry_types as exc:
            if span is not None:
                span.record_exception(exc)
                span.set_status("ERROR")
                span.set_outputs({"retry_exception": _serialize_trace_value(str(exc))})
            wait_seconds = min(60.0, 1.5 * (2 ** (attempt - 1)))
            if (
                attempt >= BACKOFF_MAX_TRIES
                or total_wait_seconds + wait_seconds > BACKOFF_MAX_TIME_SECONDS
            ):
                raise RuntimeError(
                    f"{span_name} failed after {attempt} attempts: {exc}"
                ) from exc
            total_wait_seconds += wait_seconds
            logger.info(
                "Backing off %s for %.1fs (%s)",
                span_name,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)
        except Exception as exc:
            if span is not None:
                span.record_exception(exc)
                span.set_status("ERROR")
            raise
        finally:
            if span_cm is not None:
                span_cm.__exit__(None, None, None)


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }

    @property
    def as_openai_responses_tool_dict(self):
        """Convert to OpenAI Responses API tool format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_responses_tool_choice_dict(self):
        """Convert to OpenAI Responses API tool choice format."""
        return {
            "type": "function",
            "name": self.name,
        }
