from . import backend_anthropic, backend_openai, backend_openrouter, backend_gemini
from .utils import (
    FunctionSpec,
    OutputType,
    PromptType,
    _serialize_trace_value,
    compile_prompt_to_md,
    current_trace_context,
)
import re
import logging
import os

logger = logging.getLogger("aide")


def _verbose_progress(message: str) -> None:
    print(f"[aide.backend] {message}", flush=True)


def determine_provider(model: str) -> str:
    # Check if model matches OpenAI patterns first
    if re.match(r"^(gpt-.*|o\d+(-.*)?|codex-mini-latest)$", model):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("gemini-"):
        return "gemini"
    # If OPENAI_BASE_URL is set, use openai provider for non-standard models
    elif os.getenv("OPENAI_BASE_URL"):
        return "openai"
    # all other models are handle by openrouter
    else:
        return "openrouter"


provider_to_query_func = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "openrouter": backend_openrouter.query,
    "gemini": backend_gemini.query,
}


def _mlflow_module():
    if os.getenv("AIDE_ENABLE_MLFLOW", "0") != "1":
        return None
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _trace_model_provider(provider: str, model: str) -> str:
    if provider != "openai":
        return provider

    base_url = (os.getenv("OPENAI_BASE_URL") or "").lower()
    if "groq.com" in base_url:
        return "groq"
    if "openrouter.ai" in base_url:
        return "openrouter"
    if "googleapis.com" in base_url or "generativelanguage.googleapis.com" in base_url:
        return "gemini"
    return "openai"


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    trace_model_provider = _trace_model_provider(provider, model)
    compiled_system_message = compile_prompt_to_md(system_message) if system_message else None
    compiled_user_message = compile_prompt_to_md(user_message) if user_message else None
    trace_context = current_trace_context()

    mlflow = _mlflow_module()
    span_cm = None
    span = None
    if mlflow is not None:
        span_attributes = {
            "provider": provider,
            "model": model,
            "mlflow.llm.provider": trace_model_provider,
            "mlflow.llm.model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "function_name": func_spec.name if func_spec is not None else None,
        }
        span_attributes.update(trace_context)
        span_cm = mlflow.start_span(
            name=f"aide.llm.{provider}.query",
            span_type="LLM",
            attributes={k: v for k, v in span_attributes.items() if v is not None},
        )
        span = span_cm.__enter__()
        if trace_context:
            mlflow.update_current_trace(
                tags=trace_context,
                metadata={
                    "provider": provider,
                    "model": model,
                    **trace_context,
                },
            )
        span.set_inputs(
            {
                "provider": provider,
                "model": model,
                "trace_context": trace_context,
                "system_message": _serialize_trace_value(compiled_system_message),
                "user_message": _serialize_trace_value(compiled_user_message),
                "function_name": func_spec.name if func_spec is not None else None,
                "model_kwargs": _serialize_trace_value(model_kwargs),
            }
        )

    try:
        _verbose_progress(f"starting model call provider={provider} model={model}")
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=compiled_system_message,
            user_message=compiled_user_message,
            func_spec=func_spec,
            **model_kwargs,
        )
        _verbose_progress(
            f"finished model call provider={provider} model={model} "
            f"in_tokens={in_tok_count} out_tokens={out_tok_count} req_time={req_time:.2f}s"
        )
        if span is not None:
            span.set_outputs(
                {
                    "output": _serialize_trace_value(output),
                    "request_time_sec": req_time,
                    "input_tokens": in_tok_count,
                    "output_tokens": out_tok_count,
                    "info": _serialize_trace_value(info),
                }
            )
            token_usage = {
                "input_tokens": in_tok_count,
                "output_tokens": out_tok_count,
                "total_tokens": in_tok_count + out_tok_count,
            }
            span.set_attribute("mlflow.chat.tokenUsage", token_usage)
            span.set_attribute("mlflow.llm.model", model)
            span.set_attribute("mlflow.llm.provider", trace_model_provider)
            response_attributes = {
                "request_time_sec": req_time,
                "input_tokens": in_tok_count,
                "output_tokens": out_tok_count,
                "response_model": info.get("model") if isinstance(info, dict) else None,
            }
            if isinstance(info, dict) and info.get("cost") is not None:
                response_attributes["mlflow.llm.cost"] = info.get("cost")
            span.set_attributes({k: v for k, v in response_attributes.items() if v is not None})
            span.set_status("OK")
    except Exception as exc:
        _verbose_progress(f"model call failed provider={provider} model={model}: {exc}")
        if span is not None:
            span.record_exception(exc)
            span.set_status("ERROR")
        raise
    finally:
        if span_cm is not None:
            span_cm.__exit__(None, None, None)

    return output
