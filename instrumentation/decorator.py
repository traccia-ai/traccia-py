"""@observe decorator for instrumenting functions."""

from __future__ import annotations

import functools
import inspect
import traceback
from typing import Any, Callable, Dict, Iterable, Optional
from traccia.tracer.span import SpanStatus


def _capture_args(bound_args: inspect.BoundArguments, skip: Iterable[str]) -> Dict[str, Any]:
    """Capture function arguments, converting complex types to OTel-compatible types."""
    captured = {}
    for name, value in bound_args.arguments.items():
        if name in skip:
            continue
        # Skip 'self' - it's an object, not a valid OTel attribute
        if name == "self":
            continue
        # Convert value to OTel-compatible type
        captured[name] = _convert_to_otel_type(value)
    return captured


def _convert_to_otel_type(value: Any) -> Any:
    """
    Convert a value to an OpenTelemetry-compatible type.
    
    OTel attributes must be: bool, str, bytes, int, float, or sequences of those.
    """
    # Primitive types are fine
    if isinstance(value, (bool, str, bytes, int, float)) or value is None:
        return value
    
    # For sequences, convert each element
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            if isinstance(item, (bool, str, bytes, int, float)) or item is None:
                converted.append(item)
            else:
                # Convert complex types to string representation
                converted.append(str(item)[:1000])  # Truncate long strings
        return converted[:100]  # Limit sequence length
    
    # For dicts and other complex types, convert to JSON string
    if isinstance(value, dict):
        try:
            import json
            json_str = json.dumps(value, default=str)[:1000]  # Truncate
            return json_str
        except Exception:
            return str(value)[:1000]
    
    # For other types, convert to string
    return str(value)[:1000]  # Truncate long strings


def _infer_type_from_attributes(attributes: Dict[str, Any]) -> Optional[str]:
    """
    Infer span type from attributes.
    
    Returns:
        - "llm" if LLM-related attributes found
        - "tool" if tool-related attributes found
        - None otherwise (will use default "span")
    """
    # Check for LLM indicators
    if any(key in attributes for key in ["llm.model", "llm.vendor", "model"]):
        return "llm"
    
    # Check for tool indicators
    if any(key in attributes for key in ["tool.name", "tool", "http.url"]):
        return "tool"
    
    return None


def _extract_llm_attributes(span_attrs: Dict[str, Any], bound_args: inspect.BoundArguments) -> None:
    """
    Extract LLM-related attributes from function arguments.
    
    Extracts common LLM parameters like model, temperature, max_tokens, messages.
    Fails silently if extraction fails or attributes not found.
    
    Args:
        span_attrs: Dictionary to add extracted attributes to
        bound_args: Bound arguments from the function call
    """
    try:
        args_dict = dict(bound_args.arguments)
        
        # Extract model
        if "model" in args_dict and "llm.model" not in span_attrs:
            span_attrs["llm.model"] = str(args_dict["model"])
        
        # Extract temperature
        if "temperature" in args_dict and "llm.temperature" not in span_attrs:
            temp = args_dict["temperature"]
            if isinstance(temp, (int, float)):
                span_attrs["llm.temperature"] = temp
        
        # Extract max_tokens
        if "max_tokens" in args_dict and "llm.max_tokens" not in span_attrs:
            max_tok = args_dict["max_tokens"]
            if isinstance(max_tok, int):
                span_attrs["llm.max_tokens"] = max_tok
        
        # Extract messages/prompt
        if "messages" in args_dict and "llm.prompt" not in span_attrs:
            messages = args_dict["messages"]
            if isinstance(messages, (list, str)):
                # Convert messages to string representation
                prompt_str = _convert_to_otel_type(messages)
                span_attrs["llm.prompt"] = prompt_str
        elif "prompt" in args_dict and "llm.prompt" not in span_attrs:
            prompt = args_dict["prompt"]
            if isinstance(prompt, str):
                span_attrs["llm.prompt"] = prompt[:1000]
    
    except Exception:
        # Fail silently - don't interrupt span creation if extraction fails
        pass


def _validate_guardrail_span(span_attrs: Dict[str, Any]) -> None:
    """Log warnings for guardrail-typed spans missing recommended attributes."""
    try:
        from traccia.guardrails.helpers import validate_guardrail_attributes
        import logging
        # triggered may be set after the function returns (bool auto-trigger)
        warnings = validate_guardrail_attributes(span_attrs, require_triggered=False)
        if warnings:
            _logger = logging.getLogger("traccia.guardrails")
            for w in warnings:
                _logger.warning(w)
    except Exception:
        pass


def observe(
    name: Optional[str] = None,
    *,
    attributes: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
    as_type: str = "span",
    skip_args: Optional[Iterable[str]] = None,
    skip_result: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate a function to create a span around its execution.

    - Supports sync and async functions.
    - Captures errors and records exception events.
    - Optionally captures arguments/results (skip controls).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or func.__name__
        arg_names = func.__code__.co_varnames
        skip_args_set = set(skip_args or [])
        tags_list = [str(tag) for tag in tags] if tags is not None else []

        is_coro = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = _get_tracer(func.__module__ or "default")
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()

            span_attrs = dict(attributes or {})
            if tags_list:
                span_attrs["span.tags"] = tags_list
            
            # Capture function arguments first
            span_attrs.update(_capture_args(bound, skip_args_set))
            
            # Infer type from attributes if not explicitly set (or if set to default "span")
            inferred_type = as_type
            if as_type == "span":
                # Try to infer from attributes
                detected_type = _infer_type_from_attributes(span_attrs)
                if detected_type:
                    inferred_type = detected_type
            
            # Set span type
            span_attrs["span.type"] = inferred_type
            
            # Extract LLM attributes if this is an LLM call
            if inferred_type == "llm":
                _extract_llm_attributes(span_attrs, bound)

            # Validate guardrail-typed spans for completeness
            if inferred_type == "guardrail":
                _validate_guardrail_span(span_attrs)

            with tracer.start_as_current_span(span_name, attributes=span_attrs) as span:
                try:
                    result = func(*args, **kwargs)
                    # For guardrail-typed spans: auto-set triggered from bool return value
                    # so developers don't need to manually call get_current_span().
                    # Only applies when triggered was not pre-set in attributes={}.
                    if inferred_type == "guardrail" and isinstance(result, bool):
                        if span.attributes.get("guardrail.triggered") is None:
                            span.set_attribute("guardrail.triggered", result)
                    if not skip_result:
                        # Convert result to OTel-compatible type
                        otel_result = _convert_to_otel_type(result)
                        span.set_attribute("result", otel_result)
                    return result
                except Exception as exc:
                    # Record detailed error information
                    span.record_exception(exc)
                    span.set_status(SpanStatus.ERROR, str(exc))
                    
                    # Add error attributes
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_attribute("error.message", str(exc))
                    
                    # Add truncated stack trace
                    tb = traceback.format_exc()
                    span.set_attribute("error.stack_trace", tb[:2000])  # Truncate to 2000 chars
                    
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = _get_tracer(func.__module__ or "default")
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()

            span_attrs = dict(attributes or {})
            if tags_list:
                span_attrs["span.tags"] = tags_list
            
            # Capture function arguments first
            span_attrs.update(_capture_args(bound, skip_args_set))
            
            # Infer type from attributes if not explicitly set (or if set to default "span")
            inferred_type = as_type
            if as_type == "span":
                # Try to infer from attributes
                detected_type = _infer_type_from_attributes(span_attrs)
                if detected_type:
                    inferred_type = detected_type
            
            # Set span type
            span_attrs["span.type"] = inferred_type
            
            # Extract LLM attributes if this is an LLM call
            if inferred_type == "llm":
                _extract_llm_attributes(span_attrs, bound)

            # Validate guardrail-typed spans for completeness
            if inferred_type == "guardrail":
                _validate_guardrail_span(span_attrs)

            async with tracer.start_as_current_span(span_name, attributes=span_attrs) as span:
                try:
                    result = await func(*args, **kwargs)
                    # For guardrail-typed spans: auto-set triggered from bool return value.
                    if inferred_type == "guardrail" and isinstance(result, bool):
                        if span.attributes.get("guardrail.triggered") is None:
                            span.set_attribute("guardrail.triggered", result)
                    if not skip_result:
                        # Convert result to OTel-compatible type
                        otel_result = _convert_to_otel_type(result)
                        span.set_attribute("result", otel_result)
                    return result
                except Exception as exc:
                    # Record detailed error information
                    span.record_exception(exc)
                    span.set_status(SpanStatus.ERROR, str(exc))
                    
                    # Add error attributes
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_attribute("error.message", str(exc))
                    
                    # Add truncated stack trace
                    tb = traceback.format_exc()
                    span.set_attribute("error.stack_trace", tb[:2000])  # Truncate to 2000 chars
                    
                    raise

        return async_wrapper if is_coro else sync_wrapper

    return decorator


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)

