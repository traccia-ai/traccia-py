"""OpenAI monkey patching for chat completions and responses API."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Callable
from traccia.tracer.span import SpanStatus

_patched = False
_responses_patched = False


def _compute_cost(model: Optional[str], prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    """Compute cost from token usage using pricing config."""
    if not model or prompt_tokens is None or completion_tokens is None:
        return None
    try:
        from traccia.processors.cost_engine import compute_cost as _compute
        from traccia.pricing_config import load_pricing
        return _compute(model, prompt_tokens, completion_tokens, load_pricing())
    except Exception:
        return None


def _record_llm_metrics(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    duration: Optional[float],
    cost: Optional[float]
):
    """Record LLM metrics if metrics are enabled."""
    try:
        from traccia.metrics.recorder import get_metrics_recorder
        recorder = get_metrics_recorder()
        if not recorder:
            return
        
        # Build attributes
        attributes = {
            "gen_ai.system": "openai",
        }
        if model:
            attributes["gen_ai.request.model"] = model
        
        # Record token usage
        if prompt_tokens is not None or completion_tokens is not None:
            recorder.record_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attributes=attributes
            )
        
        # Record duration
        if duration is not None:
            recorder.record_duration(duration, attributes=attributes)
        
        # Record cost
        if cost is not None:
            recorder.record_cost(cost, attributes=attributes)
    except Exception:
        # Silently fail if metrics recording fails
        pass


def _safe_get(obj, path: str, default=None):
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur if cur is not None else default


def patch_openai() -> bool:
    """Patch OpenAI chat completions for both legacy and new client APIs."""
    global _patched
    if _patched:
        return True
    try:
        import openai
    except Exception:
        return False

    def _extract_messages(kwargs, args):
        messages = kwargs.get("messages")
        # For new client, first arg after self is messages
        if messages is None and len(args) >= 2:
            messages = args[1]
        if not messages or not isinstance(messages, (list, tuple)):
            return None
        # Keep only JSON-friendly, small fields to avoid huge/sensitive payloads.
        slim = []
        for m in list(messages)[:50]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            name = m.get("name")
            content = m.get("content")
            if isinstance(content, (list, dict)):
                content = str(content)
            elif content is not None and not isinstance(content, str):
                content = str(content)
            item = {"role": role, "content": content}
            if name:
                item["name"] = name
            slim.append(item)
        return slim or None

    def _extract_prompt_text(messages_slim) -> Optional[str]:
        if not messages_slim:
            return None
        parts = []
        for m in messages_slim:
            role = m.get("role")
            content = m.get("content")
            if not content:
                continue
            parts.append(f"{role}: {content}" if role else str(content))
        return "\n".join(parts) if parts else None

    def _extract_prompt(kwargs, args) -> Optional[str]:
        messages = kwargs.get("messages")
        if messages is None and len(args) >= 2:
            messages = args[1]
        if not messages:
            return None
        parts = []
        for m in messages:
            content = m.get("content")
            role = m.get("role")
            if content:
                parts.append(f"{role}: {content}" if role else str(content))
        return "\n".join(parts) if parts else None

    def _wrap(create_fn: Callable):
        if getattr(create_fn, "_agent_trace_patched", False):
            return create_fn

        def wrapped_create(*args, **kwargs):
            tracer = _get_tracer("openai")
            model = kwargs.get("model") or _safe_get(args, "0.model", None)
            messages_slim = _extract_messages(kwargs, args)
            prompt_text = _extract_prompt_text(messages_slim) or _extract_prompt(kwargs, args)
            attributes: Dict[str, Any] = {"llm.vendor": "openai"}
            if model:
                attributes["llm.model"] = model
            if messages_slim:
                # Convert messages to JSON string for OTel compatibility
                import json
                try:
                    attributes["llm.openai.messages"] = json.dumps(messages_slim)[:1000]
                except Exception:
                    attributes["llm.openai.messages"] = str(messages_slim)[:1000]
            if prompt_text:
                attributes["llm.prompt"] = prompt_text
            t0 = time.perf_counter()
            with tracer.start_as_current_span("llm.openai.chat.completions", attributes=attributes) as span:
                try:
                    resp = create_fn(*args, **kwargs)
                    # capture model from response if not already set
                    resp_model = getattr(resp, "model", None) or (_safe_get(resp, "model"))
                    if resp_model and "llm.model" not in span.attributes:
                        span.set_attribute("llm.model", resp_model)
                    usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
                    prompt_tokens_val = None
                    completion_tokens_val = None
                    if usage:
                        span.set_attribute("llm.usage.source", "provider_usage")
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            val = getattr(usage, k, None) if not isinstance(usage, dict) else usage.get(k)
                            if val is not None:
                                span.set_attribute(f"llm.usage.{k}", val)
                                if k == "prompt_tokens":
                                    prompt_tokens_val = val
                                elif k == "completion_tokens":
                                    completion_tokens_val = val
                        if "llm.usage.prompt_tokens" in span.attributes:
                            span.set_attribute("llm.usage.prompt_source", "provider_usage")
                        if "llm.usage.completion_tokens" in span.attributes:
                            span.set_attribute("llm.usage.completion_source", "provider_usage")
                    finish_reason = _safe_get(resp, "choices.0.finish_reason")
                    if finish_reason:
                        span.set_attribute("llm.finish_reason", finish_reason)
                    completion = _safe_get(resp, "choices.0.message.content")
                    if completion:
                        span.set_attribute("llm.completion", completion)
                    
                    # Record metrics
                    duration_val = time.perf_counter() - t0
                    cost_val = _compute_cost(
                        resp_model or model,
                        prompt_tokens_val,
                        completion_tokens_val
                    )
                    _record_llm_metrics(
                        model=resp_model or model,
                        prompt_tokens=prompt_tokens_val,
                        completion_tokens=completion_tokens_val,
                        duration=duration_val,
                        cost=cost_val
                    )
                    
                    return resp
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(SpanStatus.ERROR, str(exc))
                    try:
                        from traccia.metrics.recorder import get_metrics_recorder
                        rec = get_metrics_recorder()
                        if rec:
                            rec.record_exception(attributes={"gen_ai.system": "openai", "gen_ai.request.model": model or "unknown"})
                    except Exception:
                        pass
                    raise

        wrapped_create._agent_trace_patched = True
        return wrapped_create

    patched_any = False

    # Legacy: openai.ChatCompletion.create
    target_legacy = getattr(openai, "ChatCompletion", None) or getattr(openai, "chat", None)
    if target_legacy:
        create_fn = getattr(target_legacy, "create", None)
        if create_fn:
            setattr(target_legacy, "create", _wrap(create_fn))
            patched_any = True

    # New client: OpenAI.chat.completions.create
    new_client_cls = getattr(openai, "OpenAI", None)
    if new_client_cls and hasattr(new_client_cls, "chat"):
        chat = getattr(new_client_cls, "chat", None)
        if chat and hasattr(chat, "completions"):
            completions = getattr(chat, "completions")
            if hasattr(completions, "create"):
                patched = _wrap(completions.create)
                setattr(completions, "create", patched)
                patched_any = True

    # New client resource class: openai.resources.chat.completions.Completions
    try:
        from openai.resources.chat.completions import Completions  # type: ignore

        if hasattr(Completions, "create"):
            Completions.create = _wrap(Completions.create)
            patched_any = True
    except Exception:
        pass

    if patched_any:
        _patched = True
    
    # Also patch Responses API (used by OpenAI Agents SDK)
    patch_openai_responses()
    
    return patched_any


def patch_openai_responses() -> bool:
    """Patch OpenAI Responses API for tracing."""
    global _responses_patched
    if _responses_patched:
        return True
    try:
        import openai
    except Exception:
        return False

    def _extract_responses_input(kwargs, args):
        """Extract input from responses.create call."""
        input_data = kwargs.get("input")
        if input_data is None and len(args) >= 2:
            input_data = args[1]
        if not input_data:
            return None, None
        
        # input can be a string or list of ResponseInputItem
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}], input_data
        elif isinstance(input_data, list):
            # Convert to slim representation
            slim = []
            parts = []
            for item in list(input_data)[:50]:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content_items = item.get("content", [])
                    
                    # Extract text from content items
                    text_parts = []
                    if isinstance(content_items, str):
                        text_parts.append(content_items)
                    elif isinstance(content_items, list):
                        for c in content_items:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text_parts.append(c.get("text", ""))
                    
                    content_str = " ".join(text_parts) if text_parts else ""
                    slim.append({"role": role, "content": content_str})
                    if content_str:
                        parts.append(f"{role}: {content_str}")
            
            prompt_text = "\n".join(parts) if parts else None
            return slim or None, prompt_text
        
        return None, None

    def _extract_responses_output(resp) -> Optional[str]:
        """Extract output text from Response object."""
        output = getattr(resp, "output", None) or _safe_get(resp, "output")
        if not output:
            return None
        
        parts = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content", [])
            else:
                content = getattr(item, "content", [])
            
            # Extract text from content items
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text = c.get("text", "")
                        if text:
                            parts.append(text)
                    elif hasattr(c, "type") and c.type == "text":
                        text = getattr(c, "text", "")
                        if text:
                            parts.append(text)
        
        return "\n".join(parts) if parts else None

    def _wrap_responses(create_fn: Callable):
        if getattr(create_fn, "_agent_trace_patched", False):
            return create_fn

        async def wrapped_create(*args, **kwargs):
            tracer = _get_tracer("openai.responses")
            model = kwargs.get("model") or _safe_get(args, "0.model", None)
            input_slim, prompt_text = _extract_responses_input(kwargs, args)
            
            attributes: Dict[str, Any] = {
                "llm.vendor": "openai",
                "llm.api": "responses"
            }
            if model:
                attributes["llm.model"] = model
            if input_slim:
                import json
                try:
                    attributes["llm.openai.input"] = json.dumps(input_slim)[:1000]
                except Exception:
                    attributes["llm.openai.input"] = str(input_slim)[:1000]
            if prompt_text:
                attributes["llm.prompt"] = prompt_text[:2000]
            
            t0 = time.perf_counter()
            with tracer.start_as_current_span("llm.openai.responses", attributes=attributes) as span:
                try:
                    resp = await create_fn(*args, **kwargs)
                    
                    # Extract response details
                    resp_model = getattr(resp, "model", None) or _safe_get(resp, "model")
                    if resp_model and "llm.model" not in span.attributes:
                        span.set_attribute("llm.model", str(resp_model))
                    
                    # Extract usage
                    usage = getattr(resp, "usage", None) or _safe_get(resp, "usage")
                    input_tokens_val = None
                    output_tokens_val = None
                    if usage:
                        span.set_attribute("llm.usage.source", "provider_usage")
                        input_tokens = getattr(usage, "input_tokens", None) or (usage.get("input_tokens") if isinstance(usage, dict) else None)
                        output_tokens = getattr(usage, "output_tokens", None) or (usage.get("output_tokens") if isinstance(usage, dict) else None)
                        total_tokens = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)
                        
                        if input_tokens is not None:
                            span.set_attribute("llm.usage.prompt_tokens", input_tokens)
                            span.set_attribute("llm.usage.input_tokens", input_tokens)
                            span.set_attribute("llm.usage.prompt_source", "provider_usage")
                            input_tokens_val = input_tokens
                        if output_tokens is not None:
                            span.set_attribute("llm.usage.completion_tokens", output_tokens)
                            span.set_attribute("llm.usage.output_tokens", output_tokens)
                            span.set_attribute("llm.usage.completion_source", "provider_usage")
                            output_tokens_val = output_tokens
                        if total_tokens is not None:
                            span.set_attribute("llm.usage.total_tokens", total_tokens)
                    
                    # Extract completion text
                    completion = _extract_responses_output(resp)
                    if completion:
                        span.set_attribute("llm.completion", completion[:2000])
                    
                    # Extract status
                    status = getattr(resp, "status", None) or _safe_get(resp, "status")
                    if status:
                        span.set_attribute("llm.response.status", str(status))
                    
                    # Record metrics
                    duration_val = time.perf_counter() - t0
                    cost_val = _compute_cost(
                        resp_model or model,
                        input_tokens_val,
                        output_tokens_val
                    )
                    _record_llm_metrics(
                        model=resp_model or model,
                        prompt_tokens=input_tokens_val,
                        completion_tokens=output_tokens_val,
                        duration=duration_val,
                        cost=cost_val
                    )
                    
                    return resp
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(SpanStatus.ERROR, str(exc))
                    try:
                        from traccia.metrics.recorder import get_metrics_recorder
                        rec = get_metrics_recorder()
                        if rec:
                            rec.record_exception(attributes={"gen_ai.system": "openai", "gen_ai.request.model": model or "unknown"})
                    except Exception:
                        pass
                    raise

        wrapped_create._agent_trace_patched = True
        return wrapped_create

    patched_any = False

    # Patch AsyncOpenAI.responses.create
    try:
        from openai import AsyncOpenAI
        if hasattr(AsyncOpenAI, "responses"):
            responses = getattr(AsyncOpenAI, "responses")
            if hasattr(responses, "create"):
                # This is a property/descriptor, need to patch the underlying class
                pass
    except Exception:
        pass

    # Patch the Responses resource class directly
    try:
        from openai.resources.responses import AsyncResponses
        if hasattr(AsyncResponses, "create"):
            original_create = AsyncResponses.create
            AsyncResponses.create = _wrap_responses(original_create)
            patched_any = True
    except Exception:
        pass

    if patched_any:
        _responses_patched = True
    return patched_any


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)

