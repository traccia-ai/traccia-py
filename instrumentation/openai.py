"""OpenAI monkey patching for chat completions and responses API."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Callable
from traccia.tracer.span import SpanStatus

_patched = False
_responses_patched = False


def _compute_cost(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    vendor: str = "openai",
) -> Optional[float]:
    """Cost of one call, priced like its span: with the vendor it sets as llm.vendor, from the
    shared resolver's table (which includes pricing overrides)."""
    if not model or prompt_tokens is None or completion_tokens is None:
        return None
    try:
        from traccia.cost_resolver import get_resolver
        from traccia.processors.cost_engine import compute_cost as _compute
        return _compute(model, prompt_tokens, completion_tokens, get_resolver().pricing_table, vendor=vendor)
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
        
        # Build attributes — always include the per-run agent identity so the
        # ingestion metrics converter can attribute the data point to the correct
        # agent rather than falling back to the resource-level service.name.
        attributes: Dict[str, Any] = {
            "gen_ai.system": "openai",
        }
        if model:
            attributes["gen_ai.request.model"] = model
        
        # Attach per-run agent identity from runtime_config (set by run_identity context manager).
        try:
            from traccia import runtime_config as _rc
            _aid = _rc.get_agent_id()
            _aname = _rc.get_agent_name()
            _env = _rc.get_env()
            if _aid:
                attributes["agent.id"] = _aid
                attributes["agent_id"] = _aid
            if _aname:
                attributes["agent.name"] = _aname
            if _env:
                attributes["environment"] = _env
        except Exception:
            pass
        
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


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)):
            if not part.isdecimal():
                return default
            try:
                cur = cur[int(part)]
            except (IndexError, ValueError):
                return default
        else:
            cur = getattr(cur, part, None)
    return cur if cur is not None else default


def _field(obj: Any, name: str) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` for an SDK model; None when absent."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# Responses output items that hand control back to the caller, the counterpart of a Chat
# Completions ``finish_reason`` of "tool_calls": tools the caller runs (function, custom,
# computer use, local shell, shell, apply_patch) and an MCP call awaiting the caller's
# approval. Hosted tool items (web_search_call, file_search_call, code_interpreter_call,
# image_generation_call, mcp_call, ...) run on OpenAI's side and do not end the turn.
_RESPONSES_CLIENT_TOOL_ITEMS = frozenset(
    {
        "function_call",
        "custom_tool_call",
        "computer_call",
        "local_shell_call",
        "shell_call",
        "apply_patch_call",
        "mcp_approval_request",
    }
)

# ``incomplete_details.reason`` mapped to the Chat Completions finish_reason vocabulary,
# so ``llm.finish_reason`` means the same thing for both OpenAI APIs.
_RESPONSES_INCOMPLETE_REASONS = {"max_output_tokens": "length", "content_filter": "content_filter"}


def _responses_completion(resp: Any) -> Optional[str]:
    """The text a Responses API response produced, or None when it produced none.

    Text lives two levels down: ``output`` is a list of items, and each ``message`` item
    holds ``content`` parts of type ``output_text``. They are joined the way the SDK's own
    ``Response.output_text`` joins them. A response that only refused yields the refusal
    text, so a refusal is still visible on the span.
    """
    output = _field(resp, "output")
    if not isinstance(output, (list, tuple)):
        return None
    texts = []
    refusals = []
    for item in output:
        content = _field(item, "content")
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            part_type = _field(part, "type")
            # "text" is accepted for older or hand-built payloads.
            if part_type in ("output_text", "text"):
                text = _field(part, "text")
                if isinstance(text, str):
                    texts.append(text)
            elif part_type == "refusal":
                refusal = _field(part, "refusal")
                if isinstance(refusal, str):
                    refusals.append(refusal)
    completion = "".join(texts) or "".join(refusals)
    return completion or None


def _responses_finish_reason(resp: Any) -> Optional[str]:
    """A Chat-style finish reason for a Responses API response.

    The Responses API has no finish_reason; it reports ``status`` and, when incomplete,
    ``incomplete_details.reason``. A completed response is "tool_calls" when it ends by
    asking the caller to run a tool and "stop" otherwise. An incomplete one maps its
    reason ("max_output_tokens" -> "length"; unknown reasons pass through). Other statuses
    (failed, cancelled, in progress) have no finish reason; ``llm.response.status`` records them.
    """
    status = _field(resp, "status")
    if status == "completed":
        output = _field(resp, "output")
        if isinstance(output, (list, tuple)) and any(
            _field(item, "type") in _RESPONSES_CLIENT_TOOL_ITEMS for item in output
        ):
            return "tool_calls"
        return "stop"
    if status == "incomplete":
        reason = _field(_field(resp, "incomplete_details"), "reason")
        if reason:
            return _RESPONSES_INCOMPLETE_REASONS.get(str(reason), str(reason))
    return None


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
                decision = None
                try:
                    from traccia.governance.pep import enforce_llm_call, finish_llm_call
                    from traccia.governance.policy import AgentBlockedError

                    decision = enforce_llm_call(kwargs, vendor=attributes.get("llm.vendor"))
                    if kwargs.get("model") and kwargs.get("model") != model:
                        span.set_attribute("llm.model", kwargs["model"])
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
                    finish_llm_call(decision, actual_usd=cost_val)
                    return resp
                except Exception as exc:
                    try:
                        from traccia.governance.pep import finish_llm_call as _finish
                        from traccia.governance.policy import AgentBlockedError as _Blocked
                        _finish(decision, release=True)
                    except Exception:
                        pass
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
                decision = None
                target = None
                try:
                    from traccia.governance.pep import (
                        enforce_llm_call_off_loop,
                        finish_llm_call_in_background,
                    )

                    # Off the event loop, so a slow policy server delays no other task.
                    decision, target = await enforce_llm_call_off_loop(
                        kwargs, vendor=attributes.get("llm.vendor"), api=attributes.get("llm.api")
                    )
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
                    completion = _responses_completion(resp)
                    if completion:
                        span.set_attribute("llm.completion", completion[:2000])
                    finish_reason = _responses_finish_reason(resp)
                    if finish_reason:
                        span.set_attribute("llm.finish_reason", finish_reason)

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
                    finish_llm_call_in_background(decision, target=target, actual_usd=cost_val)
                    return resp
                except BaseException as exc:
                    # BaseException too: a cancelled call (asyncio.CancelledError, a wait_for()
                    # timeout) releases its reservation instead of holding it until it expires.
                    try:
                        from traccia.governance.pep import finish_llm_call_in_background as _finish
                        _finish(decision, target=target, release=True)
                    except Exception:
                        pass
                    span.record_exception(exc)
                    span.set_status(SpanStatus.ERROR, str(exc) or type(exc).__name__)
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

