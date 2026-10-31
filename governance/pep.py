"""SDK policy enforcement point: check() on LLM and tool calls under @govern."""

from __future__ import annotations

import atexit
import contextvars
import logging
import os
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Optional, Set

from traccia import client_config
from traccia.governance.policy import AgentBlockedError, ApprovalPending, _derive_base_url, _http_session
from traccia import runtime_config

logger = logging.getLogger("traccia.governance")

CHECK_PATH = "/api/v1/policy/check"
SETTLE_PATH = "/api/v1/policy/settle"
_BUDGET_CACHE_TTL_S = 5.0
_budget_lock = threading.Lock()
_remaining_budget: Dict[str, tuple[float, float]] = {}
_read_lock = threading.Lock()
_last_reads: Dict[str, Dict[str, str]] = {}
_retrieval_by_trace: Dict[str, Dict[str, Any]] = {}
_prompt_by_trace: Dict[str, Dict[str, str]] = {}
_PROMPT_KEYS = ("traccia.prompt.name", "traccia.prompt.label", "traccia.prompt.version_id")


def _enrich_blocked(exc: AgentBlockedError, decision: Dict[str, Any]) -> AgentBlockedError:
    exc.decision_id = decision.get("id")
    exc.remaining_budget_usd = decision.get("remaining_budget_usd")
    exc.reasons = decision.get("reasons") or []
    return exc


def _credentials() -> tuple[Optional[str], Optional[str]]:
    return client_config.resolve_credentials()


def _agent_id() -> Optional[str]:
    return runtime_config.get_agent_id() or os.environ.get("TRACCIA_AGENT_ID")


def _as_otel_hex(value: Any, width: int) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return format(value, f"0{width}x")
    text = str(value).strip()
    return text or None


def _trace_ids() -> tuple[Optional[str], Optional[str]]:
    try:
        from traccia.context import get_current_span

        span = get_current_span()
        if span and getattr(span, "context", None):
            return (
                _as_otel_hex(getattr(span.context, "trace_id", None), 32),
                _as_otel_hex(getattr(span.context, "span_id", None), 16),
            )
    except Exception:
        pass
    return None, None


def _stamp_span(decision: Dict[str, Any]) -> None:
    try:
        from traccia.context import get_current_span

        span = get_current_span()
        if not span:
            return
        if decision.get("id"):
            span.set_attribute("traccia.policy.decision_id", str(decision["id"]))
        if decision.get("effect"):
            span.set_attribute("traccia.policy.effect", str(decision["effect"]))
        span.set_attribute("traccia.policy.would_have", bool(decision.get("would_have")))
        activation = decision.get("activation")
        if activation in ("observe", "warn", "block"):
            span.set_attribute("traccia.policy.activation", str(activation))
        ids = decision.get("policy_ids") or []
        if ids:
            span.set_attribute("traccia.policy.ids", ",".join(str(i) for i in ids))
        reasons = decision.get("reasons") or []
        if reasons:
            span.set_attribute("traccia.policy.reason", str(reasons[0])[:500])
    except Exception:
        logger.debug("Could not stamp policy attributes on span", exc_info=True)


def _cache_budget(agent_id: str, remaining: Optional[float]) -> None:
    if remaining is None:
        return
    with _budget_lock:
        _remaining_budget[agent_id] = (time.time(), float(remaining))


def cached_remaining_budget(agent_id: str) -> Optional[float]:
    with _budget_lock:
        entry = _remaining_budget.get(agent_id)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > _BUDGET_CACHE_TTL_S:
            del _remaining_budget[agent_id]
            return None
        return value


def _blocked_message(decision: Dict[str, Any]) -> str:
    reasons = decision.get("reasons") or []
    reason = reasons[0] if reasons else "policy denied this call"
    remaining = decision.get("remaining_budget_usd")
    decision_id = decision.get("id") or ""
    parts = [reason]
    if remaining is not None:
        parts.append(f"remaining budget ${remaining:.2f}")
    if decision_id:
        parts.append(f"decision {decision_id}")
    return ". ".join(parts)


def remember_tool_result(name: str, result: Any) -> None:
    """Keep the last as_of or updated_at from a tool result in this process."""
    if not name or not isinstance(result, dict):
        return
    for field in ("as_of", "updated_at"):
        value = result.get(field)
        if value not in (None, ""):
            with _read_lock:
                _last_reads[str(name)] = {"tool": str(name), "read_at": str(value), "field": field}
            return


def note_retrieval_attributes(trace_id: Optional[str], attributes: Optional[Dict[str, Any]]) -> None:
    """Record retrieval evidence only when a span already set it."""
    if not trace_id or not attributes:
        return
    chunk = attributes.get("traccia.retrieval.chunk_count")
    present_flag = attributes.get("traccia.retrieval.present")
    if chunk is None and present_flag is None:
        return
    with _read_lock:
        entry = _retrieval_by_trace.setdefault(str(trace_id), {})
        if present_flag is True or chunk is not None:
            entry["present"] = True
        if chunk is not None:
            entry["chunk_count"] = chunk


def _freshness_context() -> list:
    with _read_lock:
        return [dict(item) for item in _last_reads.values()]


def _retrieval_context(trace_id: Optional[str]) -> Dict[str, Any]:
    if not trace_id:
        return {"present": False}
    with _read_lock:
        found = _retrieval_by_trace.get(str(trace_id))
    if not found:
        return {"present": False}
    out: Dict[str, Any] = {"present": bool(found.get("present"))}
    if "chunk_count" in found:
        out["chunk_count"] = found["chunk_count"]
    return out


def note_prompt_attributes(trace_id: Optional[str], attributes: Optional[Dict[str, Any]]) -> None:
    """Remember a compiled Traccia prompt for later LLM checks in this trace.

    compile() stamps the parent span. The provider wrapper then starts a child
    span, and that child is current when the check runs.
    """
    if not attributes:
        return
    name = attributes.get("traccia.prompt.name")
    if not name:
        return
    key = _as_otel_hex(trace_id, 32)
    if not key:
        return
    entry = {"name": str(name)}
    label = attributes.get("traccia.prompt.label")
    version_id = attributes.get("traccia.prompt.version_id")
    if label:
        entry["label"] = str(label)
    if version_id:
        entry["version_id"] = str(version_id)
    with _read_lock:
        _prompt_by_trace[key] = entry


def _prompt_from_attributes(attributes: Any) -> Dict[str, Any]:
    if not isinstance(attributes, dict):
        attributes = getattr(attributes, "_dict", None) if attributes is not None else None
    if not isinstance(attributes, dict):
        try:
            attributes = dict(attributes or {})
        except Exception:
            return {}
    prompt: Dict[str, Any] = {}
    name = attributes.get("traccia.prompt.name")
    label = attributes.get("traccia.prompt.label")
    version_id = attributes.get("traccia.prompt.version_id")
    if name:
        prompt["name"] = name
    if label:
        prompt["label"] = label
    if version_id:
        prompt["version_id"] = version_id
    return prompt


def _prompt_context() -> Dict[str, Any]:
    try:
        from traccia.context import get_current_span

        span = get_current_span()
    except Exception:
        span = None
    from_span = _prompt_from_attributes(getattr(span, "attributes", None) if span else None)
    if from_span.get("name"):
        return from_span
    trace_id, _span_id = _trace_ids()
    if not trace_id:
        return from_span
    with _read_lock:
        remembered = dict(_prompt_by_trace.get(trace_id) or {})
    return remembered or from_span


def _customer_id(arguments: Dict[str, Any]) -> Optional[str]:
    customer = arguments.get("customer")
    if isinstance(customer, dict) and customer.get("id") not in (None, ""):
        return str(customer.get("id"))
    return None


def _unresolved(reason: str, detail: str, *, warn: bool = True) -> Dict[str, Any]:
    """No decision from Traccia. Allow, unless the @govern run is fail-closed."""
    if not runtime_config.pep_fail_open():
        logger.error("policy check could not complete (%s); blocking call (fail_open=False)", detail)
        raise AgentBlockedError(
            f"Policy check could not complete ({detail}). Blocking because fail_open is False.",
            reasons=[reason],
        )
    if warn:
        logger.warning("policy check could not complete (%s); allowing call", detail)
    return {"effect": "allow", "would_have": False, "reasons": [reason]}


def check_policy(
    *,
    action: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    resource: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST /api/v1/policy/check. Safe to call from custom tools; @govern uses this automatically."""
    api_key, endpoint = _credentials()
    if not api_key or not endpoint:
        return _unresolved("missing_credentials", "Traccia API key or endpoint not set", warn=False)
    agent_id = _agent_id()
    if not agent_id:
        return _unresolved(
            "missing_agent_id", "no agent_id on init, @govern, or TRACCIA_AGENT_ID", warn=False
        )
    trace_id, span_id = _trace_ids()
    payload = {
        "principal": {"agent_id": agent_id},
        "action": action,
        "resource": resource or {},
        "context": dict(context or {}),
        "agent_id": agent_id,
        "trace_id": trace_id,
        "span_id": str(span_id) if span_id is not None else None,
    }
    payload["context"].setdefault("agent_id", agent_id)
    remaining = cached_remaining_budget(agent_id)
    if remaining is not None:
        payload["context"].setdefault("remaining_budget_usd", remaining)
    try:
        base = _derive_base_url(endpoint)
        response = _http_session.post(
            f"{base}{CHECK_PATH}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=2,
        )
        if response.status_code != 200:
            return _unresolved(
                "check_http_error",
                f"HTTP {response.status_code}: {(response.text or '')[:180]}",
            )
        decision = response.json()
    except AgentBlockedError:
        raise
    except Exception as exc:
        return _unresolved("check_error", str(exc))
    _cache_budget(agent_id, decision.get("remaining_budget_usd"))
    _stamp_span(decision)
    if decision.get("would_have"):
        return decision
    effect = decision.get("effect") or "allow"
    if effect == "queue":
        raise ApprovalPending(
            "This action is waiting for approval. Do not run the tool or retry the check.",
            approval_id=decision.get("approval_id"),
            expires_at=decision.get("approval_expires_at"),
            decision_id=decision.get("id"),
        )
    if effect == "deny":
        raise _enrich_blocked(AgentBlockedError(_blocked_message(decision)), decision)
    return decision


def reserved_usd(decision: Optional[Dict[str, Any]]) -> Optional[float]:
    """The USD amount the policy check reserved for a call, if any."""
    reserved = ((decision or {}).get("obligations") or {}).get("reserved_usd")
    return reserved if isinstance(reserved, (int, float)) and not isinstance(reserved, bool) else None


def settle_policy(
    decision: Optional[Dict[str, Any]],
    *,
    release: bool = False,
    actual_usd: Optional[float] = None,
    model: Optional[str] = None,
    vendor: Optional[str] = None,
    input_tokens: Optional[float] = None,
    output_tokens: Optional[float] = None,
) -> None:
    """Settle the call's reservation.

    Without ``actual_usd`` the server prices the call from ``model``, ``vendor`` and the
    token counts; with none of them it releases the reservation.
    """
    if not decision:
        return
    reserved = (decision.get("obligations") or {}).get("reserved_usd")
    if reserved is None:
        return
    api_key, endpoint = _credentials()
    agent_id = _agent_id()
    if not api_key or not endpoint or not agent_id:
        return
    trace_id, _span_id = _trace_ids()
    try:
        base = _derive_base_url(endpoint)
        _http_session.post(
            f"{base}{SETTLE_PATH}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "agent_id": agent_id,
                "trace_id": trace_id,
                "reserved_usd": reserved,
                "actual_usd": actual_usd,
                "release": release,
                "model": model,
                "vendor": vendor,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            timeout=2,
        )
    except Exception as exc:
        logger.debug("policy settle failed: %s", exc)


# Tokens one image is counted as in a prompt estimate, in the range vision models charge for a
# typical image. Image parts are counted by this, never by their (base64) URL.
IMAGE_PROMPT_TOKENS = 1000
_IMAGE_PART_TYPES = {"image_url", "image", "input_image"}


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _message_text(message: Any) -> tuple[str, int]:
    """(text, image count) of one message: its content (a string or content parts) plus any
    tool call arguments, which are part of the prompt when the conversation is sent back."""
    if isinstance(message, str):
        return message, 0
    content = _get(message, "content")
    texts = []
    images = 0
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue
            text = _get(part, "text")
            if isinstance(text, str):
                texts.append(text)
            elif _get(part, "type") in _IMAGE_PART_TYPES:
                images += 1
    tool_calls = _get(message, "tool_calls")
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            arguments = _get(_get(call, "function"), "arguments")
            if isinstance(arguments, str):
                texts.append(arguments)
    return "".join(texts), images


def _estimate_tokens(kwargs: Dict[str, Any]) -> Optional[int]:
    try:
        from traccia.processors.token_counter import estimate_tokens_from_text_with_source
    except Exception:
        estimate_tokens_from_text_with_source = None
    messages = kwargs.get("messages")
    text = kwargs.get("prompt") or kwargs.get("input")
    blob = ""
    images = 0
    if isinstance(messages, (list, tuple)):
        parts = []
        for item in messages:
            message_text, message_images = _message_text(item)
            parts.append(message_text)
            images += message_images
        blob = "\n".join(parts)
    elif isinstance(text, str):
        blob = text
    image_tokens = images * IMAGE_PROMPT_TOKENS
    if not blob:
        return image_tokens or None
    rough = max(len(blob) // 4, 1)
    if estimate_tokens_from_text_with_source:
        try:
            count, source = estimate_tokens_from_text_with_source(blob)
            if source == "estimated.tiktoken":
                return int(count) + image_tokens
            # Without tiktoken the counter splits on whitespace, which counts code, JSON or
            # unspaced text as a handful of tokens; never estimate below characters / 4.
            return max(int(count), rough) + image_tokens
        except Exception:
            pass
    return rough + image_tokens


def estimate_prompt_tokens(kwargs: Dict[str, Any]) -> Optional[int]:
    """Rough prompt size in tokens, as sent with the policy check."""
    return _estimate_tokens(kwargs)


def enforce_llm_call(
    kwargs: Dict[str, Any], vendor: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Check policy before an LLM call. *vendor* is the span's llm.vendor; Spend Cap
    uses it to pick that provider's price."""
    if not runtime_config.pep_enabled():
        return None
    model = kwargs.get("model")
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    trace_id, _span_id = _trace_ids()
    context: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "input_tokens": _estimate_tokens(kwargs),
        "retrieval": _retrieval_context(trace_id),
    }
    if vendor:
        context["vendor"] = vendor
    prompt = _prompt_context()
    if prompt:
        context["prompt"] = prompt
    decision = check_policy(
        action={"type": "llm_call", "name": "llm_call", "model": model},
        context=context,
    )
    if (decision.get("effect") == "reshape") and not decision.get("would_have"):
        obligations = decision.get("obligations") or {}
        cheaper = obligations.get("cheaper_model") or obligations.get("fallback_model")
        if cheaper:
            kwargs["model"] = cheaper
        if obligations.get("clamp_max_tokens") is not None:
            kwargs["max_tokens"] = obligations["clamp_max_tokens"]
    return decision


def finish_llm_call(
    decision: Optional[Dict[str, Any]],
    *,
    release: bool = False,
    actual_usd: Optional[float] = None,
    model: Optional[str] = None,
    vendor: Optional[str] = None,
    input_tokens: Optional[float] = None,
    output_tokens: Optional[float] = None,
) -> None:
    settle_policy(
        decision,
        release=release,
        actual_usd=actual_usd,
        model=model,
        vendor=vendor,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


_settle_executor: Optional[ThreadPoolExecutor] = None
_pending_settles: Set[Future] = set()
# Reentrant: a garbage-collected stream settles from its finalizer, which can run on a thread
# already inside this lock.
_settle_lock = threading.RLock()


def _settle_pool() -> ThreadPoolExecutor:
    global _settle_executor
    with _settle_lock:
        if _settle_executor is None:
            _settle_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="traccia-settle")
            atexit.register(flush_pending_settles)
        return _settle_executor


def _settle_done(future: Future) -> None:
    with _settle_lock:
        _pending_settles.discard(future)


def finish_llm_call_in_background(decision: Optional[Dict[str, Any]], **kwargs: Any) -> None:
    """finish_llm_call() on a worker thread, so a slow policy server does not delay the caller.

    Runs in a copy of the current context, so it settles for the agent (and span) that made
    the call. flush_pending_settles() waits for it; it also runs at interpreter exit.
    """
    if reserved_usd(decision) is None:
        return  # nothing was reserved, so there is nothing to settle
    ctx = contextvars.copy_context()
    try:
        future = _settle_pool().submit(ctx.run, finish_llm_call, decision, **kwargs)
    except RuntimeError:
        # The interpreter is shutting down (e.g. a stream finalized at exit): settle here.
        ctx.run(finish_llm_call, decision, **kwargs)
        return
    with _settle_lock:
        _pending_settles.add(future)
    future.add_done_callback(_settle_done)


def flush_pending_settles(timeout: Optional[float] = 10.0) -> None:
    """Wait (up to ``timeout`` seconds) for every background settle started so far."""
    with _settle_lock:
        pending = list(_pending_settles)
    if pending:
        wait(pending, timeout=timeout)


# LLM calls still in flight: an open stream, or a request that has not returned. Each has
# settle_at_shutdown(). Weak, so a call that is dropped and collected is not kept alive here.
_open_calls: "weakref.WeakSet[Any]" = weakref.WeakSet()
_open_calls_hook = False


def track_open_call(call: Any) -> None:
    """Register a call in flight; settle_open_calls() settles it if it is still open then."""
    global _open_calls_hook
    with _settle_lock:
        _open_calls.add(call)
        if not _open_calls_hook:
            # Registered after start_tracing()'s exit hook, so it runs first: spans it ends
            # are still exported.
            atexit.register(settle_open_calls)
            _open_calls_hook = True


def untrack_open_call(call: Any) -> None:
    with _settle_lock:
        _open_calls.discard(call)


def settle_open_calls(timeout: float = 10.0) -> None:
    """Settle the calls still in flight, then wait for every pending settle, all within
    ``timeout`` seconds. Each call is stopped where possible and charged what it reported,
    or its reservation when that is unknown. Run by stop_tracing() and at interpreter exit.
    """
    deadline = time.monotonic() + timeout
    with _settle_lock:
        calls = list(_open_calls)
    for done, call in enumerate(calls):
        if time.monotonic() >= deadline:
            logger.warning(
                "%d open LLM call(s) not settled before the shutdown time limit", len(calls) - done
            )
            break
        try:
            call.settle_at_shutdown()
        except Exception as exc:
            logger.debug("settling an open call at shutdown failed: %s", exc)
    flush_pending_settles(timeout=max(deadline - time.monotonic(), 0.0))


def enforce_tool_call(name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if not runtime_config.pep_enabled():
        return None
    payload = arguments or {}
    context: Dict[str, Any] = {"input": payload, "tool_name": name}
    freshness = _freshness_context()
    if freshness:
        context["freshness"] = freshness
    customer_id = _customer_id(payload)
    if customer_id:
        context["customer_id"] = customer_id
    return check_policy(
        action={"type": "tool_call", "name": name},
        context=context,
    )
