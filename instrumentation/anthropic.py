"""Anthropic monkey patching for messages.create."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from traccia.tracer.span import SpanStatus

_patched = False


def _compute_cost(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    vendor: str = "anthropic",
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
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    duration: Optional[float],
    cost: Optional[float]
):
    """Record LLM metrics if metrics are enabled."""
    try:
        from traccia.metrics.recorder import get_metrics_recorder
        recorder = get_metrics_recorder()
        if not recorder:
            return
        
        # Build attributes — always include per-run agent identity so the ingestion
        # metrics converter can attribute data points to the correct agent.
        attributes = {
            "gen_ai.system": "anthropic",
        }
        if model:
            attributes["gen_ai.request.model"] = model

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
        if input_tokens is not None or output_tokens is not None:
            recorder.record_token_usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
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


def _settle_now(decision, **kwargs) -> None:
    from traccia.governance.pep import finish_llm_call

    finish_llm_call(decision, **kwargs)


def _settle_in_background(target):
    """A settle function for the async wrapper: settles on a worker thread, so the policy server
    never blocks the event loop, in a copy of the caller's context, with the API key, endpoint
    and agent its policy check used. stop_tracing() waits for it."""

    def settle(decision, **kwargs) -> None:
        from traccia.governance.pep import finish_llm_call_in_background

        finish_llm_call_in_background(decision, target=target, **kwargs)

    return settle


def _record_response(span, resp, model, t0, decision, settle=_settle_now) -> None:
    # The SDK returns a typed Message whose Usage is not subscriptable; read both
    # typed and dict responses through _safe_get.
    usage = _safe_get(resp, "usage")
    input_tokens_val = _safe_get(usage, "input_tokens")
    output_tokens_val = _safe_get(usage, "output_tokens")
    if usage:
        span.set_attribute("llm.usage.source", "provider_usage")
        # Provide OpenAI-style aliases so downstream processors (cost, etc.)
        # can treat Anthropic uniformly.
        for key, alias, value in (
            ("input_tokens", "prompt", input_tokens_val),
            ("output_tokens", "completion", output_tokens_val),
        ):
            if value is None:
                continue
            span.set_attribute(f"llm.usage.{key}", value)
            if f"llm.usage.{alias}_tokens" not in span.attributes:
                span.set_attribute(f"llm.usage.{alias}_tokens", value)
            span.set_attribute(f"llm.usage.{alias}_source", "provider_usage")
    stop_reason = _safe_get(resp, "stop_reason")
    if stop_reason:
        span.set_attribute("llm.stop_reason", stop_reason)

    # Record metrics
    duration_val = time.perf_counter() - t0
    cost_val = _compute_cost(model, input_tokens_val, output_tokens_val)
    _record_llm_metrics(
        model=model,
        input_tokens=input_tokens_val,
        output_tokens=output_tokens_val,
        duration=duration_val,
        cost=cost_val
    )
    settle(decision, actual_usd=cost_val)


def _record_error(span, exc, model, decision, settle=_settle_now) -> None:
    """A call that failed, or was cancelled, before a response: release its reservation."""
    if decision is not None:
        try:
            settle(decision, release=True)
        except Exception:
            pass
    span.record_exception(exc)
    span.set_status(SpanStatus.ERROR, str(exc) or type(exc).__name__)
    try:
        from traccia.metrics.recorder import get_metrics_recorder
        rec = get_metrics_recorder()
        if rec:
            rec.record_exception(attributes={"gen_ai.system": "anthropic", "gen_ai.request.model": model or "unknown"})
    except Exception:
        pass


def _span_attributes(model) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {"llm.vendor": "anthropic"}
    if model:
        attributes["llm.model"] = model
    return attributes


def _wrap_sync(create_fn):
    def wrapped_create(self, *args, **kwargs):
        model = kwargs.get("model")
        t0 = time.perf_counter()
        with _get_tracer("anthropic").start_as_current_span("llm.anthropic.messages", attributes=_span_attributes(model)) as span:
            decision = None
            try:
                from traccia.governance.pep import enforce_llm_call

                decision = enforce_llm_call(kwargs, vendor="anthropic")
                resp = create_fn(self, *args, **kwargs)
                _record_response(span, resp, model, t0, decision)
                return resp
            except BaseException as exc:
                # BaseException too (KeyboardInterrupt), so the reservation is never left held.
                _record_error(span, exc, model, decision)
                raise

    wrapped_create._agent_trace_patched = True
    return wrapped_create


def _wrap_async(create_fn):
    async def wrapped_create(self, *args, **kwargs):
        model = kwargs.get("model")
        t0 = time.perf_counter()
        with _get_tracer("anthropic").start_as_current_span("llm.anthropic.messages", attributes=_span_attributes(model)) as span:
            decision = None
            settle = _settle_now
            try:
                from traccia.governance.pep import enforce_llm_call_off_loop

                # On a worker thread, so a slow policy server delays no other task. A call
                # cancelled during the check releases what the check reserves.
                decision, target = await enforce_llm_call_off_loop(kwargs, vendor="anthropic")
                settle = _settle_in_background(target)
                resp = await create_fn(self, *args, **kwargs)
                _record_response(span, resp, model, t0, decision, settle)
                return resp
            except BaseException as exc:
                # BaseException too: a cancelled call (asyncio.CancelledError, a wait_for()
                # timeout) releases its reservation instead of holding it until it expires.
                _record_error(span, exc, model, decision, settle)
                raise

    wrapped_create._agent_trace_patched = True
    return wrapped_create


def patch_anthropic() -> bool:
    """Patch Messages.create and AsyncMessages.create; returns True if patched.

    Client.messages is a cached_property, so patch the resource classes instead.
    """
    global _patched
    if _patched:
        return True
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except Exception:
        return False

    for cls, wrap in ((Messages, _wrap_sync), (AsyncMessages, _wrap_async)):
        create_fn = getattr(cls, "create", None)
        if create_fn is None:
            return False
        if not getattr(create_fn, "_agent_trace_patched", False):
            setattr(cls, "create", wrap(create_fn))
    _patched = True
    return True


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)

