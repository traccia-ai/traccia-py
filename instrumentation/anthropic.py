"""Anthropic monkey patching for messages.create."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from traccia.tracer.span import SpanStatus

_patched = False


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
        
        # Build attributes
        attributes = {
            "gen_ai.system": "anthropic",
        }
        if model:
            attributes["gen_ai.request.model"] = model
        
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


def patch_anthropic() -> bool:
    """Patch Anthropic messages.create; returns True if patched, False otherwise."""
    global _patched
    if _patched:
        return True
    try:
        import anthropic
    except Exception:
        return False

    client_cls = getattr(anthropic, "Anthropic", None)
    if client_cls is None:
        return False

    original = getattr(client_cls, "messages", None)
    if original is None:
        return False

    create_fn = getattr(original, "create", None)
    if create_fn is None:
        return False
    if getattr(create_fn, "_agent_trace_patched", False):
        _patched = True
        return True

    def wrapped_create(self, *args, **kwargs):
        tracer = _get_tracer("anthropic")
        model = kwargs.get("model") or _safe_get(args, "0.model", None)
        attributes: Dict[str, Any] = {"llm.vendor": "anthropic"}
        if model:
            attributes["llm.model"] = model
        t0 = time.perf_counter()
        with tracer.start_as_current_span("llm.anthropic.messages", attributes=attributes) as span:
            try:
                resp = create_fn(self, *args, **kwargs)
                usage = getattr(resp, "usage", None) or resp.get("usage") if isinstance(resp, dict) else None
                input_tokens_val = None
                output_tokens_val = None
                if usage:
                    span.set_attribute("llm.usage.source", "provider_usage")
                    for k in ("input_tokens", "output_tokens"):
                        if k in usage:
                            span.set_attribute(f"llm.usage.{k}", usage[k])
                            if k == "input_tokens":
                                input_tokens_val = usage[k]
                            elif k == "output_tokens":
                                output_tokens_val = usage[k]
                    # Provide OpenAI-style aliases so downstream processors (cost, etc.)
                    # can treat Anthropic uniformly.
                    if "input_tokens" in usage and "llm.usage.prompt_tokens" not in span.attributes:
                        span.set_attribute("llm.usage.prompt_tokens", usage["input_tokens"])
                    if "input_tokens" in usage:
                        span.set_attribute("llm.usage.prompt_source", "provider_usage")
                    if "output_tokens" in usage and "llm.usage.completion_tokens" not in span.attributes:
                        span.set_attribute("llm.usage.completion_tokens", usage["output_tokens"])
                    if "output_tokens" in usage:
                        span.set_attribute("llm.usage.completion_source", "provider_usage")
                stop_reason = _safe_get(resp, "stop_reason") or _safe_get(resp, "stop_reason", None)
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
                
                return resp
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(SpanStatus.ERROR, str(exc))
                try:
                    from traccia.metrics.recorder import get_metrics_recorder
                    rec = get_metrics_recorder()
                    if rec:
                        rec.record_exception(attributes={"gen_ai.system": "anthropic", "gen_ai.request.model": model or "unknown"})
                except Exception:
                    pass
                raise

    wrapped_create._agent_trace_patched = True
    setattr(original, "create", wrapped_create)
    _patched = True
    return True


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)

