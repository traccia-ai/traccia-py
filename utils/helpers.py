"""Helper functions for OpenTelemetry compatibility."""

from __future__ import annotations

from typing import Optional

from opentelemetry.trace import Span as OTelSpan


def get_duration_ns(span: OTelSpan) -> Optional[int]:
    """
    Get span duration in nanoseconds.
    
    Args:
        span: OpenTelemetry Span instance
    
    Returns:
        Duration in nanoseconds, or None if span hasn't ended
    """
    if span.end_time is None:
        return None
    return span.end_time - span.start_time


def get_parent_span_id(span: OTelSpan) -> Optional[int]:
    """
    Get parent span ID from span context.
    
    Args:
        span: OpenTelemetry Span instance
    
    Returns:
        Parent span ID (int64), or None if no parent
    """
    # OTel doesn't expose parent_span_id directly
    # We need to get it from the span's parent context if available
    # For now, return None - parent relationship is maintained via context
    # This can be enhanced if needed by tracking parent during span creation
    return None


def format_trace_id(trace_id: int) -> str:
    """
    Format OTel trace_id (int64) to hex string.
    
    Args:
        trace_id: OTel trace_id as int64
    
    Returns:
        32-character hex string
    """
    return format(trace_id, '032x')


def format_span_id(span_id: int) -> str:
    """
    Format OTel span_id (int64) to hex string.
    
    Args:
        span_id: OTel span_id as int64
    
    Returns:
        16-character hex string
    """
    return format(span_id, '016x')


def parse_trace_id(hex_string: str) -> int:
    """
    Parse hex string trace_id to OTel int64.
    
    Args:
        hex_string: 32-character hex string
    
    Returns:
        OTel trace_id as int64
    """
    if not hex_string:
        return 0
    return int(hex_string, 16)


def parse_span_id(hex_string: str) -> int:
    """
    Parse hex string span_id to OTel int64.
    
    Args:
        hex_string: 16-character hex string
    
    Returns:
        OTel span_id as int64
    """
    if not hex_string:
        return 0
    return int(hex_string, 16)


def detach_context_token(token, attached_context=None) -> bool:
    """Undo an ``opentelemetry.context.attach()``, but only where that is safe.

    A token can be reset only in the contextvars Context it was created in: from another thread or
    an asyncio task (which runs in a copy) OTel's ``detach()`` fails and logs an error. When
    ``attached_context`` is given, the token is also left alone unless that context is still the
    current one, so a child span activated on top of it is not dropped.

    Returns True if the token was detached here, False if it was left for its own context.
    """
    import contextvars
    from opentelemetry import context

    if attached_context is not None and context.get_current() is not attached_context:
        return False
    if isinstance(token, contextvars.Token):
        try:
            token.var.reset(token)
            return True
        except (ValueError, RuntimeError):
            # Created in another Context, or already reset.
            return False
    context.detach(token)
    return True


_LEAKED = "_traccia_leaked_restore_ctx"
_MAX_SKIPS = 64


def _effective(ctx=None):
    """Resolve the current span, skipping spans that ended but stayed current because Traccia
    could not undo their activation in this context (see ``deactivate_span()``).

    Returns (span, context, skipped): span is the usable current span or None; context is the
    context it lives in (the one to resume); skipped is True if any span was skipped.
    """
    from opentelemetry import context as otel_context
    from opentelemetry.trace import get_current_span

    ctx = otel_context.get_current() if ctx is None else ctx
    span = get_current_span(ctx)
    skipped = False
    for _ in range(_MAX_SKIPS):
        previous = getattr(span, _LEAKED, None)
        if previous is None or span.is_recording():
            break
        ctx, span, skipped = previous, get_current_span(previous), True
    else:
        # Not expected: activations record an already-resolved context, so chains stay one step
        # long. Never hand back an ended, leaked span.
        if getattr(span, _LEAKED, None) is not None and not span.is_recording():
            return None, otel_context.Context(), True
    if span is None or not span.get_span_context().is_valid:
        return None, ctx, skipped
    return span, ctx, skipped


def activate_span(otel_span):
    """Make ``otel_span`` the current span. Returns the activation, for ``deactivate_span()``."""
    from opentelemetry import context
    from opentelemetry.trace import set_span_in_context

    # The context to resume if this activation leaks, already resolved past leaked spans, so
    # repeated leaks (an async chain invoked again and again in one coroutine) never chain up.
    _, previous, _ = _effective()
    ctx = set_span_in_context(otel_span)
    return (context.attach(ctx), ctx, previous, otel_span)


def deactivate_span(activation) -> bool:
    """Undo ``activate_span()``. Where that cannot be done (another thread or task, or a child span
    is current), the span stays current in its context; it is then marked so that it is skipped
    once it has ended and the context that was current before it is resumed (``_effective()``).
    Returns True if it was detached here."""
    token, ctx, previous, otel_span = activation
    if detach_context_token(token, ctx):
        return True
    try:
        setattr(otel_span, _LEAKED, previous)
    except Exception:
        pass
    return False


def effective_current_span():
    """The current OTel span, or None, skipping ended spans whose activation leaked."""
    return _effective()[0]


def current_span_and_context():
    """The current OTel span and the context to parent new spans on (see ``_effective()``).
    Returns (span, context): span is None when there is no usable parent, and context is then an
    empty Context if spans were skipped (start a new trace), else None (no parent at all)."""
    from opentelemetry import context as otel_context
    from opentelemetry.trace import set_span_in_context

    span, _, skipped = _effective()
    if span is not None:
        return span, set_span_in_context(span)
    return None, (otel_context.Context() if skipped else None)
