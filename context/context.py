"""Context helpers for managing the active span stack - using OpenTelemetry directly."""

from contextvars import Token
from typing import Optional, TYPE_CHECKING

from opentelemetry.trace import get_current_span as otel_get_current_span
from opentelemetry.trace import set_span_in_context
from opentelemetry import context as context_api

if TYPE_CHECKING:
    from traccia.tracer.span import Span


def get_current_span() -> Optional["Span"]:
    """
    Return the currently active span, if any.
    
    Uses OpenTelemetry's context API internally.
    """
    otel_span = otel_get_current_span()
    if otel_span and otel_span.get_span_context().is_valid:
        # Unsampled spans keep their attributes only on the original wrapper
        existing = getattr(otel_span, '_traccia_span', None)
        if existing is not None:
            return existing
        # Get tracer from span if available
        try:
            if hasattr(otel_span, '_traccia_tracer'):
                tracer = otel_span._traccia_tracer
            else:
                # Fallback: create a tracer from global provider
                from traccia import get_tracer_provider
                provider = get_tracer_provider()
                tracer = provider.get_tracer("context")
            
            # Wrap in Traccia Span
            from traccia.tracer.span import Span
            return Span(otel_span, tracer)
        except Exception:
            # If wrapping fails, return None
            return None
    return None


def push_span(span: "Span") -> Token:
    """
    Push a span onto the context and set it as current.
    
    Uses OpenTelemetry's context API internally.
    
    Returns:
        Token needed to restore the previous state
    """
    if hasattr(span, '_otel_span'):
        otel_span = span._otel_span
    else:
        otel_span = span
    
    ctx = set_span_in_context(otel_span)
    token = context_api.attach(ctx)
    return token


def pop_span(token: Token) -> None:
    """
    Restore the previous span context using the provided token.
    
    Args:
        token: Token returned by push_span()
    """
    context_api.detach(token)
