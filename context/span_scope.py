"""Manual span scope for long-lived spans (e.g. streaming). Mirrors Node spanScope()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, TypeVar

from traccia.context.run_with_span import run_with_span, run_with_span_async

if TYPE_CHECKING:
    from traccia.tracer.span import Span

T = TypeVar("T")


class SpanScope:
    """
    Explicit span lifecycle control.

    Unlike ``span()`` / ``start_as_current_span``, the span is not ended
    automatically — call ``scope.end()`` when the operation completes.
    Use ``scope.run`` / ``scope.run_async`` to run code with this span active.
    """

    def __init__(self, span: "Span") -> None:
        self.span = span
        self._ended = False

    def end(self, error: Optional[BaseException] = None) -> None:
        """End the span, optionally recording an exception."""
        if self._ended:
            return
        self._ended = True
        if error is not None:
            self.span.record_exception(error)
        self.span.end()

    def run(self, fn: Callable[[], T]) -> T:
        """Run *fn* with this scope's span as the active parent."""
        return run_with_span(self.span, fn)

    async def run_async(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Run *fn* (async) with this scope's span as the active parent."""
        return await run_with_span_async(self.span, fn)


def span_scope(
    name: str,
    *,
    attributes: Optional[Dict[str, Any]] = None,
    parent: Optional[Any] = None,
    parent_context: Optional[Any] = None,
    tracer_name: str = "traccia",
) -> SpanScope:
    """
    Start a span with explicit lifecycle control.

    Mirrors Node ``spanScope()`` for long-lived operations such as streaming
    chat turns where the span must stay open across multiple async callbacks.

    Example::

        async def _handler():
            ...

        turn = span_scope("chat.turn", attributes={"span.type": "span"})
        await turn.run_async(_handler)
        turn.end()
    """
    from traccia import get_tracer

    tracer = get_tracer(tracer_name)
    span = tracer.start_span(
        name,
        attributes=attributes,
        parent=parent,
        parent_context=parent_context,
    )
    return SpanScope(span)
