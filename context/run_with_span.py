"""Run code with an explicit span as the active parent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar

from traccia.context.context import pop_span, push_span

if TYPE_CHECKING:
    from traccia.tracer.span import Span

T = TypeVar("T")


def run_with_span(span: "Span", fn: Callable[[], T]) -> T:
    """Run *fn* with *span* set as the active span (sync)."""
    token = push_span(span)
    try:
        return fn()
    finally:
        pop_span(token)


async def run_with_span_async(span: "Span", fn: Callable[[], Awaitable[T]]) -> T:
    """Run *fn* with *span* set as the active span (async)."""
    token = push_span(span)
    try:
        return await fn()
    finally:
        pop_span(token)
