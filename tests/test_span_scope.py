"""Tests for span_scope and run_with_span (aligned with traccia-node)."""

from __future__ import annotations

import asyncio
import unittest

from traccia import get_tracer, set_tracer_provider, span_scope
from traccia.context import get_current_span, run_with_span, run_with_span_async
from traccia.tracer import TracerProvider
from traccia.tracer.span import SpanStatus


class TestSpanScope(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = TracerProvider()
        set_tracer_provider(self.provider)
        self.tracer = get_tracer("test")

    def test_run_async_links_child_spans(self) -> None:
        turn = span_scope("chat.turn", attributes={"span.type": "span"})

        async def work() -> None:
            async def inner() -> None:
                self.assertEqual(
                    get_current_span().context.span_id,
                    turn.span.context.span_id,
                )
                llm = self.tracer.start_span(
                    "llm.inference",
                    attributes={"span.type": "LLM"},
                )
                self.assertEqual(llm.context.trace_id, turn.span.context.trace_id)
                self.assertEqual(llm.parent_span_id, turn.span.context.span_id)
                llm.end()

            await turn.run_async(inner)

        asyncio.run(work())
        turn.end()

    def test_end_records_exception(self) -> None:
        scope = span_scope("chat.turn")
        scope.end(RuntimeError("boom"))
        self.assertEqual(scope.span.status, SpanStatus.ERROR)

    def test_run_with_span_sync(self) -> None:
        parent = self.tracer.start_span("chat.turn")
        seen = {}

        def inner() -> None:
            current = get_current_span()
            seen["span_id"] = current.context.span_id if current else None

        run_with_span(parent, inner)
        self.assertEqual(seen["span_id"], parent.context.span_id)
        parent.end()

    def test_run_with_span_async(self) -> None:
        parent = self.tracer.start_span("chat.turn")

        async def work() -> None:
            async def inner() -> None:
                current = get_current_span()
                self.assertEqual(current.context.span_id, parent.context.span_id)

            await run_with_span_async(parent, inner)

        asyncio.run(work())
        parent.end()

    def test_nested_start_as_current_span(self) -> None:
        with self.tracer.start_as_current_span("chat.turn") as turn:
            child = self.tracer.start_span("llm.inference")
            self.assertEqual(child.parent_span_id, turn.context.span_id)
            child.end()


if __name__ == "__main__":
    unittest.main()
