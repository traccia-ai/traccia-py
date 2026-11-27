"""Each exception is exported once per occurrence, on every path that records one.

Before the fix, @observe recorded an exception and re-raised it, and the span's context
exit recorded it again (two `exception` events). The adapter also forwarded its own
event to OpenTelemetry and called OTel's record_exception. Instrumentation wrappers
(requests, OpenAI, Anthropic, Gemini, Groq, LangChain, CrewAI) follow the same
"record, then re-raise inside the span's `with` block" pattern, covered here by
``test_wrapper_pattern_records_once``.

Explicit record_exception calls are never de-duplicated: the same exception instance
recorded in two attempts is two events.
"""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import traccia
from traccia.tracer.provider import TracerProvider


@pytest.fixture
def exported():
    """Routes the global provider (used by @observe and trace()) to an in-memory exporter."""
    previous = traccia.get_tracer_provider()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider._otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
    traccia.set_tracer_provider(provider)
    yield provider, exporter
    traccia.set_tracer_provider(previous)


def _exceptions(exporter, name=None):
    spans = [s for s in exporter.get_finished_spans() if name is None or s.name == name]
    assert spans, "no span exported"
    return [e for e in spans[-1].events if e.name == "exception"]


def _attrs(event):
    return dict(event.attributes or {})


def test_observe_sync_records_once(exported):
    _, exporter = exported

    @traccia.observe(name="sync-op")
    def op():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        op()
    [event] = _exceptions(exporter, "sync-op")
    attrs = _attrs(event)
    assert attrs["exception.type"] == "ValueError"
    assert attrs["exception.message"] == "boom"
    assert "exception.stacktrace" in attrs and "exception.escaped" in attrs


def test_observe_async_records_once(exported):
    _, exporter = exported

    @traccia.observe(name="async-op")
    async def op():
        raise KeyError("missing")

    with pytest.raises(KeyError):
        asyncio.run(op())
    assert len(_exceptions(exporter, "async-op")) == 1


def test_context_manager_without_explicit_call_records_once(exported):
    provider, exporter = exported
    tracer = provider.get_tracer("t")
    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("cm"):
            raise RuntimeError("x")
    assert len(_exceptions(exporter, "cm")) == 1


def test_wrapper_pattern_records_once(exported):
    """requests / OpenAI / Anthropic / Gemini / Groq / LangChain / CrewAI wrappers."""
    provider, exporter = exported
    tracer = provider.get_tracer("t")
    with pytest.raises(ConnectionError):
        with tracer.start_as_current_span("wrapped") as span:
            try:
                raise ConnectionError("refused")
            except Exception as exc:
                span.record_exception(exc)
                raise
    assert len(_exceptions(exporter, "wrapped")) == 1


def test_explicit_trace_where_the_caller_records_and_reraises_records_once(exported):
    _, exporter = exported
    with pytest.raises(ValueError):
        with traccia.trace("explicit") as span:
            try:
                raise ValueError("caller")
            except ValueError as exc:
                span.record_exception(exc)
                raise
    assert len(_exceptions(exporter, "explicit")) == 1


def test_explicit_trace_without_a_caller_record_records_once(exported):
    _, exporter = exported
    with pytest.raises(ValueError):
        with traccia.trace("explicit2"):
            raise ValueError("escaping")
    assert len(_exceptions(exporter, "explicit2")) == 1


def test_span_scope_end_records_once_even_if_already_recorded(exported):
    from traccia.context.span_scope import SpanScope

    provider, exporter = exported
    span = provider.get_tracer("t").start_span("scoped")
    scope = SpanScope(span)
    err = ValueError("scoped")
    span.record_exception(err)
    scope.end(err)
    assert len(_exceptions(exporter, "scoped")) == 1


def test_same_instance_recorded_in_two_attempts_is_two_events(exported):
    provider, exporter = exported
    tracer = provider.get_tracer("t")
    err = TimeoutError("flaky")
    with tracer.start_as_current_span("retries") as span:
        for _attempt in range(2):
            try:
                raise err
            except TimeoutError as exc:
                span.record_exception(exc)
    assert len(_exceptions(exporter, "retries")) == 2


def test_two_different_exceptions_are_two_events(exported):
    provider, exporter = exported
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("two") as span:
        span.record_exception(ValueError("a"))
        span.record_exception(KeyError("b"))
    assert [_attrs(e)["exception.type"] for e in _exceptions(exporter, "two")] == ["ValueError", "KeyError"]


def test_documented_edge_recorded_then_reraised_later_unrecorded_is_one_event(exported):
    provider, exporter = exported
    tracer = provider.get_tracer("t")
    err = ValueError("again")
    with pytest.raises(ValueError):
        with tracer.start_as_current_span("edge") as span:
            span.record_exception(err)
            raise err
    assert len(_exceptions(exporter, "edge")) == 1


def test_adapter_path_records_once():
    from traccia.tracer.otel_adapter import TracciaTracerProviderAdapter

    provider = TracciaTracerProviderAdapter()
    exporter = InMemorySpanExporter()
    provider._otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("adapter")
    with pytest.raises(ValueError):
        with tracer.start_as_current_span("adapted") as span:
            try:
                raise ValueError("adapted")
            except ValueError as exc:
                span.record_exception(exc)
                raise
    [event] = _exceptions(exporter, "adapted")
    assert "exception.escaped" in _attrs(event)
    # Traccia's own events list still has it, for processors that read it.
    assert [e["name"] for e in span.events] == ["exception"]


# ── exceptions that are falsey, on both span implementations ──

class FalseyError(Exception):
    """An exception whose truth value is False (e.g. a custom __bool__ or __len__)."""

    def __bool__(self):
        return False


def _adapter():
    from traccia.tracer.otel_adapter import TracciaTracerProviderAdapter

    provider = TracciaTracerProviderAdapter()
    exporter = InMemorySpanExporter()
    provider._otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.mark.parametrize("implementation", ["native", "adapter"])
def test_a_falsey_exception_is_recorded_on_sync_exit(exported, implementation):
    provider, exporter = exported if implementation == "native" else _adapter()
    with pytest.raises(FalseyError):
        with provider.get_tracer("t").start_as_current_span("falsey"):
            raise FalseyError("hidden")
    [event] = _exceptions(exporter, "falsey")
    assert _attrs(event)["exception.type"].endswith("FalseyError")


@pytest.mark.parametrize("implementation", ["native", "adapter"])
def test_a_falsey_exception_is_recorded_on_async_exit(exported, implementation):
    provider, exporter = exported if implementation == "native" else _adapter()
    span = provider.get_tracer("t").start_span("falsey-async")

    async def run():
        async with span:
            raise FalseyError("hidden")

    with pytest.raises(FalseyError):
        asyncio.run(run())
    [event] = _exceptions(exporter, "falsey-async")
    assert _attrs(event)["exception.type"].endswith("FalseyError")


# ── the real call sites, not a copy of their pattern ──

def test_crewai_wrapper_records_once(exported):
    from traccia.integrations.crewai.instrumentation import _create_crew_wrapper

    _, exporter = exported

    def kickoff(self):
        raise RuntimeError("crew failed")

    wrapped = _create_crew_wrapper(kickoff, "crewai.crew.kickoff")

    class Crew:
        id = "crew-1"
        name = "crew"
        agents = []
        tasks = []

    with pytest.raises(RuntimeError):
        wrapped(Crew())
    assert len(_exceptions(exporter, "crewai.crew.kickoff")) == 1


def test_eval_task_failure_records_once(exported, monkeypatch):
    from traccia import auto as traccia_auto
    from traccia.eval import evaluate

    _, exporter = exported
    monkeypatch.setattr(traccia_auto, "_started", True, raising=False)  # keep the test's provider: no init()

    def task(_input):
        raise ValueError("task failed")

    result = evaluate("exc-once", data=[{"input": {"q": 1}}], task=task, persist=False, progress=False)
    assert result is not None
    failed = [s for s in exporter.get_finished_spans() if any(e.name == "exception" for e in s.events)]
    assert failed, "the failing task span was exported"
    for span in failed:
        assert len([e for e in span.events if e.name == "exception"]) == 1, span.name
