"""Head sampling controls exports (TRC-276).

sample_rate used to be stored but never applied, so rate 0 still exported every
span. These tests cover the sampler, the provider wiring, parent-based
inheritance and each export path.
"""

from __future__ import annotations

import logging
from collections import Counter

import pytest
from opentelemetry.sdk.trace import SpanProcessor as OTelSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON, ParentBased, TraceIdRatioBased

import traccia
from traccia import auto, get_tracer, get_tracer_provider, set_tracer_provider
from traccia.processors import sampler as sampler_module
from traccia.processors.batch_processor import BatchSpanProcessor
from traccia.processors.logging_processor import LoggingSpanProcessor
from traccia.processors.sampler import Sampler
from traccia.tracer.provider import TracerProvider
from traccia.tracer.span_context import SpanContext

CHILDREN_PER_TRACE = 3
GRANDCHILDREN_PER_CHILD = 1
SPANS_PER_TRACE = 1 + CHILDREN_PER_TRACE * (1 + GRANDCHILDREN_PER_CHILD)


class CollectingExporter:
    """Traccia-style exporter that keeps every exported span."""

    def __init__(self) -> None:
        self.spans = []

    def export(self, spans) -> bool:
        self.spans.extend(spans)
        return True

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout=None) -> bool:
        return True


def _make_traces(tracer, count: int) -> None:
    """Create `count` traces of root -> children -> grandchildren via the current-span context."""
    for _ in range(count):
        with tracer.start_as_current_span("root"):
            for _ in range(CHILDREN_PER_TRACE):
                with tracer.start_as_current_span("child"):
                    for _ in range(GRANDCHILDREN_PER_CHILD):
                        with tracer.start_as_current_span("grandchild"):
                            pass


@pytest.fixture
def provider_with_memory():
    """Fresh global provider whose OTel export path writes to memory."""
    previous = get_tracer_provider()
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    set_tracer_provider(provider)
    yield provider, memory
    set_tracer_provider(previous)


@pytest.fixture
def stop_tracing_after():
    previous = get_tracer_provider()
    set_tracer_provider(TracerProvider())
    yield
    auto.stop_tracing()
    set_tracer_provider(previous)


# --- Sampler unit ------------------------------------------------------------


@pytest.mark.parametrize("rate", [-0.01, 1.01, 2.0])
def test_sampler_rejects_out_of_range_rates(rate):
    with pytest.raises(ValueError):
        Sampler(rate)


def test_rate_zero_never_samples_even_when_random_returns_zero(monkeypatch):
    monkeypatch.setattr(sampler_module.random, "random", lambda: 0.0)
    assert Sampler(0.0).should_sample().sampled is False


def test_rate_one_always_samples_at_top_of_random_range(monkeypatch):
    monkeypatch.setattr(sampler_module.random, "random", lambda: 0.9999999999)
    assert Sampler(1.0).should_sample().sampled is True


def test_to_otel_sampler_is_parent_based_ratio():
    otel = Sampler(0.25).to_otel_sampler()
    assert isinstance(otel, ParentBased)
    assert "TraceIdRatioBased{0.25}" in otel.get_description()


# --- Provider wiring ---------------------------------------------------------


def test_set_sampler_applies_to_otel_provider():
    provider = TracerProvider()
    sampler = Sampler(0.0)
    provider.set_sampler(sampler)
    assert provider.get_sampler() is sampler
    assert "TraceIdRatioBased{0.0}" in provider._otel_provider.sampler.get_description()


def test_set_sampler_accepts_otel_sampler_as_is():
    provider = TracerProvider()
    provider.set_sampler(ALWAYS_OFF)
    assert provider._sampler_switch.delegate is ALWAYS_OFF


def test_set_sampler_none_restores_otel_default():
    provider = TracerProvider()
    default = provider._sampler_switch.delegate
    provider.set_sampler(Sampler(0.0))
    provider.set_sampler(None)
    assert provider._sampler_switch.delegate is default
    assert provider.get_sampler() is None


def test_set_sampler_rejects_unknown_types():
    with pytest.raises(TypeError):
        TracerProvider().set_sampler(object())


# --- Export counts through the OTel export path ------------------------------


def test_rate_zero_exports_nothing(provider_with_memory):
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(0.0))
    _make_traces(get_tracer("t"), 50)
    assert memory.get_finished_spans() == ()


def test_rate_zero_spans_are_non_recording_and_unsampled(provider_with_memory):
    provider, _ = provider_with_memory
    provider.set_sampler(Sampler(0.0))
    tracer = get_tracer("t")
    with tracer.start_as_current_span("root") as root:
        with tracer.start_as_current_span("child") as child:
            pass
    for span in (root, child):
        assert span._otel_span.is_recording() is False
        assert span.context.trace_flags == 0
    # Unsampled spans still belong to one trace, so context propagation keeps working.
    assert child.context.trace_id == root.context.trace_id


def test_rate_one_exports_every_span(provider_with_memory):
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(1.0))
    _make_traces(get_tracer("t"), 50)
    assert len(memory.get_finished_spans()) == 50 * SPANS_PER_TRACE


def test_intermediate_rate_keeps_whole_traces_only(provider_with_memory):
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(0.5))
    traces = 2000
    _make_traces(get_tracer("t"), traces)

    spans = memory.get_finished_spans()
    per_trace = Counter(s.context.trace_id for s in spans)
    # Each kept trace is complete: no orphaned children, no rootless traces.
    assert set(per_trace.values()) == {SPANS_PER_TRACE}
    roots = [s for s in spans if s.parent is None]
    assert len(roots) == len(per_trace)
    # Binomial(2000, 0.5) has sd ~22; 0.42..0.58 is a ~7 sd window.
    assert 0.42 < len(per_trace) / traces < 0.58


@pytest.mark.parametrize("rate", [0.1, 0.9])
def test_rate_is_respected_at_other_intermediate_values(provider_with_memory, rate):
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(rate))
    traces = 2000
    _make_traces(get_tracer("t"), traces)
    kept = len({s.context.trace_id for s in memory.get_finished_spans()})
    assert abs(kept / traces - rate) < 0.06


def test_sampling_decision_is_deterministic_per_trace_id(provider_with_memory):
    """TraceIdRatioBased decides from the trace ID, so a kept trace ID is always kept."""
    otel = Sampler(0.5).to_otel_sampler()
    for trace_id in (1, 2**64, 2**127 + 12345, 2**128 - 1):
        first = otel.should_sample(None, trace_id, "root").decision
        for _ in range(5):
            assert otel.should_sample(None, trace_id, "root").decision == first


# --- Parent-based inheritance ------------------------------------------------


def test_explicit_parent_argument_inherits_unsampled_decision(provider_with_memory):
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(0.0))
    tracer = get_tracer("t")
    root = tracer.start_span("root")
    child = tracer.start_span("child", parent=root)
    child.end()
    root.end()
    assert child.context.trace_flags == 0
    assert memory.get_finished_spans() == ()


def test_children_of_sampled_root_are_kept_when_sampler_changes_mid_trace(provider_with_memory):
    """A child follows its parent, not the root ratio, so traces never split."""
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(1.0))
    tracer = get_tracer("t")
    with tracer.start_as_current_span("root"):
        provider.set_sampler(Sampler(0.0))
        with tracer.start_as_current_span("child"):
            pass
    names = sorted(s.name for s in memory.get_finished_spans())
    assert names == ["child", "root"]


@pytest.mark.parametrize("rate, flags, exported", [(1.0, 0, 0), (0.0, 1, 1)])
def test_remote_parent_flags_decide(provider_with_memory, rate, flags, exported):
    """Incoming W3C context wins over the local ratio, per the parent-based contract."""
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(rate))
    remote = SpanContext(
        trace_id="0af7651916cd43dd8448eb211c80319c",
        span_id="b7ad6b7169203331",
        trace_flags=flags,
    )
    span = get_tracer("t").start_span("server", parent_context=remote)
    span.end()
    finished = memory.get_finished_spans()
    assert len(finished) == exported
    assert span.context.trace_id == "0af7651916cd43dd8448eb211c80319c"


# --- Tracers created before the sampler or provider changed ------------------


def test_tracer_created_before_set_sampler_uses_the_new_sampler(provider_with_memory):
    provider, memory = provider_with_memory
    tracer = get_tracer("early")
    _make_traces(tracer, 1)
    assert len(memory.get_finished_spans()) == SPANS_PER_TRACE

    memory.clear()
    provider.set_sampler(Sampler(0.0))
    _make_traces(tracer, 10)
    assert memory.get_finished_spans() == ()


def test_tracer_follows_replaced_otel_provider():
    from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider

    provider = TracerProvider()
    tracer = provider.get_tracer("early")
    tracer.start_span("warmup").end()

    replacement = OTelTracerProvider(sampler=ALWAYS_ON)
    memory = InMemorySpanExporter()
    replacement.add_span_processor(SimpleSpanProcessor(memory))
    provider._otel_provider = replacement

    tracer.start_span("after").end()
    assert [s.name for s in memory.get_finished_spans()] == ["after"]


def test_tracer_reuses_otel_tracer_across_sampler_changes():
    provider = TracerProvider()
    tracer = provider.get_tracer("t")
    first = tracer._otel_tracer
    provider.set_sampler(Sampler(0.5))
    assert tracer._otel_tracer is first


def test_otel_tracer_cached_by_otel_sdk_sees_sampler_changes():
    """The OTel SDK caches tracers per scope with the sampler they were built with."""
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    raw = provider._otel_provider.get_tracer("raw-otel")
    provider.set_sampler(Sampler(0.0))
    raw.start_span("direct-otel").end()
    assert memory.get_finished_spans() == ()


# --- Traccia processors on unsampled spans -----------------------------------


def test_batch_processor_drops_unsampled_spans():
    provider = TracerProvider()
    exporter = CollectingExporter()
    batch = BatchSpanProcessor(exporter=exporter, sampler=Sampler(0.0), schedule_delay_millis=60_000)
    provider.add_span_processor(batch)
    provider.set_sampler(Sampler(0.0))
    tracer = provider.get_tracer("t")
    for _ in range(20):
        tracer.start_span("s").end()
    batch.shutdown()
    assert exporter.spans == []


def test_logging_processor_skips_unsampled_spans(caplog):
    provider = TracerProvider()
    provider.add_span_processor(LoggingSpanProcessor())
    tracer = provider.get_tracer("t")

    provider.set_sampler(Sampler(0.0))
    with caplog.at_level(logging.INFO, logger="traccia.traces"):
        tracer.start_span("dropped").end()
    assert "dropped" not in caplog.text

    provider.set_sampler(Sampler(1.0))
    with caplog.at_level(logging.INFO, logger="traccia.traces"):
        tracer.start_span("kept").end()
    assert "name=kept" in caplog.text


def test_enrichment_still_runs_on_unsampled_spans():
    """In-process consumers (eval cost rollups) read enriched attributes even when unsampled."""
    seen = []

    class Recorder:
        def on_end(self, span):
            seen.append(span.name)

        def shutdown(self):
            return None

        def force_flush(self, timeout=None):
            return None

    provider = TracerProvider()
    provider.add_span_processor(Recorder())
    provider.set_sampler(Sampler(0.0))
    span = provider.get_tracer("t").start_span("unsampled")
    span.end()
    assert seen == ["unsampled"]
    assert span.name == "unsampled"


def test_otel_export_processor_never_sees_unsampled_spans():
    ended = []

    class Spy(OTelSpanProcessor):
        def on_end(self, span):
            ended.append(span.name)

    provider = TracerProvider()
    provider.add_span_processor(Spy())
    provider.set_sampler(Sampler(0.0))
    provider.get_tracer("t").start_span("unsampled").end()
    provider.set_sampler(Sampler(1.0))
    provider.get_tracer("t").start_span("sampled").end()
    assert ended == ["sampled"]


# --- start_tracing end to end ------------------------------------------------


def _start(exporter, rate, **kwargs):
    return auto.start_tracing(
        exporter=exporter,
        sample_rate=rate,
        enable_costs=False,
        enable_metrics=False,
        enable_patching=False,
        load_env=False,
        **kwargs,
    )


def test_start_tracing_rate_zero_exports_nothing(stop_tracing_after):
    exporter = CollectingExporter()
    _start(exporter, 0.0)
    _make_traces(get_tracer("e2e"), 100)
    auto.force_flush()
    assert exporter.spans == []


def test_start_tracing_rate_one_exports_every_trace(stop_tracing_after):
    exporter = CollectingExporter()
    _start(exporter, 1.0)
    _make_traces(get_tracer("e2e"), 100)
    auto.force_flush()
    # Count traces, not spans: the custom batch processor has a separate
    # end-vs-flush race that can rarely drop a single span.
    assert len({s.context.trace_id for s in exporter.spans}) == 100


def test_start_tracing_intermediate_rate(stop_tracing_after):
    exporter = CollectingExporter()
    _start(exporter, 0.5)
    _make_traces(get_tracer("e2e"), 1000)
    auto.force_flush()
    kept = len({s.context.trace_id for s in exporter.spans})
    assert 0.40 < kept / 1000 < 0.60


def test_start_tracing_applies_to_tracer_fetched_before_init(stop_tracing_after):
    early = traccia.get_tracer("module-level")
    exporter = CollectingExporter()
    _start(exporter, 0.0, service_name="sampling-test")
    _make_traces(early, 20)
    auto.force_flush()
    assert exporter.spans == []


def test_start_tracing_keeps_sampler_when_otel_provider_is_rebuilt(stop_tracing_after):
    exporter = CollectingExporter()
    provider = _start(exporter, 0.0, service_name="sampling-test", service_role="orchestrator")
    assert provider._otel_provider.sampler is provider._sampler_switch
    assert "TraceIdRatioBased{0.0}" in provider._otel_provider.sampler.get_description()


def test_start_tracing_sample_rate_from_env(stop_tracing_after, monkeypatch):
    monkeypatch.setenv("TRACCIA_SAMPLE_RATE", "0")
    exporter = CollectingExporter()
    auto.start_tracing(
        exporter=exporter,
        enable_costs=False,
        enable_metrics=False,
        enable_patching=False,
    )
    _make_traces(get_tracer("e2e"), 20)
    auto.force_flush()
    assert exporter.spans == []


# --- Attributes on unsampled spans (PR review) -------------------------------


def _cost_provider(rate):
    from traccia.processors.cost_processor import CostAnnotatingProcessor

    provider = TracerProvider()
    provider.add_span_processor(
        CostAnnotatingProcessor(
            pricing_table={"gpt-4o": {"prompt": 0.005, "completion": 0.015}},
            pricing_source="test",
            pricing_generated_at="2026-01-01T00:00:00Z",
        )
    )
    provider.set_sampler(Sampler(rate))
    return provider


LLM_ATTRS = {
    "llm.model": "gpt-4o",
    "llm.usage.prompt_tokens": 1000,
    "llm.usage.completion_tokens": 500,
}


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_initial_attributes_drive_enrichment_at_any_rate(rate):
    """Identical inputs give the same cost whether or not the trace is sampled."""
    span = _cost_provider(rate).get_tracer("t").start_span("llm", attributes=dict(LLM_ATTRS))
    span.end()
    for key, value in LLM_ATTRS.items():
        assert span.attributes[key] == value
    assert span.attributes["llm.cost.usd"] == pytest.approx(0.0125)


def test_attributes_set_after_start_are_kept_on_unsampled_spans():
    span = _cost_provider(0.0).get_tracer("t").start_span("llm")
    for key, value in LLM_ATTRS.items():
        span.set_attribute(key, value)
    span.end()
    assert span.attributes["llm.cost.usd"] == pytest.approx(0.0125)


def test_unsampled_span_attributes_are_a_copy_of_the_input():
    attrs = dict(LLM_ATTRS)
    span = _cost_provider(0.0).get_tracer("t").start_span("llm", attributes=attrs)
    span.set_attribute("extra", 1)
    assert "extra" not in attrs


def test_unsampled_attributes_are_still_not_exported(provider_with_memory):
    provider, memory = provider_with_memory
    provider.set_sampler(Sampler(0.0))
    get_tracer("t").start_span("llm", attributes=dict(LLM_ATTRS)).end()
    assert memory.get_finished_spans() == ()


@pytest.mark.parametrize("lookup", ["context", "tracer"])
def test_attributes_set_via_current_span_reach_enrichment_when_unsampled(lookup):
    from traccia.context.context import get_current_span as context_current_span

    provider = _cost_provider(0.0)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("llm") as span:
        current = context_current_span() if lookup == "context" else tracer.get_current_span()
        assert current is span
        for key, value in LLM_ATTRS.items():
            current.set_attribute(key, value)
    assert span.attributes["llm.cost.usd"] == pytest.approx(0.0125)


def test_current_span_lookup_still_wraps_sampled_spans(provider_with_memory):
    provider, memory = provider_with_memory
    tracer = get_tracer("t")
    with tracer.start_as_current_span("root") as span:
        current = tracer.get_current_span()
        current.set_attribute("k", "v")
        assert current.context.span_id == span.context.span_id
    assert memory.get_finished_spans()[0].attributes["k"] == "v"
