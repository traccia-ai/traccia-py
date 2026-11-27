"""Tracers fetched before start_tracing() export through the provider it builds (TRC-298).

With OTLP on, start_tracing() replaces the OTel provider so it can set the resource
(OTel resources are immutable). Tracers handed out earlier, typically module-level
``traccia.get_tracer(...)`` calls, must create their spans on the new provider, and a
stop_tracing() / start_tracing() cycle must leave exactly one live export path.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import traccia
from traccia import auto, get_tracer_provider, set_tracer_provider
from traccia.tracer.provider import TracerProvider


class CollectingExporter:
    """Traccia-style exporter that keeps every exported span."""

    def __init__(self) -> None:
        self.spans = []

    def export(self, spans) -> bool:
        self.spans.extend(spans)
        return True

    def shutdown(self) -> None:
        pass

    @property
    def names(self):
        return [s.name for s in self.spans]


@pytest.fixture
def fresh_provider():
    """A fresh global provider; tracing is stopped and the old provider restored after."""
    previous = get_tracer_provider()
    provider = TracerProvider()
    set_tracer_provider(provider)
    yield provider
    auto.stop_tracing()
    set_tracer_provider(previous)


def _start(exporter, **kwargs):
    # use_otlp stays at its default (True): that is the path that replaces the OTel provider.
    return auto.start_tracing(
        exporter=exporter,
        enable_costs=False,
        enable_metrics=False,
        enable_patching=False,
        load_env=False,
        service_name="trc298-test",
        **kwargs,
    )


def test_init_replaces_the_otel_provider(fresh_provider):
    """Precondition for the rest: the OTLP path really swaps the OTel provider."""
    before = fresh_provider._otel_provider
    _start(CollectingExporter())
    assert fresh_provider._otel_provider is not before


def test_pre_init_tracer_spans_reach_processor_on_new_provider(fresh_provider):
    """The ticket's reproduction, step for step."""
    early = traccia.get_tracer("module-level")
    _start(CollectingExporter())
    memory = InMemorySpanExporter()
    fresh_provider._otel_provider.add_span_processor(SimpleSpanProcessor(memory))

    early.start_span("after-init").end()

    assert [s.name for s in memory.get_finished_spans()] == ["after-init"]


def test_pre_init_tracer_exports_through_init_exporter(fresh_provider):
    early = traccia.get_tracer("module-level")
    exporter = CollectingExporter()
    _start(exporter)

    with early.start_as_current_span("root"):
        with early.start_as_current_span("child"):
            pass
    auto.force_flush()

    assert sorted(exporter.names) == ["child", "root"]


def test_pre_init_tracer_spans_carry_init_resource(fresh_provider):
    early = traccia.get_tracer("module-level")
    _start(CollectingExporter(), service_role="orchestrator")
    memory = InMemorySpanExporter()
    fresh_provider._otel_provider.add_span_processor(SimpleSpanProcessor(memory))

    early.start_span("after-init").end()

    (span,) = memory.get_finished_spans()
    assert span.resource.attributes["service.name"] == "trc298-test"
    assert span.resource.attributes["traccia.service_role"] == "orchestrator"


def test_pre_init_and_post_init_tracers_share_the_live_provider(fresh_provider):
    early = traccia.get_tracer("shared-scope")
    _start(CollectingExporter())
    late = traccia.get_tracer("shared-scope")

    assert late is early
    assert early._otel_tracer is fresh_provider._otel_provider.get_tracer("shared-scope")


def test_export_processor_registered_before_init_survives_replacement(fresh_provider):
    memory = InMemorySpanExporter()
    fresh_provider.add_span_processor(SimpleSpanProcessor(memory))
    early = traccia.get_tracer("module-level")
    _start(CollectingExporter())

    early.start_span("after-init").end()

    assert [s.name for s in memory.get_finished_spans()] == ["after-init"]


def test_retired_otel_provider_receives_no_new_spans(fresh_provider):
    early = traccia.get_tracer("module-level")
    retired_memory = InMemorySpanExporter()
    retired = fresh_provider._otel_provider
    retired.add_span_processor(SimpleSpanProcessor(retired_memory))
    _start(CollectingExporter())

    early.start_span("after-init").end()

    assert retired_memory.get_finished_spans() == ()


# --- Reinitialization: stop_tracing() then start_tracing() ---------------------


def test_reinit_exports_pre_init_tracer_spans_only_to_new_exporter(fresh_provider):
    early = traccia.get_tracer("module-level")
    first = CollectingExporter()
    _start(first)
    early.start_span("first-run").end()
    auto.stop_tracing()

    second = CollectingExporter()
    _start(second)
    early.start_span("second-run").end()
    auto.force_flush()

    assert first.names == ["first-run"]
    assert second.names == ["second-run"]


def test_reinit_does_not_duplicate_exports(fresh_provider):
    """Each cycle must not leave an extra export processor behind on the provider."""
    early = traccia.get_tracer("module-level")
    for _ in range(3):
        _start(CollectingExporter())
        auto.stop_tracing()

    last = CollectingExporter()
    _start(last)
    for _ in range(5):
        early.start_span("span").end()
    auto.force_flush()

    assert last.names == ["span"] * 5


def test_reinit_does_not_accumulate_processors(fresh_provider):
    _start(CollectingExporter())
    enrichment = len(fresh_provider._enrichment_processors)
    export = len(fresh_provider._export_processors)
    auto.stop_tracing()

    _start(CollectingExporter())

    assert len(fresh_provider._enrichment_processors) == enrichment
    assert len(fresh_provider._export_processors) == export


class CountingProcessor:
    """Traccia enrichment processor that counts the spans it sees."""

    def __init__(self) -> None:
        self.ended = 0

    def on_end(self, span) -> None:
        self.ended += 1

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout=None) -> None:
        pass


def test_reinit_keeps_user_registered_enrichment_processor(fresh_provider):
    """stop_tracing() removes only what start_tracing() added."""
    counting = CountingProcessor()
    fresh_provider.add_span_processor(counting)
    early = traccia.get_tracer("module-level")
    _start(CollectingExporter())
    auto.stop_tracing()
    _start(CollectingExporter())

    early.start_span("after-reinit").end()

    assert counting in fresh_provider._enrichment_processors
    assert counting.ended == 1


def test_reinit_without_otlp_does_not_duplicate_exports(fresh_provider, tmp_path):
    """use_otlp=False keeps the same OTel provider, so only the processor cleanup applies."""
    early = traccia.get_tracer("module-level")
    for run in range(2):
        exporter = CollectingExporter()
        auto.start_tracing(
            exporter=exporter,
            use_otlp=False,
            enable_file_exporter=True,
            file_exporter_path=str(tmp_path / f"traces-{run}.jsonl"),
            enable_costs=False,
            enable_metrics=False,
            enable_patching=False,
            load_env=False,
        )
        early.start_span(f"run-{run}").end()
        auto.force_flush()
        assert exporter.names == [f"run-{run}"]
        auto.stop_tracing()


def test_failed_start_does_not_leave_processors_behind(fresh_provider, monkeypatch):
    _start(CollectingExporter())
    clean = len(fresh_provider._enrichment_processors)
    auto.stop_tracing()

    def _fail(*args, **kwargs):
        raise RuntimeError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(auto, "_start_pricing_refresh", _fail)
        with pytest.raises(RuntimeError):
            _start(CollectingExporter())

    _start(CollectingExporter())
    assert len(fresh_provider._enrichment_processors) == clean


def test_reinit_resource_reflects_second_init(fresh_provider):
    early = traccia.get_tracer("module-level")
    _start(CollectingExporter(), service_role="first-role")
    auto.stop_tracing()

    _start(CollectingExporter(), service_role="second-role")
    memory = InMemorySpanExporter()
    fresh_provider._otel_provider.add_span_processor(SimpleSpanProcessor(memory))
    early.start_span("after-reinit").end()

    (span,) = memory.get_finished_spans()
    assert span.resource.attributes["traccia.service_role"] == "second-role"
