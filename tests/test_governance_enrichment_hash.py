"""governance.integrity_hash is set on every span, with one fingerprint contract (TRC-300).

Before the fix the processor called ``format(ctx.trace_id, "032x")`` on Traccia's
``SpanContext``, whose IDs are hex strings. That raised ``ValueError``, the tracer's
enrichment loop swallowed it, and no SDK span carried the hash. Ingestion then
backfilled one, which hid the defect in the UI.

The contract is ``sha256("<32 hex trace id>:<16 hex span id>:<event_type>")``, the same
payload the Node SDK and the ingestion backfill hash.
"""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traccia.governance import schema as G
from traccia.processors.governance_enrichment import (
    GovernanceEnrichmentProcessor,
    integrity_fingerprint,
)
from traccia.tracer import tracer as tracer_module
from traccia.tracer.provider import SpanProcessor, TracerProvider
from traccia.tracer.span_context import SpanContext

TRACE_INT = 0x0AF7651916CD43DD8448EB211C80319C
SPAN_INT = 0x00F067AA0BA902B7
TRACE_HEX = "0af7651916cd43dd8448eb211c80319c"
SPAN_HEX = "00f067aa0ba902b7"


def _expected(trace_hex: str, span_hex: str, event_type: str) -> str:
    """The ingestion backfill's formula (traccia-trace-injection-service governance_normalizer)."""
    return hashlib.sha256(f"{trace_hex}:{span_hex}:{event_type}".encode()).hexdigest()


class _FakeSpan:
    def __init__(self, context=None, attributes=None, name="llm.call"):
        self.context = context
        self.attributes = dict(attributes or {})
        self.name = name

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.fixture
def provider():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider._otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(GovernanceEnrichmentProcessor())
    return provider, exporter


# --- direct processor call --------------------------------------------------------


def test_direct_enrichment_with_string_ids_does_not_raise():
    """The audit's direct probe: Traccia SpanContext carries hex-string IDs."""
    span = _FakeSpan(context=SpanContext(trace_id=TRACE_HEX, span_id=SPAN_HEX))
    GovernanceEnrichmentProcessor().on_end(span)
    assert span.attributes[G.INTEGRITY_HASH] == _expected(TRACE_HEX, SPAN_HEX, "inference")


def test_direct_enrichment_on_real_traccia_span(provider):
    tp, _ = provider
    with tp.get_tracer("t").start_as_current_span("direct") as span:
        assert isinstance(span.context.trace_id, str)
        GovernanceEnrichmentProcessor().on_end(span)
        expected = _expected(span.context.trace_id, span.context.span_id, "inference")
        assert span.attributes[G.INTEGRITY_HASH] == expected


# --- normal span completion -------------------------------------------------------


def test_every_exported_span_carries_the_sdk_hash(provider):
    tp, exporter = provider
    tracer = tp.get_tracer("t")
    for i in range(50):
        with tracer.start_as_current_span(f"llm-{i}"):
            pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 50
    for s in spans:
        ctx = s.get_span_context()
        attrs = s.attributes
        assert attrs[G.TIMESTAMP_SOURCE] == "sdk"
        assert attrs[G.INTEGRITY_HASH] == _expected(
            f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}", attrs[G.EVENT_TYPE]
        )


def test_nested_spans_get_distinct_hashes(provider):
    tp, exporter = provider
    tracer = tp.get_tracer("t")
    with tracer.start_as_current_span("agent"):
        with tracer.start_as_current_span("tool.search"):
            pass
    by_name = {s.name: s.attributes for s in exporter.get_finished_spans()}
    assert by_name["tool.search"][G.EVENT_TYPE] == "tool_call"
    assert by_name["agent"][G.EVENT_TYPE] == "inference"
    assert by_name["agent"][G.INTEGRITY_HASH] != by_name["tool.search"][G.INTEGRITY_HASH]


def test_hash_uses_explicit_event_type(provider):
    tp, exporter = provider
    with tp.get_tracer("t").start_as_current_span("x", attributes={G.EVENT_TYPE: "retrieval"}):
        pass
    [s] = exporter.get_finished_spans()
    ctx = s.get_span_context()
    assert s.attributes[G.INTEGRITY_HASH] == _expected(
        f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}", "retrieval"
    )


def test_existing_hash_is_not_overwritten(provider):
    tp, exporter = provider
    with tp.get_tracer("t").start_as_current_span("x", attributes={G.INTEGRITY_HASH: "preset"}):
        pass
    [s] = exporter.get_finished_spans()
    assert s.attributes[G.INTEGRITY_HASH] == "preset"


def test_other_governance_attributes_still_set():
    proc = GovernanceEnrichmentProcessor(eu_risk_tier="high", hipaa_enabled=True)
    span = _FakeSpan(
        context=SpanContext(trace_id=TRACE_HEX, span_id=SPAN_HEX),
        attributes={G.REDACTION_APPLIED: True},
    )
    proc.on_end(span)
    a = span.attributes
    assert a[G.EVENT_TYPE] == "inference"
    assert a[G.TIMESTAMP_SOURCE] == "sdk"
    assert a[G.RISK_TIER] == "high"
    assert a[G.HIPAA_FRAMEWORK_ENABLED] is True
    assert a[G.HIPAA_PHI_REDACTION_APPLIED] is True
    assert G.INTEGRITY_HASH in a


# --- ID canonicalisation ------------------------------------------------------------


@pytest.mark.parametrize(
    "trace_id, span_id",
    [
        (TRACE_INT, SPAN_INT),  # OTel SDK ints
        (TRACE_HEX, SPAN_HEX),  # Traccia SpanContext / OTLP JSON
        (TRACE_HEX.upper(), SPAN_HEX.upper()),
        ("0x" + TRACE_HEX, "0X" + SPAN_HEX.upper()),
        (f"  {TRACE_HEX}\n", f" {SPAN_HEX} "),
        (TRACE_HEX.lstrip("0"), SPAN_HEX.lstrip("0")),  # leading zeros dropped
        (bytes.fromhex(TRACE_HEX), bytes.fromhex(SPAN_HEX)),  # OTLP protobuf bytes
    ],
)
def test_every_id_form_gives_the_same_fingerprint(trace_id, span_id):
    assert integrity_fingerprint(trace_id, span_id, "inference") == _expected(
        TRACE_HEX, SPAN_HEX, "inference"
    )


@pytest.mark.parametrize("ctx", [None, SpanContext(trace_id="", span_id="")])
def test_missing_context_still_gets_a_hash(ctx):
    span = _FakeSpan(context=ctx)
    GovernanceEnrichmentProcessor().on_end(span)
    assert span.attributes[G.INTEGRITY_HASH] == _expected("", "", "inference")


def test_span_without_context_attribute():
    span = SimpleNamespace(attributes={}, name="x")
    span.set_attribute = lambda k, v: span.attributes.__setitem__(k, v)
    GovernanceEnrichmentProcessor().on_end(span)
    assert span.attributes[G.INTEGRITY_HASH] == _expected("", "", "inference")


def test_non_hex_string_id_does_not_raise():
    # Not a valid OTel ID, but enrichment must still finish and stay deterministic.
    a = integrity_fingerprint("not-hex", "??", "inference")
    assert a == integrity_fingerprint("NOT-HEX", "??", "inference")
    assert a == _expected("not-hex", "??", "inference")


@pytest.mark.parametrize("bad", [object(), 1.5, True, -1, ["a"]])
def test_unexpected_id_types_do_not_raise(bad):
    assert integrity_fingerprint(bad, bad, "inference") == _expected("", "", "inference")


# --- failures are no longer silent -----------------------------------------------


_TRACER_LOGGER = "traccia.tracer.tracer"


def _tracer_records(caplog):
    """Only the tracer's own records: caplog also captures unrelated OTel warnings."""
    return [r for r in caplog.records if r.name == _TRACER_LOGGER]


class _Boom(SpanProcessor):
    def on_end(self, span):
        raise RuntimeError("boom")


def test_failing_processor_warns_once_and_does_not_block_others(caplog, monkeypatch):
    monkeypatch.setattr(tracer_module, "_warned_processor_failures", set())
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    tp._otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tp.add_span_processor(_Boom())
    tp.add_span_processor(GovernanceEnrichmentProcessor())
    tracer = tp.get_tracer("t")
    with caplog.at_level(logging.DEBUG, logger=_TRACER_LOGGER):
        for _ in range(3):
            with tracer.start_as_current_span("x"):
                pass
    records = _tracer_records(caplog)
    warnings = [r for r in records if r.levelno == logging.WARNING]
    debugs = [r for r in records if r.levelno == logging.DEBUG and "_Boom" in r.getMessage()]
    assert len(warnings) == 1 and "_Boom.on_end" in warnings[0].getMessage()
    assert warnings[0].exc_info is not None
    assert len(debugs) == 2
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert all(G.INTEGRITY_HASH in s.attributes for s in spans)


def test_governance_processor_logs_nothing_on_success(caplog, provider):
    tp, _ = provider
    with caplog.at_level(logging.DEBUG, logger=_TRACER_LOGGER):
        with tp.get_tracer("t").start_as_current_span("x"):
            pass
    assert not _tracer_records(caplog)
