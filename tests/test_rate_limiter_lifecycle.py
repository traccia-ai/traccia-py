"""Tests for rate limiter OpenTelemetry lifecycle integration and adapters."""

import os
import tempfile
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import (
    SpanProcessor as OTelSpanProcessor,
    TracerProvider as OTelTracerProvider,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traccia import get_tracer, init, stop_tracing
from traccia.processors.rate_limiter import RateLimitingSpanProcessor
from traccia.tracer.provider import TracerProvider


def test_rate_limiting_processor_subclasses_otel_span_processor():
    """Verify RateLimitingSpanProcessor subclasses OpenTelemetry SpanProcessor."""
    proc = RateLimitingSpanProcessor()
    assert isinstance(proc, OTelSpanProcessor)
    assert hasattr(proc, "_on_ending")
    assert hasattr(proc, "on_start")
    assert hasattr(proc, "on_end")
    assert hasattr(proc, "shutdown")
    assert hasattr(proc, "force_flush")


def test_native_otel_span_lifecycle():
    """Verify native OTel span completion does not raise on _on_ending and exports."""
    exporter = InMemorySpanExporter()
    otel_proc = SimpleSpanProcessor(exporter)
    rl_proc = RateLimitingSpanProcessor(next_processor=otel_proc, max_spans_per_second=10.0)

    tp = OTelTracerProvider()
    tp.add_span_processor(rl_proc)
    tracer = tp.get_tracer("native_test")

    span = tracer.start_span("native_span")
    span.end()

    assert len(exporter.get_finished_spans()) == 1
    assert exporter.get_finished_spans()[0].name == "native_span"


def test_traccia_wrapped_span_lifecycle_export():
    """Verify Traccia-wrapped span completion exports through RateLimitingSpanProcessor."""
    exporter = InMemorySpanExporter()
    otel_proc = SimpleSpanProcessor(exporter)
    rl_proc = RateLimitingSpanProcessor(next_processor=otel_proc, max_spans_per_second=10.0)

    provider = TracerProvider()
    provider._otel_provider.add_span_processor(rl_proc)

    tracer = provider.get_tracer("traccia_test")
    with tracer.start_as_current_span("traccia_span"):
        pass

    assert len(exporter.get_finished_spans()) == 1
    assert exporter.get_finished_spans()[0].name == "traccia_span"


def test_on_ending_forwarding():
    """Verify _on_ending is forwarded to next processor if implemented."""
    mock_downstream = MagicMock()
    rl_proc = RateLimitingSpanProcessor(next_processor=mock_downstream)

    fake_span = MagicMock()
    rl_proc._on_ending(fake_span)
    mock_downstream._on_ending.assert_called_once_with(fake_span)


def test_force_flush_timeout_adapters():
    """Verify force_flush adapts between OTel timeout_millis and Traccia timeout."""
    class OTelMockProc:
        def __init__(self):
            self.last_millis = None

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            self.last_millis = timeout_millis
            return True

    otel_mock = OTelMockProc()
    rl_proc = RateLimitingSpanProcessor(next_processor=otel_mock)
    rl_proc.force_flush(timeout=2.5)
    assert otel_mock.last_millis == 2500

    class TracciaMockProc:
        def __init__(self):
            self.last_timeout = None

        def force_flush(self, timeout=None):
            self.last_timeout = timeout
            return True

    traccia_mock = TracciaMockProc()
    rl_proc2 = RateLimitingSpanProcessor(next_processor=traccia_mock)
    rl_proc2.force_flush(timeout=1.5)
    assert traccia_mock.last_timeout == 1.5

    rl_proc2.force_flush(timeout_millis=4000)
    assert traccia_mock.last_timeout == 4.0


def test_stop_tracing_file_export_with_limiter():
    """Regression check: stop_tracing with file export and limiter does not raise TypeError."""
    tf = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tf.close()
    try:
        init(
            enable_file_exporter=True,
            file_exporter_path=tf.name,
            max_spans_per_second=10.0,
            auto_start_trace=False,
            enable_metrics=False,
            use_otlp=False,
        )
        tracer = get_tracer("file_export_test")
        with tracer.start_as_current_span("file_span"):
            pass
        stop_tracing(flush_timeout=1.0)

        with open(tf.name, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1
    finally:
        if os.path.exists(tf.name):
            try:
                os.unlink(tf.name)
            except Exception:
                pass
