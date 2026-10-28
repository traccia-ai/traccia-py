"""Tests for the batching span processor (TRC-278)."""

import threading
import time

from traccia.auto import _combine_exporters, start_tracing, stop_tracing
from traccia.processors.batch_processor import BatchSpanProcessor
from traccia.processors.rate_limiter import RateLimitingSpanProcessor
from traccia.tracer import TracerProvider
from traccia.tracer.provider import SpanProcessor


class _CollectingExporter:
    def __init__(self):
        self.events = []

    @property
    def exported_names(self):
        return [name for kind, names in self.events if kind == "export" for name in names]

    def export(self, spans):
        self.events.append(("export", [span.name for span in spans]))
        return True

    def shutdown(self):
        self.events.append(("shutdown", []))


class _FlushDuringEnrichment(SpanProcessor):
    """Drains the batch queue while Span.end() is still running enrichment."""

    def __init__(self, batch):
        self.batch = batch

    def on_end(self, span):
        self.batch.force_flush()


class _PendingSpan:
    def __init__(self, name):
        self.name = name
        self._ended = False


def _ended_span(name):
    span = _PendingSpan(name)
    span._ended = True
    return span


def _batch(exporter, **kwargs):
    # Long delay keeps the timed flush out of the way; tests flush explicitly.
    return BatchSpanProcessor(exporter, schedule_delay_millis=60_000, **kwargs)


def test_span_drained_while_ending_is_still_exported():
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    provider.add_span_processor(batch)
    provider.add_span_processor(_FlushDuringEnrichment(batch))

    span = tracer.start_span("work")
    span.end()
    batch.shutdown()

    assert exporter.exported_names == ["work"]


def test_span_is_queued_only_once_it_has_ended():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    span = _PendingSpan("pending")

    batch.on_end(span)
    batch.force_flush()
    assert exporter.exported_names == []
    assert not batch._queue

    span._ended = True
    batch.on_ended(span)
    batch.force_flush()
    assert exporter.exported_names == ["pending"]
    batch.shutdown()


def test_unended_span_does_not_block_ended_spans():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    pending = _PendingSpan("pending")

    batch.on_end(pending)
    # Spans that reach on_end() already ended (the OTel adapter) are queued at once.
    batch.on_end(_ended_span("done"))
    batch.force_flush()
    assert exporter.exported_names == ["done"]

    pending._ended = True
    batch.on_ended(pending)
    batch.force_flush()
    assert exporter.exported_names == ["done", "pending"]
    batch.shutdown()


def test_shutdown_exports_then_shuts_down_exporter_once():
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    provider.add_span_processor(batch)

    tracer.start_span("work").end()
    batch.shutdown()
    # stop_tracing() shuts the processor down directly and again via the provider.
    provider.shutdown()

    assert exporter.events == [("export", ["work"]), ("shutdown", [])]


def test_concurrent_shutdowns_shut_the_exporter_down_once():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    batch.on_end(_ended_span("work"))

    start = threading.Barrier(8)

    def stop():
        start.wait()
        batch.shutdown()

    threads = [threading.Thread(target=stop) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert exporter.events == [("export", ["work"]), ("shutdown", [])]


def test_span_still_ending_at_shutdown_is_exported():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    span = _PendingSpan("ending")
    batch.on_end(span)

    stopper = threading.Thread(target=batch.shutdown)
    stopper.start()
    time.sleep(0.1)
    span._ended = True
    batch.on_ended(span)
    stopper.join(timeout=5)

    assert exporter.events == [("export", ["ending"]), ("shutdown", [])]


class _PausingLock:
    """Lets the worker run each time one thread releases the batch lock."""

    def __init__(self, lock, thread, until):
        self._lock = lock
        self._thread = thread
        self._until = until
        self.pausing = False

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        if self.pausing and threading.current_thread() is self._thread:
            deadline = time.monotonic() + 1
            while not self._until() and time.monotonic() < deadline:
                time.sleep(0.005)


def test_span_ending_at_shutdown_is_kept_whenever_the_worker_runs():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    # After every lock release in on_ended(), give the final flush time to finish.
    lock = _PausingLock(batch._lock, threading.current_thread(), lambda: not batch._accepting)
    batch._lock = lock
    span = _PendingSpan("ending")
    batch.on_end(span)

    stopper = threading.Thread(target=batch.shutdown)
    stopper.start()
    time.sleep(0.1)
    span._ended = True
    lock.pausing = True
    batch.on_ended(span)
    lock.pausing = False
    stopper.join(timeout=5)

    assert exporter.events == [("export", ["ending"]), ("shutdown", [])]


class _BlockingExporter:
    """The first export blocks until released, like a slow network export."""

    def __init__(self):
        self.events = []
        self.started = threading.Event()
        self.release = threading.Event()
        self._in_flight = 0
        self.max_in_flight = 0

    def export(self, spans):
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        self.events.append(f"export started {[span.name for span in spans]}")
        self.started.set()
        self.release.wait(timeout=10)
        self.events.append("export finished")
        self._in_flight -= 1
        return True

    def shutdown(self):
        self.events.append("exporter shutdown")


def _batch_exporting(exporter, **kwargs):
    """A batch whose worker is blocked inside an export of span 'first'."""
    batch = BatchSpanProcessor(exporter, schedule_delay_millis=10, **kwargs)
    batch.on_end(_ended_span("first"))
    assert exporter.started.wait(timeout=5)
    return batch


def test_shutdown_waits_for_an_active_export_before_closing_the_exporter():
    exporter = _BlockingExporter()
    batch = _batch_exporting(exporter)

    stopper = threading.Thread(target=batch.shutdown)
    stopper.start()
    stopper.join(timeout=0.3)
    assert stopper.is_alive()
    assert exporter.events == ["export started ['first']"]

    exporter.release.set()
    stopper.join(timeout=5)
    assert not stopper.is_alive()
    assert exporter.events == ["export started ['first']", "export finished", "exporter shutdown"]


def test_shutdown_is_bounded_with_spans_queued_behind_a_stuck_export():
    exporter = _BlockingExporter()
    batch = _batch_exporting(exporter, shutdown_timeout_millis=200)
    batch.on_end(_ended_span("second"))

    started = time.monotonic()
    batch.shutdown()
    elapsed = time.monotonic() - started
    exporter.release.set()

    assert elapsed < 2
    assert "exporter shutdown" not in exporter.events


def test_exports_never_run_concurrently():
    exporter = _BlockingExporter()
    batch = _batch_exporting(exporter)
    batch.on_end(_ended_span("second"))

    flusher = threading.Thread(target=batch.force_flush, kwargs={"timeout": 5})
    flusher.start()
    time.sleep(0.2)
    exporter.release.set()
    flusher.join(timeout=5)
    batch.shutdown()

    assert exporter.max_in_flight == 1
    assert exporter.events[:4] == [
        "export started ['first']",
        "export finished",
        "export started ['second']",
        "export finished",
    ]


def test_force_flush_returns_by_its_timeout_while_an_export_is_stuck():
    exporter = _BlockingExporter()
    batch = _batch_exporting(exporter)

    started = time.monotonic()
    batch.force_flush(timeout=0.2)
    elapsed = time.monotonic() - started
    exporter.release.set()
    batch.shutdown()

    assert elapsed < 2


def test_rate_limited_batch_exports_spans_after_they_end():
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    limiter = RateLimitingSpanProcessor(next_processor=batch, max_spans_per_second=1000)
    provider.add_span_processor(limiter)

    tracer.start_span("work").end()
    limiter.force_flush(timeout=1.0)
    assert exporter.exported_names == ["work"]
    limiter.shutdown()


def test_stop_tracing_with_rate_limiting_keeps_a_caller_owned_exporter_open():
    exporter = _CollectingExporter()
    start_tracing(
        exporter=exporter,
        enable_patching=False,
        max_spans_per_second=1000,
    )
    try:
        from traccia import get_tracer

        get_tracer("test").start_span("work").end()
    finally:
        stop_tracing()

    assert "work" in exporter.exported_names
    # The caller passed the exporter in, so the caller decides when to shut it down.
    assert ("shutdown", []) not in exporter.events


def test_stop_tracing_keeps_a_caller_owned_otlp_exporter_open():
    from traccia.exporter import OTLPExporter

    exporter = OTLPExporter(endpoint="http://127.0.0.1:9/v1/traces")
    exported = []
    shutdowns = []
    exporter._otel_exporter.export = lambda spans: exported.extend(span.name for span in spans)
    exporter._otel_exporter.shutdown = lambda: shutdowns.append(True)
    exporter.shutdown = lambda: shutdowns.append(True)

    start_tracing(exporter=exporter, enable_patching=False)
    try:
        from traccia import get_tracer

        # A tracer cached before start_tracing() still uses the previous OTel provider.
        get_tracer("otlp-test").start_span("work").end()
    finally:
        stop_tracing()

    assert exported == ["work"]
    # OTel's processor exports it, but closing it is still up to the caller.
    assert shutdowns == []


def test_combined_exporters_all_shut_down_when_one_fails():
    class _FailingShutdown(_CollectingExporter):
        def shutdown(self):
            raise RuntimeError("session already closed")

    second = _CollectingExporter()
    combined = _combine_exporters(_FailingShutdown(), second)

    combined.shutdown()

    assert second.events == [("shutdown", [])]
