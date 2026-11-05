"""Tests for the batching span processor (TRC-278)."""

import asyncio
import json
import logging
import os
import select
import signal
import threading
import time
import traceback
import warnings

import pytest

from traccia.auto import _combine_exporters, start_tracing, stop_tracing
from traccia.exporter import FileExporter
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


def test_shutdown_does_not_wait_for_spans_that_start_ending_after_it():
    exporter = _CollectingExporter()
    batch = _batch(exporter, shutdown_timeout_millis=5000)
    early = _PendingSpan("early")
    batch.on_end(early)

    stopper = threading.Thread(target=batch.shutdown)
    stopper.start()
    deadline = time.monotonic() + 5
    while not batch._shutdown and time.monotonic() < deadline:
        time.sleep(0.005)
    # Under load, spans keep starting to end; shutdown must not wait for those.
    late = _PendingSpan("late")
    batch.on_end(late)
    early._ended = True
    batch.on_ended(early)
    stopper.join(timeout=2)

    assert not stopper.is_alive()
    assert exporter.events == [("export", ["early"]), ("shutdown", [])]


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


def test_shutdown_is_bounded_with_spans_queued_behind_a_stuck_export(caplog):
    exporter = _BlockingExporter()
    batch = _batch_exporting(exporter, shutdown_timeout_millis=200)
    batch.on_end(_ended_span("second"))

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="traccia.processors.batch_processor"):
        batch.shutdown()
    elapsed = time.monotonic() - started

    assert elapsed < 2
    # Not closed under the running export...
    assert "exporter shutdown" not in exporter.events
    assert "Span export still running after 0.2s; the exporter is closed once" in caplog.text
    # ...but by the worker once it has exported the rest of the queue.
    exporter.release.set()
    batch._worker.join(timeout=5)
    assert exporter.events[-3:] == ["export started ['second']", "export finished", "exporter shutdown"]


def test_shutdown_closes_the_exporter_when_the_worker_is_late_but_not_exporting(caplog):
    class _LateWorker(BatchSpanProcessor):
        def _export_pending(self):
            if self._shutdown and time.monotonic() >= self._shutdown_deadline:
                # The worker's last steps run just after shutdown's join gave up.
                time.sleep(0.2)
            super()._export_pending()

    exporter = _CollectingExporter()
    batch = _LateWorker(exporter, schedule_delay_millis=60_000, shutdown_timeout_millis=200)
    # Still ending at the deadline, so the final flush waits until then.
    pending = _PendingSpan("pending")
    batch.on_end(pending)

    with caplog.at_level(logging.WARNING, logger="traccia.processors.batch_processor"):
        batch.shutdown()
        # The worker shuts the exporter down once it is past its last steps.
        batch._worker.join(timeout=5)

    assert exporter.events == [("shutdown", [])]
    assert "Span export still running" not in caplog.text


class _SlowToCloseExporter(_CollectingExporter):
    """shutdown() blocks until released, like a cleanup stuck on the network."""

    def __init__(self):
        super().__init__()
        self.closing = threading.Event()
        self.release = threading.Event()
        self.exporting = False
        self.closed_while_exporting = False

    def export(self, spans):
        self.exporting = True
        try:
            return super().export(spans)
        finally:
            self.exporting = False

    def shutdown(self):
        self.closed_while_exporting = self.closed_while_exporting or self.exporting
        self.closing.set()
        self.release.wait(timeout=10)
        super().shutdown()


def test_shutdown_timeout_bounds_the_exporters_own_shutdown():
    exporter = _SlowToCloseExporter()
    batch = _batch(exporter)
    batch.on_end(_ended_span("work"))

    started = time.monotonic()
    batch.shutdown(timeout=0.02)
    elapsed = time.monotonic() - started
    # A later stop_tracing() (or the exit hook) does not wait for it either.
    batch.shutdown()
    elapsed_again = time.monotonic() - started - elapsed

    assert elapsed < 0.2
    assert elapsed_again < 0.2
    # The exporter is still shut down, once, after the shutdown() calls returned.
    assert exporter.closing.wait(timeout=5)
    exporter.release.set()
    batch._worker.join(timeout=5)
    assert not batch._worker.is_alive()
    assert exporter.events.count(("shutdown", [])) == 1
    assert exporter.events[-1] == ("shutdown", [])


def test_exporter_shutdown_runs_once_after_the_last_export_when_it_is_fast():
    exporter = _SlowToCloseExporter()
    exporter.release.set()
    batch = _batch(exporter)
    batch.on_end(_ended_span("work"))

    batch.shutdown()
    batch.shutdown()

    assert exporter.events == [("export", ["work"]), ("shutdown", [])]
    assert not exporter.closed_while_exporting


def test_exporter_shutdown_waits_for_an_export_still_running_at_the_deadline():
    class _StuckThenSlowToClose(_BlockingExporter):
        def __init__(self):
            super().__init__()
            self.shutdowns = 0
            self.closed_while_exporting = False

        def shutdown(self):
            self.shutdowns += 1
            self.closed_while_exporting = self.closed_while_exporting or self._in_flight > 0
            super().shutdown()

    exporter = _StuckThenSlowToClose()
    batch = _batch_exporting(exporter)

    started = time.monotonic()
    batch.shutdown(timeout=0.02)
    assert time.monotonic() - started < 0.2
    assert exporter.shutdowns == 0

    exporter.release.set()
    batch._worker.join(timeout=5)
    assert not batch._worker.is_alive()
    assert exporter.shutdowns == 1
    assert not exporter.closed_while_exporting


def test_stop_tracing_flush_timeout_bounds_the_exporters_own_shutdown(tmp_path, monkeypatch):
    closing, release = threading.Event(), threading.Event()
    calls = []

    def slow_shutdown(self):
        calls.append(self)
        closing.set()
        release.wait(timeout=10)

    # An exporter the SDK created, so stop_tracing() shuts it down.
    monkeypatch.setattr(FileExporter, "shutdown", slow_shutdown)
    start_tracing(
        use_otlp=False,
        enable_file_exporter=True,
        file_exporter_path=str(tmp_path / "traces.jsonl"),
        enable_patching=False,
        enable_metrics=False,
    )
    try:
        from traccia import get_tracer

        get_tracer("test").start_span("work").end()
        started = time.monotonic()
        stop_tracing(flush_timeout=0.05)
        elapsed = time.monotonic() - started
        assert closing.wait(timeout=5)
    finally:
        release.set()

    assert elapsed < 1
    assert len(calls) == 1


class _ForwardingOnlyOnEnd(SpanProcessor):
    """A user's wrapper written before on_ended() existed."""

    def __init__(self, batch):
        self.batch = batch

    def on_end(self, span):
        self.batch.on_end(span)

    def shutdown(self):
        self.batch.shutdown()

    def force_flush(self, timeout=None):
        self.batch.force_flush(timeout=timeout)


def test_spans_forwarded_only_through_on_end_are_exported_by_force_flush():
    provider = TracerProvider()
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    provider.add_span_processor(_ForwardingOnlyOnEnd(batch))

    provider.get_tracer("test").start_span("work").end()
    batch.force_flush(timeout=5)

    assert exporter.exported_names == ["work"]
    batch.shutdown()


def test_spans_forwarded_only_through_on_end_do_not_stall_shutdown():
    provider = TracerProvider()
    exporter = _CollectingExporter()
    batch = _batch(exporter, shutdown_timeout_millis=5000)
    provider.add_span_processor(_ForwardingOnlyOnEnd(batch))

    provider.get_tracer("test").start_span("work").end()
    started = time.monotonic()
    batch.shutdown()

    assert time.monotonic() - started < 2
    assert exporter.events == [("export", ["work"]), ("shutdown", [])]


def test_span_queued_by_the_worker_is_not_queued_again_by_on_ended():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    span = _PendingSpan("work")
    batch.on_end(span)
    span._ended = True
    # The worker queues the ended span before its on_ended() runs.
    batch.force_flush(timeout=5)
    batch.on_ended(span)
    batch.shutdown()

    assert exporter.events == [("export", ["work"]), ("shutdown", [])]


def test_shutdown_does_not_wait_for_a_span_whose_end_was_interrupted():
    class _Interrupting(SpanProcessor):
        def on_end(self, span):
            raise KeyboardInterrupt()

    provider = TracerProvider()
    exporter = _CollectingExporter()
    batch = _batch(exporter, shutdown_timeout_millis=5000)
    provider.add_span_processor(batch)
    provider.add_span_processor(_Interrupting())

    span = provider.get_tracer("test").start_span("work")
    with pytest.raises(KeyboardInterrupt):
        span.end()
    started = time.monotonic()
    batch.shutdown()

    assert time.monotonic() - started < 2
    assert exporter.events == [("shutdown", [])]


def test_worker_survives_a_base_exception_from_the_exporter():
    class _CancellingExporter(_CollectingExporter):
        def export(self, spans):
            if not self.events:
                self.events.append(("cancelled", []))
                raise asyncio.CancelledError()
            return super().export(spans)

    exporter = _CancellingExporter()
    batch = _batch(exporter)
    batch.on_end(_ended_span("first"))
    batch.force_flush(timeout=5)
    batch.on_end(_ended_span("second"))

    flusher = threading.Thread(target=batch.force_flush, daemon=True)
    flusher.start()
    flusher.join(timeout=5)
    assert not flusher.is_alive()
    batch.shutdown()

    assert exporter.events == [("cancelled", []), ("export", ["second"]), ("shutdown", [])]


# The worker thread dies on purpose; pytest reports that as a warning.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_force_flush_and_shutdown_export_after_the_worker_dies():
    class _DyingWorker(BatchSpanProcessor):
        died = False

        def _export_pending(self):
            if threading.current_thread() is self._worker and not self.died:
                self.died = True
                # A BaseException outside any export kills the worker thread.
                raise SystemExit()
            super()._export_pending()

    exporter = _CollectingExporter()
    batch = _DyingWorker(exporter, schedule_delay_millis=10)
    batch._worker.join(timeout=5)
    assert not batch._worker.is_alive()

    batch.on_end(_ended_span("first"))
    flusher = threading.Thread(target=batch.force_flush, daemon=True)
    flusher.start()
    flusher.join(timeout=5)
    assert not flusher.is_alive()
    assert exporter.exported_names == ["first"]

    batch.on_end(_ended_span("second"))
    batch.shutdown()
    assert exporter.events == [("export", ["first"]), ("export", ["second"]), ("shutdown", [])]


class _SlowExporter(_CollectingExporter):
    def export(self, spans):
        time.sleep(0.05)
        return super().export(spans)


def _batch_with_queue(exporter, count, **kwargs):
    batch = BatchSpanProcessor(exporter, schedule_delay_millis=60_000, max_export_batch_size=10, **kwargs)
    # Queued without waking the worker, so shutdown's final flush exports them.
    with batch._lock:
        for i in range(count):
            batch._queue.append(_ended_span(str(i)))
    return batch


def test_exporter_left_open_at_the_deadline_still_gets_the_rest_of_the_queue(caplog):
    exporter = _SlowExporter()
    # 20 batches of 50ms cannot finish in 0.22s; the deadline falls inside an export.
    batch = _batch_with_queue(exporter, 200, shutdown_timeout_millis=220)
    with caplog.at_level(logging.WARNING, logger="traccia.processors.batch_processor"):
        batch.shutdown()
    assert "Span export still running" in caplog.text
    assert ("shutdown", []) not in exporter.events

    batch._worker.join(timeout=5)
    assert len(exporter.exported_names) == 200
    # The worker closes the exporter once the queue is exported.
    assert exporter.events[-1] == ("shutdown", [])


def test_shutdown_after_shutdown_from_inside_an_export_waits_for_the_final_flush():
    class _StopsTracing(_CollectingExporter):
        def __init__(self):
            super().__init__()
            self.stopped = threading.Event()

        def export(self, spans):
            if [span.name for span in spans] == ["first"]:
                batch.on_end(_ended_span("second"))
                batch.shutdown()  # e.g. stop_tracing() on a 401, on the worker
                self.stopped.set()
            time.sleep(0.1)
            return super().export(spans)

    exporter = _StopsTracing()
    batch = _batch(exporter)
    batch.on_end(_ended_span("first"))
    assert exporter.stopped.wait(timeout=5)
    # stop_tracing() from the main thread, just before the process exits.
    batch.shutdown()

    assert exporter.events == [("export", ["first"]), ("export", ["second"]), ("shutdown", [])]


# The worker thread dies on purpose; pytest reports that as a warning.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_shutdown_finishes_for_a_worker_that_stopped_tracing_then_died():
    class _DiesInFinalFlush(BatchSpanProcessor):
        def _final_flush(self):
            if threading.current_thread() is self._worker:
                # A BaseException outside any export kills the worker.
                raise SystemExit()
            super()._final_flush()

    class _StopsTracing(_CollectingExporter):
        def export(self, spans):
            super().export(spans)
            if [span.name for span in spans] == ["first"]:
                batch.on_end(_ended_span("second"))
                batch.shutdown()
            return True

    exporter = _StopsTracing()
    batch = _DiesInFinalFlush(exporter, schedule_delay_millis=60_000)
    batch.on_end(_ended_span("first"))
    batch._worker.join(timeout=5)
    assert not batch._worker.is_alive()
    # stop_tracing() later: finishes the final flush the worker never ran.
    batch.shutdown()

    assert exporter.events == [("export", ["first"]), ("export", ["second"]), ("shutdown", [])]


def test_exit_waits_for_a_shutdown_started_from_inside_an_export(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "stops_tracing_in_export.py"
    script.write_text(
        "import time\n"
        "from traccia import get_tracer\n"
        "from traccia.auto import start_tracing, stop_tracing\n"
        "class Exp:\n"
        "    calls = 0\n"
        "    def export(self, spans):\n"
        "        Exp.calls += 1\n"
        "        if Exp.calls == 1:\n"
        "            time.sleep(0.3)\n"
        "            stop_tracing()\n"
        "        else:\n"
        "            time.sleep(0.1)\n"
        "        for span in spans:\n"
        "            print(span.name, flush=True)\n"
        "        return True\n"
        "start_tracing(exporter=Exp(), enable_patching=False, enable_metrics=False, max_export_batch_size=10)\n"
        "tracer = get_tracer('exit-test')\n"
        "tracer.start_span('first').end()\n"
        "time.sleep(0.05)\n"
        "for i in range(100):\n"
        "    tracer.start_span(f's{i}').end()\n"
        "time.sleep(0.5)\n"
        "# No stop_tracing(): the app relies on the exit hook.\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=env
    )

    assert result.returncode == 0, result.stderr
    # 10 more exports of 0.1s are still running when the script ends.
    assert len(result.stdout.split()) == 101


def test_concurrent_shutdown_callers_all_wait_for_the_final_flush():
    exporter = _SlowExporter()
    batch = _batch_with_queue(exporter, 30)
    first = threading.Thread(target=batch.shutdown)
    first.start()
    deadline = time.monotonic() + 5
    while not batch._shutdown and time.monotonic() < deadline:
        time.sleep(0.001)
    # A second caller (e.g. the atexit hook) finds shutdown under way.
    batch.shutdown()
    returned_with = list(exporter.events)
    first.join(timeout=5)

    assert len([name for name in exporter.exported_names]) == 30
    assert returned_with[-1] == ("shutdown", [])


def test_spans_dropped_when_shutdown_closes_the_exporter_are_logged(caplog):
    class _LateWorker(BatchSpanProcessor):
        def _export_pending(self):
            if self._shutdown:
                # The deadline passes before the final flush exports anything.
                time.sleep(0.3)
            super()._export_pending()

    exporter = _CollectingExporter()
    batch = _LateWorker(exporter, schedule_delay_millis=60_000, shutdown_timeout_millis=100)
    with batch._lock:
        for i in range(3):
            batch._queue.append(_ended_span(str(i)))
    with caplog.at_level(logging.WARNING, logger="traccia.processors.batch_processor"):
        batch.shutdown()
        batch._worker.join(timeout=5)

    assert exporter.events == [("shutdown", [])]
    assert "Dropped 3 queued spans" in caplog.text


def test_shutdown_from_inside_an_export_closes_the_exporter_after_it():
    class _GivingUp(_CollectingExporter):
        def export(self, spans):
            super().export(spans)
            if [span.name for span in spans] == ["a"]:
                # e.g. stop_tracing() on a 401: runs on the worker thread.
                batch.shutdown()
            return True

    exporter = _GivingUp()
    batch = _batch(exporter)
    batch.on_end(_ended_span("a"))
    batch._worker.join(timeout=5)
    assert not batch._worker.is_alive()
    # A later stop_tracing() finds it already shut down.
    batch.shutdown()

    assert exporter.events == [("export", ["a"]), ("shutdown", [])]


def test_rate_limiter_passes_the_shutdown_timeout_only_where_accepted():
    class _Plain:
        def __init__(self):
            self.calls = []

        def shutdown(self):
            self.calls.append("shutdown")

    plain = _Plain()
    RateLimitingSpanProcessor(next_processor=plain, max_spans_per_second=10).shutdown(timeout=1.0)
    assert plain.calls == ["shutdown"]

    exporter = _CollectingExporter()
    batch = _batch(exporter)
    RateLimitingSpanProcessor(next_processor=batch, max_spans_per_second=10).shutdown(timeout=1.0)
    assert exporter.events == [("shutdown", [])]


def test_force_flush_with_timeout_zero_waits_for_the_flush():
    exporter = _BlockingExporter()
    batch = _batch_exporting(exporter)
    batch.on_end(_ended_span("second"))

    # timeout=0 has always meant no deadline (traccia.force_flush(flush_timeout=0)).
    flusher = threading.Thread(target=batch.force_flush, kwargs={"timeout": 0}, daemon=True)
    flusher.start()
    flusher.join(timeout=0.3)
    assert flusher.is_alive()

    exporter.release.set()
    flusher.join(timeout=5)
    assert not flusher.is_alive()
    assert "export started ['second']" in exporter.events
    batch.shutdown()


def test_stop_tracing_flush_timeout_bounds_the_shutdown():
    exporter = _BlockingExporter()
    start_tracing(exporter=exporter, enable_patching=False)
    try:
        from traccia import get_tracer

        get_tracer("test").start_span("work").end()
        assert exporter.started.wait(timeout=5)
        started = time.monotonic()
        stop_tracing(flush_timeout=0.3)
        elapsed = time.monotonic() - started
    finally:
        exporter.release.set()

    # Not flush_timeout plus the processor's own 30s shutdown timeout.
    assert elapsed < 5


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


needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def _in_forked_child(child, timeout=10):
    """Run child() in a forked process and return its JSON result; fail if it hangs."""
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        # Python 3.12 warns when a process with threads forks; that is the case under test.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        status = 1
        try:
            os.close(read_fd)
            os.write(write_fd, json.dumps(child()).encode())
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    os.close(write_fd)
    chunks = []
    try:
        ready, _, _ = select.select([read_fd], [], [], timeout)
        if not ready:
            os.kill(pid, signal.SIGKILL)
        else:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
    finally:
        os.close(read_fd)
        _, wait_status = os.waitpid(pid, 0)
    if not ready:
        pytest.fail(f"forked child still running after {timeout}s")
    assert os.waitstatus_to_exitcode(wait_status) == 0
    return json.loads(b"".join(chunks))


def _holding(lock):
    """Start a thread that holds lock; returns the Event that releases it."""
    held, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            held.set()
            release.wait(timeout=10)

    threading.Thread(target=hold, daemon=True).start()
    assert held.wait(timeout=5)
    return release


@needs_fork
def test_forked_child_flushes_its_own_spans():
    exporter = _CollectingExporter()
    batch = _batch(exporter)

    def child():
        batch.on_end(_ended_span("child"))
        batch.force_flush()
        return exporter.exported_names

    assert _in_forked_child(child) == ["child"]
    batch.shutdown()


@needs_fork
def test_forked_child_exports_on_schedule():
    exporter = _CollectingExporter()
    batch = BatchSpanProcessor(exporter, schedule_delay_millis=20)

    def child():
        batch.on_end(_ended_span("child"))
        deadline = time.monotonic() + 5
        while not exporter.exported_names and time.monotonic() < deadline:
            time.sleep(0.01)
        return exporter.exported_names

    assert _in_forked_child(child) == ["child"]
    batch.shutdown()


@needs_fork
def test_forked_child_shutdown_exports_without_waiting_for_parent_spans():
    exporter = _CollectingExporter()
    batch = _batch(exporter, shutdown_timeout_millis=5000)
    # Still ending on a parent thread at the fork; that thread does not exist in the child.
    pending = _PendingSpan("parent")
    batch.on_end(pending)

    def child():
        batch.on_end(_ended_span("child"))
        started = time.monotonic()
        batch.shutdown()
        return {"elapsed": time.monotonic() - started, "events": exporter.events}

    result = _in_forked_child(child)
    assert result["events"] == [["export", ["child"]], ["shutdown", []]]
    assert result["elapsed"] < 2

    pending._ended = True
    batch.on_ended(pending)
    batch.shutdown()
    assert exporter.events == [("export", ["parent"]), ("shutdown", [])]


@needs_fork
def test_forked_child_works_when_the_fork_lands_while_the_lock_is_held():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    release = _holding(batch._lock)

    def child():
        batch.on_end(_ended_span("child"))
        batch.force_flush()
        return exporter.exported_names

    try:
        assert _in_forked_child(child) == ["child"]
    finally:
        release.set()
    batch.shutdown()


@needs_fork
def test_forked_child_does_not_export_the_parents_spans():
    exporter = _BlockingExporter()
    batch = _batch_exporting(exporter)
    # Queued behind the parent's running export; the parent exports it.
    batch.on_end(_ended_span("second"))

    def child():
        # Only the parent's worker is blocked; let the child's exports through.
        exporter.release = threading.Event()
        exporter.release.set()
        batch.on_end(_ended_span("child"))
        batch.force_flush()
        return exporter.events

    child_events = _in_forked_child(child)
    assert child_events == [
        "export started ['first']",
        "export started ['child']",
        "export finished",
    ]

    exporter.release.set()
    batch.shutdown()
    assert exporter.events == [
        "export started ['first']",
        "export finished",
        "export started ['second']",
        "export finished",
        "exporter shutdown",
    ]


@needs_fork
def test_forked_child_of_a_shut_down_processor_stays_shut_down():
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    batch.shutdown()

    def child():
        batch.on_end(_ended_span("child"))
        batch.force_flush()
        batch.shutdown()
        return {"events": exporter.events, "worker": batch._worker.is_alive()}

    assert _in_forked_child(child) == {"events": [["shutdown", []]], "worker": False}


@needs_fork
def test_forked_child_of_a_processor_shutting_down_stays_shut_down():
    exporter = _CollectingExporter()
    batch = _batch(exporter, shutdown_timeout_millis=5000)
    pending = _PendingSpan("parent")
    batch.on_end(pending)
    # The final flush waits for the pending span, so shutdown is in progress at the fork.
    stopper = threading.Thread(target=batch.shutdown)
    stopper.start()
    deadline = time.monotonic() + 5
    while not batch._shutdown and time.monotonic() < deadline:
        time.sleep(0.005)

    def child():
        batch.on_end(_ended_span("child"))
        batch.force_flush()
        batch.shutdown()
        return exporter.events

    try:
        # Neither exported nor shut down: the parent finishes both.
        assert _in_forked_child(child) == []
    finally:
        pending._ended = True
        batch.on_ended(pending)
        stopper.join(timeout=5)
    assert exporter.events == [("export", ["parent"]), ("shutdown", [])]


@needs_fork
def test_processor_created_in_a_forked_child_exports():
    def child():
        exporter = _CollectingExporter()
        batch = _batch(exporter)
        batch.on_end(_ended_span("child"))
        batch.shutdown()
        return exporter.events

    assert _in_forked_child(child) == [["export", ["child"]], ["shutdown", []]]


@needs_fork
def test_fork_during_an_export_keeps_one_worker_in_the_child():
    results = []

    class _ForkingExporter(_CollectingExporter):
        def export(self, spans):
            super().export(spans)
            if [span.name for span in spans] == ["fork"]:

                def child():
                    # This thread is the child's worker; no second worker may start.
                    is_worker = batch._worker is threading.current_thread()
                    batch.on_end(_ended_span("child"))
                    batch.force_flush()
                    return {"is_worker": is_worker, "exported": self.exported_names}

                results.append(_in_forked_child(child))
            return True

    exporter = _ForkingExporter()
    batch = _batch(exporter)
    batch.on_end(_ended_span("fork"))
    batch.force_flush(timeout=15)
    batch.shutdown()

    assert results == [{"is_worker": True, "exported": ["fork", "child"]}]


@needs_fork
def test_rate_limiter_admits_spans_in_a_child_forked_while_its_lock_is_held():
    limiter = RateLimitingSpanProcessor(next_processor=None, max_spans_per_second=1000)
    release = _holding(limiter.rate_limiter._lock)
    try:
        assert _in_forked_child(lambda: limiter.rate_limiter.acquire()) is True
    finally:
        release.set()


@needs_fork
def test_file_exporter_writes_in_a_child_forked_while_its_lock_is_held(tmp_path):
    path = tmp_path / "traces.jsonl"
    exporter = FileExporter(file_path=str(path))
    span = TracerProvider().get_tracer("test").start_span("child")
    span.end()
    release = _holding(exporter._lock)

    def child():
        return exporter.export([span])

    try:
        assert _in_forked_child(child) is True
    finally:
        release.set()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


@needs_fork
def test_span_end_in_a_child_forked_while_span_path_locks_are_held():
    from traccia.governance import evidence
    from traccia.processors.guardrail_detector import GuardrailDetectorProcessor

    provider = TracerProvider()
    guardrails = GuardrailDetectorProcessor()
    provider.add_span_processor(guardrails)
    exporter = _CollectingExporter()
    batch = _batch(exporter)
    provider.add_span_processor(batch)
    # Span.end() of a root span takes the evidence lock (forget_trace), the
    # guardrail detector takes its own, get_tracer() takes the provider's.
    releases = [_holding(evidence._lock), _holding(guardrails._lock), _holding(provider._lock)]

    def child():
        provider.get_tracer("child").start_span("child").end()
        batch.force_flush()
        return exporter.exported_names

    try:
        assert _in_forked_child(child) == ["child"]
    finally:
        for release in releases:
            release.set()
    batch.shutdown()
