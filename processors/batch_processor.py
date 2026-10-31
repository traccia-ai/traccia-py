"""Batching span processor with bounded queue and background flush."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import deque
from typing import Deque, Iterable, List, Optional

from traccia.processors.drop_policy import DEFAULT_DROP_POLICY, DropPolicy
from traccia.processors.sampler import Sampler
from traccia.tracer.provider import SpanProcessor
from traccia.tracer.span import Span

logger = logging.getLogger(__name__)


class BatchSpanProcessor(SpanProcessor):
    """
    Batch span processor that queues spans for export.

    Span.end() runs enrichment processors (on_end) while the span is still
    mutable, marks it ended, then calls on_ended(). Spans are queued in
    on_ended(), so the queue only holds ended spans. Every export, including
    force_flush() and the final one at shutdown, runs on the worker thread.
    """

    def __init__(
        self,
        exporter=None,
        *,
        max_queue_size: int = 5000,
        max_export_batch_size: int = 512,
        schedule_delay_millis: int = 5000,
        drop_policy: Optional[DropPolicy] = None,
        sampler: Optional[Sampler] = None,
        shutdown_timeout_millis: int = 30_000,
    ) -> None:
        self.exporter = exporter
        self.max_queue_size = max_queue_size
        self.max_export_batch_size = max_export_batch_size
        self.schedule_delay = schedule_delay_millis / 1000.0
        self.drop_policy = drop_policy or DEFAULT_DROP_POLICY
        self.sampler = sampler
        self.shutdown_timeout = shutdown_timeout_millis / 1000.0

        self._queue: Deque[Span] = deque()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._shutdown = False
        self._shutdown_deadline = 0.0
        # Spans between on_end() and on_ended(); the final flush waits for them.
        self._ending: "weakref.WeakSet[Span]" = weakref.WeakSet()
        self._accepting = True
        self._flush_requests: List[threading.Event] = []
        self._worker_done = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def on_end(self, span: Span) -> None:
        """
        Called by Span.end() before it marks the span ended.

        The span is queued in on_ended(). A span that is already ended here
        (the OTel adapter notifies processors after ending) is queued now.
        """
        if getattr(span, "_ended", True):
            self._enqueue(span)
            return
        if not self._wants(span):
            return
        with self._lock:
            if self._accepting:
                self._ending.add(span)

    def on_ended(self, span: Span) -> None:
        """Called by Span.end() once the span is ended; queues it for export."""
        with self._lock:
            # One step, so the final flush never sees the span in neither place.
            self._ending.discard(span)
            self._enqueue_locked(span)

    def force_flush(self, timeout: Optional[float] = None) -> None:
        """Export the spans queued so far, waiting up to timeout seconds."""
        if threading.current_thread() is self._worker:
            self._export_pending()
            return
        done = threading.Event()
        with self._lock:
            if self._worker_done:
                return
            self._flush_requests.append(done)
        self._event.set()
        done.wait(timeout)

    def shutdown(self) -> None:
        """Export what is queued, then shut down the exporter. Safe to call more than once."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown_deadline = time.monotonic() + self.shutdown_timeout
            self._shutdown = True
        self._event.set()
        # The worker runs the final export, so the timeout bounds all of shutdown.
        self._worker.join(timeout=self.shutdown_timeout)
        if self._worker.is_alive():
            # Closing the exporter under a running export would lose its batch.
            logger.warning(
                "Span export still running after %.0fs; leaving the exporter open",
                self.shutdown_timeout,
            )
            return
        if self.exporter is not None and hasattr(self.exporter, "shutdown"):
            try:
                self.exporter.shutdown()
            except Exception:
                import traceback
                traceback.print_exc()

    # Internal
    def _wants(self, span: Span) -> bool:
        # Head-based sampling is recorded on SpanContext.trace_flags.
        # If a sampler is configured, traces marked as not-sampled (0) are dropped.
        if self.sampler and getattr(span.context, "trace_flags", 1) == 0:
            return False
        # Prevent double-queuing (on_end and on_ended both see the span)
        return not getattr(span, "_batch_queued", False)

    def _enqueue(self, span: Span) -> None:
        with self._lock:
            self._enqueue_locked(span)

    def _enqueue_locked(self, span: Span) -> None:
        if not self._accepting or not self._wants(span):
            return
        if self.drop_policy.handle(self._queue, span, self.max_queue_size):
            span._batch_queued = True
            self._event.set()

    def _worker_loop(self) -> None:
        """Background worker that exports spans periodically and on request."""
        while not self._shutdown:
            self._event.wait(timeout=self.schedule_delay)
            self._event.clear()
            self._export_pending()
        self._final_flush()

    def _final_flush(self) -> None:
        """Export everything, including spans that are still ending, until the deadline."""
        while True:
            self._export_pending()
            with self._lock:
                remaining = self._shutdown_deadline - time.monotonic()
                if not self._ending or remaining <= 0:
                    self._accepting = False
                    break
            # on_ended() sets the event when a span that was ending is queued.
            self._event.wait(timeout=min(remaining, 0.05))
            self._event.clear()
        self._export_pending()
        with self._lock:
            self._worker_done = True
            requests, self._flush_requests = self._flush_requests, []
        for request in requests:
            request.set()

    def _export_pending(self) -> None:
        """Export the spans queued so far, then release force_flush() callers."""
        with self._lock:
            requests, self._flush_requests = self._flush_requests, []
            remaining = len(self._queue)
        # Spans queued meanwhile wait for the next round, so this always returns.
        while remaining > 0:
            spans = self._drain_queue(min(remaining, self.max_export_batch_size))
            if not spans:
                break
            remaining -= len(spans)
            self._export(spans)
        for request in requests:
            request.set()

    def _drain_queue(self, limit: int) -> List[Span]:
        """Drain spans from queue up to limit."""
        items: List[Span] = []
        with self._lock:
            while self._queue and len(items) < limit:
                items.append(self._queue.popleft())
        return items

    def _export(self, spans: Iterable[Span]) -> None:
        """Export ended spans to exporter."""
        if self.exporter is None:
            return

        try:
            # Export spans - exporter will handle conversion if needed
            self.exporter.export(list(spans))
        except Exception as e:
            # Export errors are swallowed; resilience over strictness.
            import traceback
            traceback.print_exc()
            return
