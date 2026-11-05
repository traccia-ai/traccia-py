"""Batching span processor with bounded queue and background flush."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from typing import Deque, Dict, List, Optional

from traccia.processors.drop_policy import DEFAULT_DROP_POLICY, DropPolicy
from traccia.processors.sampler import Sampler
from traccia.tracer.provider import SpanProcessor
from traccia.tracer.span import Span
from traccia.utils.fork import reinit_after_fork

logger = logging.getLogger(__name__)

_UNTRACKED = object()


class BatchSpanProcessor(SpanProcessor):
    """
    Batch span processor that queues spans for export.

    Span.end() runs enrichment processors (on_end) while the span is still
    mutable, marks it ended, then calls on_ended(). Spans are queued once
    ended: in on_ended(), or by the worker's next round for a span whose
    on_ended() never comes (a wrapper that forwards only on_end()), so the
    queue only holds ended spans. Every export, including force_flush() and
    the final one at shutdown, runs on the worker thread, and so does the
    exporter's own shutdown(); if the worker has died, the caller exports
    instead. Exports never overlap.
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
        # Held for each export, so shutdown can tell whether one is running.
        self._export_lock = threading.RLock()
        self._shutdown = False
        self._shutdown_deadline = 0.0
        # Spans between on_end() and being queued, in order, tracked until
        # shutdown starts; the final flush waits for them. Held strongly: a
        # span forwarded only through on_end() may have no other reference.
        self._ending: Dict[Span, None] = {}
        self._accepting = True
        # Set once shutdown is closing the exporter: no export starts after it.
        self._closing = False
        self._dropped_at_close = 0
        # A shutdown() caller finishes the final flush if the worker died before it.
        self._taken_over = False
        # Set when shutdown's work is done; later shutdown() callers wait for it.
        self._shutdown_done = threading.Event()
        self._flush_requests: List[threading.Event] = []
        self._worker_done = False
        self._final_flush_done = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        reinit_after_fork(self._at_fork_reinit)

    def on_end(self, span: Span) -> None:
        """
        Called by Span.end() before it marks the span ended.

        The span is queued once it is ended. A span that is already ended here
        (the OTel adapter notifies processors after ending) is queued now.
        """
        if getattr(span, "_ended", True):
            self._enqueue(span)
            return
        if not self._wants(span):
            return
        with self._lock:
            # Shutdown waits only for spans that were ending when it started.
            if not self._shutdown:
                self._ending[span] = None

    def on_ended(self, span: Span) -> None:
        """Called by Span.end() once the span is ended; queues it for export."""
        with self._lock:
            # One step, so the final flush never sees the span in neither place.
            tracked = self._ending.pop(span, _UNTRACKED) is not _UNTRACKED
            self._enqueue_locked(span, tracked)

    def force_flush(self, timeout: Optional[float] = None) -> None:
        """Export the spans queued so far, waiting up to timeout seconds (None or 0: no limit)."""
        if threading.current_thread() is self._worker:
            self._export_pending()
            return
        done = threading.Event()
        with self._lock:
            worker_done = self._worker_done
            if not worker_done:
                self._flush_requests.append(done)
        if not worker_done:
            self._event.set()
            done.wait(timeout or None)
        if self._worker_done and not self._shutdown:
            # The worker died (a BaseException outside an export); export here.
            self._export_pending()

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Export what is queued, then shut down the exporter. Safe to call more than once.

        Args:
            timeout: Seconds to wait for the final export and the exporter's
                own shutdown (default shutdown_timeout_millis). Unlike
                force_flush(timeout=0), which has no limit, shutdown(timeout=0)
                does not wait.
        """
        on_worker = threading.current_thread() is self._worker
        with self._lock:
            already = self._shutdown
            if not already:
                self._shutdown_deadline = time.monotonic() + (
                    self.shutdown_timeout if timeout is None else timeout
                )
                self._shutdown = True
        if already:
            if not on_worker:
                # Shutdown is under way (another thread, or the worker itself from
                # inside an export): wait for it too, so the process does not exit
                # in the middle of the final flush.
                wait = self._remaining() if timeout is None else min(timeout, self._remaining())
                deadline = time.monotonic() + wait
                self._shutdown_done.wait(wait)
                self._take_over_from_dead_worker(deadline)
            return
        if timeout is None:
            timeout = self.shutdown_timeout
        self._event.set()
        if on_worker:
            # Called from inside an export (e.g. an exporter giving up): the worker
            # cannot join itself. Once this export returns, it runs the final flush
            # and closes the exporter.
            return
        try:
            self._shutdown_first(timeout)
        finally:
            self._shutdown_done.set()

    def _shutdown_first(self, timeout: float) -> None:
        # The worker runs the final export, then shuts the exporter down, so the
        # timeout bounds all of shutdown.
        self._worker.join(timeout=self._remaining())
        if not self._worker.is_alive():
            self._take_over_from_dead_worker(self._shutdown_deadline)
            return
        # Past the deadline. If no export is running, set _closing so none starts:
        # the worker drops what is left and shuts the exporter down.
        if not self._export_lock.acquire(blocking=False):
            # Closing the exporter under a running export would lose its batch;
            # the worker exports what is queued, then closes it.
            logger.warning(
                "Span export still running after %gs; the exporter is closed "
                "once the queued spans are exported",
                timeout,
            )
            return
        try:
            with self._lock:
                self._closing = True
                closing_exporter = self._final_flush_done
        finally:
            self._export_lock.release()
        if closing_exporter:
            logger.warning(
                "Exporter shutdown still running after %gs; it finishes in the background",
                timeout,
            )

    def _take_over_from_dead_worker(self, deadline: float) -> None:
        """Finish shutdown, by the deadline, if the worker died before its final flush."""
        with self._lock:
            take_over = (
                self._worker_done and not self._final_flush_done and not self._taken_over
            )
            if take_over:
                self._taken_over = True
        if not take_over:
            return
        # Nothing else exports once the worker is gone. A thread of its own does
        # the worker's last steps, so the deadline still bounds them.
        finisher = threading.Thread(target=self._final_flush_and_close, daemon=True)
        finisher.start()
        finisher.join(timeout=max(0.0, deadline - time.monotonic()))

    def _final_flush_and_close(self) -> None:
        self._final_flush()
        with self._export_lock:
            with self._lock:
                self._closing = True
        self._close_exporter()
        self._shutdown_done.set()

    def _close_exporter(self) -> None:
        if self.exporter is not None and hasattr(self.exporter, "shutdown"):
            try:
                self.exporter.shutdown()
            except Exception:
                traceback.print_exc()

    # Internal
    def _at_fork_reinit(self) -> None:
        """
        Start over in a child process after os.fork().

        The child has no worker, and the locks or event may have been held by a
        thread that does not exist there. Spans queued or ending in the parent
        are the parent's to export, so the child drops them, along with the
        parent's force_flush() waiters.
        """
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._export_lock = threading.RLock()
        self._queue = deque()
        self._ending = {}
        self._flush_requests = []
        self._dropped_at_close = 0
        self._taken_over = False
        self._shutdown_done = threading.Event()
        if self._shutdown:
            # Shut down, or shutting down, in the parent: the child stays shut down,
            # and leaves the final flush and the exporter to the parent.
            self._accepting = False
            self._worker_done = True
            self._taken_over = True
            self._shutdown_done.set()
            return
        if threading.current_thread() is self._worker:
            # Forked during an export: this thread carries on as the child's worker.
            return
        self._worker_done = False
        self._final_flush_done = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _remaining(self) -> float:
        return max(0.0, self._shutdown_deadline - time.monotonic())

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

    def _enqueue_locked(self, span: Span, tracked: bool = False) -> None:
        # Once shutdown starts, only spans it waits for are queued, so a steady
        # stream of new spans cannot hold the final export until the deadline.
        if not self._accepting or (self._shutdown and not tracked) or not self._wants(span):
            return
        if self.drop_policy.handle(self._queue, span, self.max_queue_size):
            span._batch_queued = True
            self._event.set()

    def _queue_ended_locked(self) -> None:
        """
        Queue tracked spans that ended without on_ended(), which never comes
        through a wrapper that forwards only on_end(); stop tracking spans
        whose end() was interrupted.
        """
        for span in list(self._ending):
            if getattr(span, "_ended", False):
                del self._ending[span]
                self._enqueue_locked(span, tracked=True)
            elif getattr(span, "_end_interrupted", False):
                del self._ending[span]

    def _worker_loop(self) -> None:
        """Background worker that exports spans periodically and on request."""
        try:
            while not self._shutdown:
                self._event.wait(timeout=self.schedule_delay)
                self._event.clear()
                self._export_pending()
            self._final_flush_and_close()
        finally:
            # Also if the worker dies: callers stop waiting for it and export themselves.
            with self._lock:
                self._worker_done = True
                shutting_down = self._shutdown
                requests, self._flush_requests = self._flush_requests, []
            for request in requests:
                request.set()
            if shutting_down:
                # Closed above, or the worker died first: stop shutdown() callers waiting.
                self._shutdown_done.set()

    def _final_flush(self) -> None:
        """Export everything, including spans that are still ending, until the deadline."""
        while True:
            self._export_pending()
            with self._lock:
                self._queue_ended_locked()
                remaining = self._shutdown_deadline - time.monotonic()
                if not self._ending or remaining <= 0:
                    self._accepting = False
                    self._ending.clear()
                    break
            # on_ended() sets the event when a span that was ending is queued.
            self._event.wait(timeout=min(remaining, 0.05))
            self._event.clear()
        self._export_pending()
        with self._lock:
            self._final_flush_done = True
            dropped, self._dropped_at_close = self._dropped_at_close, 0
            requests, self._flush_requests = self._flush_requests, []
        for request in requests:
            request.set()
        if dropped:
            logger.warning("Dropped %d queued spans: shutdown timed out and closed the exporter", dropped)

    def _export_pending(self) -> None:
        """Export the spans queued so far, then release force_flush() callers."""
        with self._lock:
            self._queue_ended_locked()
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

    def _export(self, spans: List[Span]) -> None:
        """Export ended spans to exporter."""
        if self.exporter is None:
            return

        with self._export_lock:
            with self._lock:
                if self._closing:
                    # Shutdown timed out between exports and closed the exporter.
                    self._dropped_at_close += len(spans)
                    return
            try:
                # Export spans - exporter will handle conversion if needed
                self.exporter.export(list(spans))
            except Exception:
                # Export errors are swallowed; resilience over strictness.
                traceback.print_exc()
            except BaseException:
                # CancelledError, gevent.Timeout, SystemExit: the worker keeps
                # running. A caller exporting in its place gets it back.
                if threading.current_thread() is not self._worker:
                    raise
                traceback.print_exc()
