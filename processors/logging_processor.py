"""Span processor that logs spans when they end."""

from __future__ import annotations

import logging
from typing import Optional

from traccia.tracer.provider import SpanProcessor


class LoggingSpanProcessor(SpanProcessor):
    """Logs span summary on end using the standard logging module."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("traccia.traces")

    def on_end(self, span) -> None:
        # Logging is an output path: skip spans the sampler dropped.
        if getattr(span.context, "trace_flags", 1) == 0:
            return
        attrs = span.attributes or {}
        msg = (
            f"[trace] name={span.name} trace_id={span.context.trace_id} "
            f"span_id={span.context.span_id} status={span.status.name} "
            f"duration_ns={span.duration_ns} attrs={attrs}"
        )
        self.logger.info(msg)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout: Optional[float] = None) -> None:
        return None

