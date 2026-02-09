"""Metrics module for OpenTelemetry-based metrics emission."""

from __future__ import annotations

from .metrics import StandardMetrics, MetricsRecorder
from .recorder import get_metrics_recorder, record_counter, record_histogram

__all__ = [
    "StandardMetrics",
    "MetricsRecorder",
    "get_metrics_recorder",
    "record_counter",
    "record_histogram",
]
