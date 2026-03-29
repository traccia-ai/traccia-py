"""Span processor that detects guardrails from span attributes and writes
findings directly onto spans so they flow through the OTel export pipeline.

Per-span findings are written as attributes on each span that produces them.
The aggregated summary (detected, missing, capabilities) is written onto the
root span when it ends -- by that point all child spans have already been
processed, so the full picture is available.

State is scoped by trace_id, making this safe for concurrent runs.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("traccia.guardrails.processor")

from traccia.guardrails.detectors import detect_all
from traccia.guardrails.evaluator import evaluate_run
from traccia.guardrails.schema import GuardrailFinding
from traccia.tracer.provider import SpanProcessor


@dataclass
class _TraceState:
    """Accumulated state for a single trace (one agent run)."""
    span_attrs: List[Dict[str, Any]] = field(default_factory=list)
    span_ids: List[Optional[str]] = field(default_factory=list)
    findings: List[GuardrailFinding] = field(default_factory=list)


class GuardrailDetectorProcessor(SpanProcessor):
    """Enrichment processor that detects guardrails and writes results to spans.

    Concurrent-safe: internal state is keyed by trace_id so parallel runs
    do not interfere with each other.

    Usage::

        processor = GuardrailDetectorProcessor()
        traccia.get_tracer_provider().add_span_processor(processor)
        # No further calls needed -- findings appear on exported spans.
    """

    def __init__(self, *, heuristics_enabled: bool = True) -> None:
        self._heuristics_enabled = heuristics_enabled
        self._lock = threading.Lock()
        self._traces: Dict[str, _TraceState] = {}

    def on_end(self, span: Any) -> None:
        """Called before OTel span.end(); span is still mutable."""
        try:
            attrs = dict(span.attributes) if hasattr(span, "attributes") else {}
            span_id: Optional[str] = None
            trace_id: Optional[str] = None
            if hasattr(span, "context"):
                span_id = getattr(span.context, "span_id", None)
                trace_id = getattr(span.context, "trace_id", None)

            if not trace_id:
                return

            findings = detect_all(
                attrs,
                trace_id=trace_id,
                span_id=span_id,
                heuristics_enabled=self._heuristics_enabled,
            )

            # Write per-span findings as attributes (flows through OTel export)
            if findings:
                span.set_attribute("guardrail.finding.count", len(findings))
                try:
                    span.set_attribute(
                        "guardrail.findings",
                        json.dumps([f.to_dict() for f in findings]),
                    )
                except Exception:
                    pass

            # Accumulate into trace-scoped state
            with self._lock:
                state = self._traces.get(trace_id)
                if state is None:
                    state = _TraceState()
                    self._traces[trace_id] = state
                state.span_attrs.append(attrs)
                state.span_ids.append(span_id)
                state.findings.extend(findings)

            # If this is the root span, compute and write the aggregated summary
            is_root = getattr(span, "parent_span_id", "not_none") is None
            if is_root:
                self._write_summary(span, trace_id)

        except Exception as exc:
            # WARNING keeps failures visible without crashing export; full traceback at DEBUG
            _logger.warning(
                "GuardrailDetectorProcessor.on_end failed (trace export continues): %s",
                exc,
            )
            _logger.debug("GuardrailDetectorProcessor.on_end traceback", exc_info=True)

    def _write_summary(self, span: Any, trace_id: str) -> None:
        """Compute the full guardrail summary and write it onto the root span."""
        with self._lock:
            state = self._traces.pop(trace_id, None)

        if state is None:
            return

        try:
            summary = evaluate_run(
                all_span_attrs=state.span_attrs,
                span_ids=state.span_ids,
                findings=state.findings,
                trace_id=trace_id,
                heuristics_enabled=self._heuristics_enabled,
            )

            summary_dict = summary.to_dict()
            span.set_attribute("guardrail.summary", json.dumps(summary_dict))
            span.set_attribute(
                "guardrail.summary.detected_categories",
                summary_dict.get("detected_categories", []),
            )
            span.set_attribute(
                "guardrail.summary.missing_count",
                len(summary_dict.get("missing_categories", [])),
            )
            span.set_attribute(
                "guardrail.summary.coverage_confidence",
                summary_dict.get("coverage_confidence", "low"),
            )

            # Also write the full findings list on the root span for easy access
            if state.findings:
                span.set_attribute(
                    "guardrail.findings",
                    json.dumps([f.to_dict() for f in state.findings]),
                )
                span.set_attribute("guardrail.finding.count", len(state.findings))

        except Exception as exc:
            _logger.warning(
                "GuardrailDetectorProcessor._write_summary failed (trace export continues): %s",
                exc,
            )
            _logger.debug("GuardrailDetectorProcessor._write_summary traceback", exc_info=True)

    def shutdown(self) -> None:
        with self._lock:
            self._traces.clear()

    def force_flush(self, timeout: Optional[float] = None) -> None:
        pass
