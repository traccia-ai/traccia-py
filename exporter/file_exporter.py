"""File exporter for writing traces to a JSON file."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Iterable, List

from traccia.tracer.span import Span
from traccia.tracer.span import SpanStatus
from traccia import runtime_config


class FileExporter:
    """
    Exporter that writes spans to a JSON file in JSONL format.
    
    Each export() call writes one JSON object (containing resource and scopeSpans)
    per line, following the same OTLP-compatible JSON structure used by the SDK's
    network exporters.
    """

    def __init__(self, file_path: str = "traces.jsonl", reset_on_start: bool = False) -> None:
        """
        Initialize the file exporter.
        
        Args:
            file_path: Path to the file where traces will be written (default: "traces.jsonl")
            reset_on_start: If True, the file will be cleared on first export. 
                           If False, traces will be appended to the file.
        """
        self.file_path = file_path
        self.reset_on_start = reset_on_start
        self._lock = threading.Lock()
        self._first_export = True

    def export(self, spans: Iterable[Span]) -> bool:
        """
        Export spans to the file in JSONL format.
        
        Args:
            spans: Iterable of Span objects to export
            
        Returns:
            True if export succeeded, False otherwise
        """
        spans_list = list(spans)
        if not spans_list:
            return True

        try:
            payload = self._serialize(spans_list)
            # Convert bytes to string for JSONL format
            json_str = payload.decode("utf-8")
            
            with self._lock:
                # Handle reset_on_start: clear file on first export if True
                if self.reset_on_start and self._first_export:
                    mode = "w"
                    self._first_export = False
                else:
                    mode = "a"
                    if self._first_export:
                        self._first_export = False
                
                with open(self.file_path, mode, encoding="utf-8") as f:
                    f.write(json_str)
                    f.write("\n")  # JSONL format: one JSON object per line
            
            return True
        except Exception:
            # Silently fail on file write errors to avoid breaking the application
            return False

    def shutdown(self) -> None:
        """Shutdown the exporter. No cleanup needed for file-based exporter."""
        pass

    def _serialize(self, spans: List[Span]) -> bytes:
        """
        Serialize spans to JSON bytes using the same OTLP-compatible JSON structure
        used by the SDK's network exporters.
        """
        trunc = runtime_config.get_attr_truncation_limit()

        def _truncate_str(s: str) -> str:
            if trunc is None or trunc <= 0:
                return s
            if len(s) <= trunc:
                return s
            return s[: max(0, trunc - 1)] + "…"

        def _sanitize(value: Any, depth: int = 0) -> Any:
            # Keep payload JSON-safe and bounded.
            if value is None or isinstance(value, (bool, int, float)):
                return value
            if isinstance(value, str):
                return _truncate_str(value)
            if depth >= 6:
                return _truncate_str(repr(value))
            if isinstance(value, dict):
                out: Dict[str, Any] = {}
                for k, v in value.items():
                    try:
                        key = str(k)
                    except Exception:
                        key = "<unstringifiable>"
                    out[_truncate_str(key)] = _sanitize(v, depth + 1)
                return out
            if isinstance(value, (list, tuple, set)):
                return [_sanitize(v, depth + 1) for v in list(value)[:100]]
            # Fallback for unknown objects
            return _truncate_str(repr(value))

        def _status_code(status: SpanStatus) -> int:
            # OTLP-style: 0=UNSET, 1=OK, 2=ERROR
            if status == SpanStatus.OK:
                return 1
            if status == SpanStatus.ERROR:
                return 2
            return 0

        def to_event(ev):
            return {
                "name": ev.get("name"),
                "attributes": _sanitize(ev.get("attributes", {})),
                "timestamp_ns": ev.get("timestamp_ns"),
            }

        resource_attrs: Dict[str, Any] = {}
        try:
            if spans and getattr(spans[0], "tracer", None) is not None:
                provider = getattr(spans[0].tracer, "_provider", None)
                if provider is not None:
                    resource_attrs.update(getattr(provider, "resource", {}) or {})
        except Exception:
            resource_attrs = resource_attrs

        # Add optional session/user identifiers to resource for easier querying.
        if runtime_config.get_session_id():
            resource_attrs.setdefault("session.id", runtime_config.get_session_id())
        if runtime_config.get_user_id():
            resource_attrs.setdefault("user.id", runtime_config.get_user_id())
        if runtime_config.get_tenant_id():
            resource_attrs.setdefault("tenant.id", runtime_config.get_tenant_id())
        if runtime_config.get_project_id():
            resource_attrs.setdefault("project.id", runtime_config.get_project_id())
        if runtime_config.get_debug():
            resource_attrs.setdefault("trace.debug", True)

        payload = {
            "resource": {"attributes": _sanitize(resource_attrs)},
            "scopeSpans": [
                {
                    "scope": {"name": "agent-tracing-sdk", "version": "0.1.0"},
                    "spans": [
                        {
                            "traceId": span.context.trace_id,
                            "spanId": span.context.span_id,
                            "parentSpanId": span.parent_span_id,
                            "name": span.name,
                            "startTimeUnixNano": span.start_time_ns,
                            "endTimeUnixNano": span.end_time_ns,
                            "attributes": _sanitize(span.attributes),
                            "events": [to_event(e) for e in span.events],
                            "status": {
                                "code": _status_code(span.status),
                                "message": span.status_description or "",
                            },
                        }
                        for span in spans
                    ],
                }
            ]
        }
        # Ensure we never raise due to serialization edge cases.
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")
