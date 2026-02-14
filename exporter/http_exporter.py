"""HTTP exporter with retry, backoff, and graceful shutdown."""

from __future__ import annotations

import json
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from traccia.tracer.span import Span
from traccia.tracer.span import SpanStatus
from traccia import runtime_config

TransientStatus = {429, 503, 504}
DEFAULT_ENDPOINT = "https://api.traccia.ai/v2/traces"


class HttpExporter:
    """
    Push spans to a backend over HTTP with retry/backoff.

    A transport callable can be injected for testing. It should accept (payload_bytes, headers)
    and return an HTTP status code integer.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_jitter: float = 0.5,
        transport: Optional[Callable[[bytes, dict], int]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_jitter = backoff_jitter
        self._transport = transport or self._http_post

    def export(self, spans: Iterable[Span]) -> bool:
        spans_list = list(spans)
        if not spans_list:
            return True

        payload = self._serialize(spans_list)
        headers = self._headers()

        for attempt in range(self.max_retries):
            status = self._safe_send(payload, headers)
            if status is None:
                status = 503  # treat transport errors as transient

            if 200 <= status < 300:
                return True

            if status not in TransientStatus:
                return False

            sleep_for = self._compute_backoff(attempt)
            time.sleep(sleep_for)

        return False

    def shutdown(self) -> None:
        # No persistent connections when using urllib; if using requests Session it
        # could be closed here. Kept for API symmetry.
        pass

    # Internal helpers
    def _safe_send(self, payload: bytes, headers: dict) -> Optional[int]:
        try:
            return self._transport(payload, headers)
        except Exception:
            return None

    def _compute_backoff(self, attempt: int) -> float:
        base = self.backoff_base * (2 ** attempt)
        jitter = random.uniform(0, self.backoff_jitter)
        return base + jitter

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "agent-dashboard-sdk/0.1.0",
            "X-SDK-Version": "0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _serialize(self, spans: List[Span]) -> bytes:
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
        if runtime_config.get_agent_id():
            resource_attrs.setdefault("agent.id", runtime_config.get_agent_id())
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

    def _http_post(self, payload: bytes, headers: dict) -> int:
        # Lightweight stdlib HTTP client to avoid extra deps.
        # Run in empty context to prevent exporter's own HTTP calls from being instrumented
        import urllib.request
        from opentelemetry import context as context_api

        # Run HTTP call in an empty context (no active span) to prevent instrumentation
        # This ensures the exporter's HTTP calls don't create spans that pollute business traces
        empty_context = context_api.Context()
        token = context_api.attach(empty_context)
        
        try:
            req = urllib.request.Request(
                self.endpoint, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.getcode()
        except Exception:
            return None
        finally:
            context_api.detach(token)

