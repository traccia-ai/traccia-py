"""Span processor that adds default GovernanceEvent attributes on every span."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from traccia.governance import schema as G
from traccia.tracer.provider import SpanProcessor

_logger = logging.getLogger(__name__)
_hipaa_warned = False


class GovernanceEnrichmentProcessor(SpanProcessor):
    """Ensures governance.event_type and integrity_hash exist before export."""

    def __init__(
        self,
        *,
        default_event_type: str = "inference",
        eu_risk_tier: Optional[str] = None,
        hipaa_enabled: bool = False,
    ) -> None:
        self._default_event_type = default_event_type
        self._eu_risk_tier = eu_risk_tier
        self._hipaa_enabled = hipaa_enabled

    def on_end(self, span: Any) -> None:
        global _hipaa_warned
        if not hasattr(span, "set_attribute"):
            return
        attrs = dict(span.attributes) if hasattr(span, "attributes") else {}
        event_type = attrs.get(G.EVENT_TYPE)
        if not event_type:
            name = str(attrs.get("name", "") or getattr(span, "name", "") or "")
            event_type = "tool_call" if "tool" in name.lower() else self._default_event_type
            span.set_attribute(G.EVENT_TYPE, event_type)
        if G.TIMESTAMP_SOURCE not in attrs:
            span.set_attribute(G.TIMESTAMP_SOURCE, "sdk")
        if self._eu_risk_tier and G.RISK_TIER not in attrs:
            span.set_attribute(G.RISK_TIER, self._eu_risk_tier)
        if self._hipaa_enabled:
            if G.HIPAA_FRAMEWORK_ENABLED not in attrs:
                span.set_attribute(G.HIPAA_FRAMEWORK_ENABLED, True)
            if attrs.get(G.REDACTION_APPLIED) and G.HIPAA_PHI_REDACTION_APPLIED not in attrs:
                span.set_attribute(G.HIPAA_PHI_REDACTION_APPLIED, True)
            if not _hipaa_warned:
                _hipaa_warned = True
                _logger.warning(
                    "Traccia HIPAA framework enabled on spans. Prefer redact_pii=True. "
                    "Traccia Cloud does not currently offer a signed BAA — contact the Traccia team "
                    "for healthcare onboarding. Enabling this does not make you HIPAA compliant."
                )
        if G.INTEGRITY_HASH not in attrs:
            ctx = getattr(span, "context", None)
            span.set_attribute(
                G.INTEGRITY_HASH,
                integrity_fingerprint(
                    getattr(ctx, "trace_id", None),
                    getattr(ctx, "span_id", None),
                    event_type,
                ),
            )


def _canonical_id(value: Any, width: int) -> str:
    """Return an OTel ID as lowercase hex, zero-padded to ``width`` characters.

    Accepts the int form (OTel SDK), the hex-string form (Traccia ``SpanContext``,
    OTLP JSON, with or without ``0x``) and raw bytes (OTLP protobuf). A missing or
    unrecognised value gives ``""``.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return format(value, f"0{width}x") if value >= 0 else ""
    if isinstance(value, (bytes, bytearray)):
        return value.hex().zfill(width)
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            text = text[2:]
        if not text:
            return ""
        try:
            int(text, 16)
        except ValueError:
            return text
        return text.zfill(width)
    return ""


def integrity_fingerprint(trace_id: Any, span_id: Any, event_type: Any) -> str:
    """SDK span fingerprint, contract v1.

    ``sha256("<trace_id>:<span_id>:<event_type>")`` as lowercase hex, where the trace ID
    is 32 and the span ID 16 lowercase hex characters. This is the same payload the
    Node SDK and the ingestion backfill (``governance_normalizer``) hash, so a span gets
    the same value whichever side computes it. It identifies the span; it is not a
    digest of the span's content.
    """
    payload = f"{_canonical_id(trace_id, 32)}:{_canonical_id(span_id, 16)}:{event_type or ''}"
    return hashlib.sha256(payload.encode()).hexdigest()
