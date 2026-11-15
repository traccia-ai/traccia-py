"""Optional governance hooks for agent runtimes."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from traccia.governance import schema as G


def _hash_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def disclosure(
    *,
    channel: str = "ui",
    disclosed_to_user: bool = True,
    synthetic_content: bool = False,
    generator: Optional[str] = None,
) -> None:
    """
    Record that the end user was informed they are interacting with AI (EU AI Act Art. 50).

    Call when your product shows an AI disclosure banner, label, or synthetic-media notice.
    Writes trace evidence only — it does not display UI. Human review belongs in Governance Hub.
    """
    from traccia.utils.helpers import effective_current_span

    span = effective_current_span()
    if not span or not span.is_recording():
        return
    span.set_attribute(G.EVENT_TYPE, "transparency")
    span.set_attribute(G.TRANSPARENCY_DISCLOSED, disclosed_to_user)
    span.set_attribute("governance.transparency.channel", channel)
    if synthetic_content:
        span.set_attribute(G.CONTENT_SYNTHETIC, True)
        if generator:
            span.set_attribute("governance.content.generator", generator)


def enrich_governance_attributes(
    attributes: Dict[str, Any],
    *,
    event_type: str = "inference",
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    session_id: Optional[str] = None,
    eu_risk_tier: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge governance attributes into a span attribute dict (hashes, model id, integrity hash)."""
    out = dict(attributes)
    out[G.EVENT_TYPE] = event_type
    out[G.TIMESTAMP_SOURCE] = "sdk"
    if model_id:
        out[G.MODEL_ID] = model_id
    if model_version:
        out[G.MODEL_VERSION] = model_version
    if session_id:
        out[G.SESSION_ID] = session_id
    ih = _hash_text(input_text)
    oh = _hash_text(output_text)
    if ih:
        out[G.INPUT_HASH] = ih
    if oh:
        out[G.OUTPUT_HASH] = oh
    if eu_risk_tier:
        out[G.RISK_TIER] = eu_risk_tier
    payload = f"{ih or ''}:{oh or ''}:{event_type}"
    out[G.INTEGRITY_HASH] = hashlib.sha256(payload.encode()).hexdigest()
    return out
