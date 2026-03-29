"""Guardrail detectors that extract findings from span attributes.

Each detector examines a dict of span attributes and produces zero or more
GuardrailFinding instances.  Detectors are stateless functions; aggregation
and deduplication happen in the processor layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from traccia.guardrails.constants import (
    ATTR_AGENT_GUARDRAIL_NAME,
    ATTR_AGENT_GUARDRAIL_TRIGGERED,
    ATTR_AGENT_SPAN_TYPE,
    ATTR_GUARDRAIL_CATEGORY,
    ATTR_GUARDRAIL_ENFORCEMENT_MODE,
    ATTR_GUARDRAIL_NAME,
    ATTR_GUARDRAIL_TRIGGERED,
    ATTR_LLM_COMPLETION,
    ATTR_LLM_FINISH_REASON,
    ATTR_LLM_MODEL,
    ATTR_LLM_PROMPT,
    ATTR_LLM_RESPONSE_STATUS,
    ATTR_LLM_SAFETY_RATINGS,
    ATTR_LLM_STOP_REASON,
    ATTR_LLM_VENDOR,
    ATTR_SPAN_TYPE,
    ATTR_ERROR_TYPE,
    ATTR_ERROR_MESSAGE,
)
from traccia.guardrails.schema import (
    Confidence,
    EnforcementMode,
    EvidenceRef,
    GuardrailCategory,
    GuardrailFinding,
    SourceType,
)


# ---------------------------------------------------------------------------
# Explicit detector (Tier A) -- OpenAI Agents + manual observe
# ---------------------------------------------------------------------------


def detect_explicit(
    attrs: Dict[str, Any],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> List[GuardrailFinding]:
    """Detect guardrails from explicit typed spans (OpenAI Agents or manual observe)."""
    findings: List[GuardrailFinding] = []

    # OpenAI Agents SDK guardrail spans
    agent_span_type = attrs.get(ATTR_AGENT_SPAN_TYPE)
    if agent_span_type == "guardrail":
        name = attrs.get(ATTR_AGENT_GUARDRAIL_NAME) or attrs.get(ATTR_GUARDRAIL_NAME) or "unnamed_guardrail"
        triggered = attrs.get(ATTR_AGENT_GUARDRAIL_TRIGGERED)
        if triggered is not None:
            triggered = bool(triggered)
        category_raw = attrs.get(ATTR_GUARDRAIL_CATEGORY, "unknown")
        try:
            category = GuardrailCategory(category_raw)
        except ValueError:
            category = GuardrailCategory.UNKNOWN

        findings.append(GuardrailFinding(
            category=category,
            name=str(name),
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            triggered=triggered,
            enforcement_mode=EnforcementMode.UNKNOWN,
            detection_reason="openai_agents_guardrail_span",
            evidence_ref=EvidenceRef(
                trace_id=trace_id,
                span_id=span_id,
                integration="openai_agents",
                attribute_keys=[ATTR_AGENT_SPAN_TYPE, ATTR_AGENT_GUARDRAIL_NAME, ATTR_AGENT_GUARDRAIL_TRIGGERED],
            ),
        ))

    # Manual observe(as_type="guardrail") spans
    span_type = attrs.get(ATTR_SPAN_TYPE)
    if span_type == "guardrail" and agent_span_type != "guardrail":
        name = attrs.get(ATTR_GUARDRAIL_NAME) or "unnamed_guardrail"
        triggered = attrs.get(ATTR_GUARDRAIL_TRIGGERED)
        if triggered is not None:
            triggered = bool(triggered)

        category_raw = attrs.get(ATTR_GUARDRAIL_CATEGORY, "unknown")
        try:
            category = GuardrailCategory(category_raw)
        except ValueError:
            category = GuardrailCategory.UNKNOWN

        enforcement_raw = attrs.get(ATTR_GUARDRAIL_ENFORCEMENT_MODE, "unknown")
        try:
            enforcement = EnforcementMode(enforcement_raw)
        except ValueError:
            enforcement = EnforcementMode.UNKNOWN

        has_required = all(
            attrs.get(k) is not None
            for k in (ATTR_GUARDRAIL_NAME, ATTR_GUARDRAIL_CATEGORY, ATTR_GUARDRAIL_TRIGGERED)
        )
        confidence = Confidence.HIGH if has_required else Confidence.MEDIUM

        findings.append(GuardrailFinding(
            category=category,
            name=str(name),
            source_type=SourceType.EXPLICIT,
            confidence=confidence,
            triggered=triggered,
            enforcement_mode=enforcement,
            detection_reason="manual_observe_guardrail_span",
            evidence_ref=EvidenceRef(
                trace_id=trace_id,
                span_id=span_id,
                integration="manual_observe",
                attribute_keys=[ATTR_SPAN_TYPE, ATTR_GUARDRAIL_NAME, ATTR_GUARDRAIL_CATEGORY],
            ),
        ))

    return findings


# ---------------------------------------------------------------------------
# Provider-native detector (Tier B) -- OpenAI / LangChain structured signals
# ---------------------------------------------------------------------------

# finish_reason values that indicate provider-enforced content blocking.
# OpenAI: "content_filter"
# Azure OpenAI: "content_filtered"
# Google GenAI (direct): "SAFETY"
_REFUSAL_FINISH_REASONS = frozenset({"content_filter", "content_filtered", "SAFETY"})

# Anthropic stop_reason values that indicate content-policy blocking.
_ANTHROPIC_BLOCK_STOP_REASONS = frozenset({"content_filtered", "content_filter"})

# Anthropic error message substrings that indicate a policy violation.
_ANTHROPIC_POLICY_PHRASES = ("content policy", "violates anthropic's usage policy", "usage policy violation")


def _anthropic_vendor(attrs: Dict[str, Any]) -> bool:
    v = str(attrs.get(ATTR_LLM_VENDOR, "")).lower()
    return v in ("anthropic", "claude")


def _looks_like_llm_span(attrs: Dict[str, Any]) -> bool:
    """True if span has LLM call signals (not a random error-only span)."""
    if attrs.get(ATTR_SPAN_TYPE) == "llm":
        return True
    return any(
        attrs.get(k) is not None
        for k in (
            ATTR_LLM_MODEL,
            ATTR_LLM_PROMPT,
            ATTR_LLM_COMPLETION,
            ATTR_LLM_FINISH_REASON,
            ATTR_LLM_STOP_REASON,
        )
    )


def detect_provider_native(
    attrs: Dict[str, Any],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> List[GuardrailFinding]:
    """Detect guardrails from provider-native structured fields."""
    findings: List[GuardrailFinding] = []

    # --- OpenAI / Azure / Google: llm.finish_reason ---
    finish_reason = attrs.get(ATTR_LLM_FINISH_REASON)
    if finish_reason and str(finish_reason) in _REFUSAL_FINISH_REASONS:
        vendor = str(attrs.get(ATTR_LLM_VENDOR, "")).lower()
        integration = "azure_openai" if vendor == "azure" else ("google" if vendor in ("google", "gemini") else "openai")
        findings.append(GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="provider_content_filter",
            source_type=SourceType.PROVIDER_NATIVE,
            confidence=Confidence.HIGH,
            triggered=True,
            enforcement_mode=EnforcementMode.BLOCK,
            detection_reason="llm_finish_reason_content_filter",
            evidence_ref=EvidenceRef(
                trace_id=trace_id,
                span_id=span_id,
                integration=integration,
                attribute_keys=[ATTR_LLM_FINISH_REASON],
            ),
            raw_excerpt=str(finish_reason),
        ))

    # --- Anthropic: llm.stop_reason (only when vendor is Anthropic — avoids
    # false positives if another stack reuses the same stop_reason strings)
    stop_reason = attrs.get(ATTR_LLM_STOP_REASON)
    vendor = str(attrs.get(ATTR_LLM_VENDOR, "")).lower()
    if (
        stop_reason
        and str(stop_reason) in _ANTHROPIC_BLOCK_STOP_REASONS
        and _anthropic_vendor(attrs)
    ):
        findings.append(GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="anthropic_content_filter",
            source_type=SourceType.PROVIDER_NATIVE,
            confidence=Confidence.HIGH,
            triggered=True,
            enforcement_mode=EnforcementMode.BLOCK,
            detection_reason="llm_stop_reason_content_filter",
            evidence_ref=EvidenceRef(
                trace_id=trace_id,
                span_id=span_id,
                integration="anthropic",
                attribute_keys=[ATTR_LLM_STOP_REASON],
            ),
            raw_excerpt=str(stop_reason),
        ))

    # --- Anthropic: error.message with policy violation text (LLM span only)
    error_msg = str(attrs.get(ATTR_ERROR_MESSAGE, "")).lower()
    if _anthropic_vendor(attrs) and error_msg and _looks_like_llm_span(attrs):
        if any(phrase in error_msg for phrase in _ANTHROPIC_POLICY_PHRASES):
            findings.append(GuardrailFinding(
                category=GuardrailCategory.MODERATION,
                name="anthropic_policy_violation",
                source_type=SourceType.PROVIDER_NATIVE,
                confidence=Confidence.MEDIUM,
                triggered=True,
                enforcement_mode=EnforcementMode.BLOCK,
                detection_reason="anthropic_error_message_policy_phrase",
                evidence_ref=EvidenceRef(
                    trace_id=trace_id,
                    span_id=span_id,
                    integration="anthropic",
                    attribute_keys=[ATTR_ERROR_MESSAGE, ATTR_LLM_VENDOR],
                ),
                raw_excerpt=error_msg[:200],
            ))

    # --- Google / LangChain: llm.safety_ratings (JSON string) ---
    safety_ratings_raw = attrs.get(ATTR_LLM_SAFETY_RATINGS)
    if safety_ratings_raw:
        blocked = _parse_safety_ratings_blocked(str(safety_ratings_raw))
        if blocked:
            findings.append(GuardrailFinding(
                category=GuardrailCategory.MODERATION,
                name="google_safety_ratings_blocked",
                source_type=SourceType.PROVIDER_NATIVE,
                confidence=Confidence.MEDIUM,
                triggered=True,
                enforcement_mode=EnforcementMode.BLOCK,
                detection_reason="llm_safety_ratings_high_or_blocked",
                evidence_ref=EvidenceRef(
                    trace_id=trace_id,
                    span_id=span_id,
                    integration="google_langchain",
                    attribute_keys=[ATTR_LLM_SAFETY_RATINGS],
                ),
                raw_excerpt=str(safety_ratings_raw)[:200],
            ))

    # --- Refusal signal from response status + refusal text ---
    response_status = attrs.get(ATTR_LLM_RESPONSE_STATUS)
    if response_status and str(response_status).lower() in ("incomplete", "failed"):
        # Only emit if we also see a refusal-like pattern, not generic errors
        completion = attrs.get(ATTR_LLM_COMPLETION, "") or ""
        if _looks_like_refusal(str(completion)):
            findings.append(GuardrailFinding(
                category=GuardrailCategory.MODERATION,
                name="provider_refusal",
                source_type=SourceType.PROVIDER_NATIVE,
                confidence=Confidence.MEDIUM,
                triggered=True,
                enforcement_mode=EnforcementMode.BLOCK,
                detection_reason="llm_response_status_with_refusal_text",
                evidence_ref=EvidenceRef(
                    trace_id=trace_id,
                    span_id=span_id,
                    integration="openai",
                    attribute_keys=[ATTR_LLM_RESPONSE_STATUS, ATTR_LLM_COMPLETION],
                ),
                raw_excerpt=str(completion)[:200],
            ))

    return findings


def _parse_safety_ratings_blocked(ratings_json: str) -> bool:
    """Return True if any Google safety rating is HIGH probability or blocked."""
    import json
    try:
        ratings = json.loads(ratings_json)
        if not isinstance(ratings, list):
            ratings = [ratings]
        for entry in ratings:
            if not isinstance(entry, dict):
                continue
            if entry.get("blocked") is True:
                return True
            if str(entry.get("probability", "")).upper() == "HIGH":
                return True
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return False


_REFUSAL_PHRASES = (
    "i can't help with",
    "i cannot help with",
    "i'm not able to",
    "i am not able to",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "against my guidelines",
    "violates my usage policies",
    "content policy",
    "i must decline",
)


def _looks_like_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Heuristic detector (Tier C) -- conservative pattern matching
# ---------------------------------------------------------------------------

_HEURISTIC_ENABLED = True

# Error types that commonly produce denial-sounding messages but are NOT
# permission/policy violations.  These are excluded to reduce false positives.
_NON_DENIAL_ERROR_TYPES = frozenset({
    "TimeoutError",
    "asyncio.TimeoutError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "OSError",
    "socket.timeout",
    "requests.exceptions.Timeout",
    "requests.exceptions.ConnectionError",
    "httpx.TimeoutException",
    "httpx.ConnectError",
})


def detect_heuristic(
    attrs: Dict[str, Any],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    enabled: bool = _HEURISTIC_ENABLED,
) -> List[GuardrailFinding]:
    """Detect guardrails from heuristic signals (always marked low confidence)."""
    if not enabled:
        return []

    findings: List[GuardrailFinding] = []

    # Heuristic: error with denial-like message on a tool span
    span_type = attrs.get(ATTR_SPAN_TYPE) or attrs.get(ATTR_AGENT_SPAN_TYPE, "")
    error_type = str(attrs.get(ATTR_ERROR_TYPE, ""))
    error_msg = str(attrs.get(ATTR_ERROR_MESSAGE, "")).lower()

    if span_type in ("tool", "function") and error_type:
        # Skip if the error type is a known non-denial error (timeout, network, etc.)
        # to avoid false positives from infrastructure failures.
        if error_type not in _NON_DENIAL_ERROR_TYPES:
            denial_keywords = ("permission", "denied", "unauthorized", "forbidden", "not allowed")
            if any(kw in error_msg for kw in denial_keywords):
                findings.append(GuardrailFinding(
                    category=GuardrailCategory.TOOL_PERMISSION,
                    name="inferred_tool_denial",
                    source_type=SourceType.HEURISTIC,
                    confidence=Confidence.LOW,
                    triggered=True,
                    enforcement_mode=EnforcementMode.UNKNOWN,
                    detection_reason="tool_span_error_with_denial_keywords",
                    evidence_ref=EvidenceRef(
                        trace_id=trace_id,
                        span_id=span_id,
                        integration="heuristic",
                        attribute_keys=[ATTR_ERROR_TYPE, ATTR_ERROR_MESSAGE],
                    ),
                    raw_excerpt=error_msg[:200],
                ))

    return findings


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------


def detect_all(
    attrs: Dict[str, Any],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    heuristics_enabled: bool = True,
) -> List[GuardrailFinding]:
    """Run all detection tiers on a span's attributes and return findings."""
    results: List[GuardrailFinding] = []
    results.extend(detect_explicit(attrs, trace_id, span_id))
    results.extend(detect_provider_native(attrs, trace_id, span_id))
    results.extend(detect_heuristic(attrs, trace_id, span_id, enabled=heuristics_enabled))
    return results
