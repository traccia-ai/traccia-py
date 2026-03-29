"""Guardrail detection for Traccia traces."""

from traccia.guardrails.schema import (
    GuardrailCategory,
    SourceType,
    Confidence,
    EnforcementMode,
    FindingStatus,
    EvidenceRef,
    GuardrailFinding,
    MissingGuardrail,
    GuardrailSummary,
    dedupe_findings,
)
from traccia.guardrails.constants import (
    ATTR_GUARDRAIL_CATEGORY,
    ATTR_GUARDRAIL_NAME,
    ATTR_GUARDRAIL_TRIGGERED,
    ATTR_GUARDRAIL_ENFORCEMENT_MODE,
    ATTR_GUARDRAIL_POLICY_ID,
    ATTR_GUARDRAIL_SOURCE_SDK,
    ATTR_GUARDRAIL_EVIDENCE_TYPE,
    ATTR_GUARDRAIL_SUPPRESS_MISSING,
)
from traccia.guardrails.helpers import (
    guardrail_span,
    validate_guardrail_attributes,
)

__all__ = [
    "GuardrailCategory",
    "SourceType",
    "Confidence",
    "EnforcementMode",
    "FindingStatus",
    "EvidenceRef",
    "GuardrailFinding",
    "MissingGuardrail",
    "GuardrailSummary",
    "dedupe_findings",
    "guardrail_span",
    "validate_guardrail_attributes",
    "ATTR_GUARDRAIL_CATEGORY",
    "ATTR_GUARDRAIL_NAME",
    "ATTR_GUARDRAIL_TRIGGERED",
    "ATTR_GUARDRAIL_ENFORCEMENT_MODE",
    "ATTR_GUARDRAIL_POLICY_ID",
    "ATTR_GUARDRAIL_SOURCE_SDK",
    "ATTR_GUARDRAIL_EVIDENCE_TYPE",
    "ATTR_GUARDRAIL_SUPPRESS_MISSING",
]
