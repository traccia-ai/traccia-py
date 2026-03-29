"""Canonical guardrail detection schema for Traccia traces.

Defines the normalized data contract for guardrail findings, missing-guardrail
evaluations, and per-run summaries. All detectors must produce GuardrailFinding
instances; the summary is computed from aggregated findings + capability inference.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class GuardrailCategory(str, Enum):
    INPUT_VALIDATION = "input_validation"
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    MODERATION = "moderation"
    TOOL_PERMISSION = "tool_permission"
    OUTPUT_VALIDATION = "output_validation"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    EXPLICIT = "explicit"
    PROVIDER_NATIVE = "provider_native"
    HEURISTIC = "heuristic"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EnforcementMode(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    LOG_ONLY = "log_only"
    UNKNOWN = "unknown"


class FindingStatus(str, Enum):
    PRESENT = "present"
    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"


@dataclass
class EvidenceRef:
    """Pointer back to the trace/span data that produced a finding."""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    integration: Optional[str] = None
    attribute_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.trace_id is not None:
            d["trace_id"] = self.trace_id
        if self.span_id is not None:
            d["span_id"] = self.span_id
        if self.integration is not None:
            d["integration"] = self.integration
        if self.attribute_keys:
            d["attribute_keys"] = self.attribute_keys
        return d


@dataclass
class GuardrailFinding:
    """A single detected guardrail signal from trace data."""
    category: GuardrailCategory
    name: str
    source_type: SourceType
    confidence: Confidence
    triggered: Optional[bool] = None
    enforcement_mode: EnforcementMode = EnforcementMode.UNKNOWN
    detection_reason: str = ""
    evidence_ref: EvidenceRef = field(default_factory=EvidenceRef)
    raw_excerpt: Optional[str] = None

    @property
    def id(self) -> str:
        """Stable fingerprint for deduplication."""
        key = "|".join([
            self.evidence_ref.trace_id or "",
            self.source_type.value,
            self.category.value,
            self.name,
            self.evidence_ref.span_id or "",
        ])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def status(self) -> FindingStatus:
        if self.triggered is True:
            return FindingStatus.TRIGGERED
        elif self.triggered is False:
            return FindingStatus.NOT_TRIGGERED
        return FindingStatus.PRESENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "name": self.name,
            "source_type": self.source_type.value,
            "confidence": self.confidence.value,
            "triggered": self.triggered,
            "enforcement_mode": self.enforcement_mode.value,
            "status": self.status.value,
            "detection_reason": self.detection_reason,
            "evidence_ref": self.evidence_ref.to_dict(),
            "raw_excerpt": self.raw_excerpt,
        }


@dataclass
class MissingGuardrail:
    """A guardrail category that should be present based on observed capabilities."""
    category: GuardrailCategory
    why_required: str
    missing_confidence: Confidence
    evidence_ref: EvidenceRef = field(default_factory=EvidenceRef)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "why_required": self.why_required,
            "missing_confidence": self.missing_confidence.value,
            "evidence_ref": self.evidence_ref.to_dict(),
        }


@dataclass
class GuardrailSummary:
    """Aggregated guardrail posture for a single agent run."""
    detected_categories: List[str] = field(default_factory=list)
    triggered_categories: List[str] = field(default_factory=list)
    missing_categories: List[MissingGuardrail] = field(default_factory=list)
    coverage_confidence: Confidence = Confidence.LOW
    capabilities_observed: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_categories": self.detected_categories,
            "triggered_categories": self.triggered_categories,
            "missing_categories": [m.to_dict() for m in self.missing_categories],
            "coverage_confidence": self.coverage_confidence.value,
            "capabilities_observed": self.capabilities_observed,
            "limitations": self.limitations,
        }


def dedupe_findings(findings: List[GuardrailFinding]) -> List[GuardrailFinding]:
    """Remove duplicate findings by stable id, keeping highest-confidence version."""
    priority = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
    by_id: Dict[str, GuardrailFinding] = {}
    for f in findings:
        fid = f.id
        existing = by_id.get(fid)
        if existing is None or priority.get(f.confidence, 3) < priority.get(existing.confidence, 3):
            by_id[fid] = f
    return list(by_id.values())
