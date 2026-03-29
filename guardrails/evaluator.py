"""Missing-guardrail evaluator: infer agent capabilities from trace spans,
apply a required-guardrail matrix, and report missing categories.

The evaluator operates on *all* spans from a single run (collected by the
processor) and compares detected guardrail categories against what the
observed capabilities require.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from traccia.guardrails.constants import (
    ATTR_AGENT_SPAN_TYPE,
    ATTR_AGENT_TOOL_NAME,
    ATTR_GUARDRAIL_SUPPRESS_MISSING,
    ATTR_LLM_COMPLETION,
    ATTR_LLM_MODEL,
    ATTR_LLM_PROMPT,
    ATTR_SPAN_TYPE,
)
from traccia.guardrails.schema import (
    Confidence,
    EvidenceRef,
    GuardrailCategory,
    GuardrailFinding,
    GuardrailSummary,
    MissingGuardrail,
    SourceType,
    dedupe_findings,
)


# ---------------------------------------------------------------------------
# Capability inference
# ---------------------------------------------------------------------------

class _Capabilities:
    """Capabilities observed across all spans of a run."""

    def __init__(self) -> None:
        self.calls_llm = False
        self.handles_user_text = False
        self.produces_user_text = False
        self.uses_tools = False
        self.has_external_actions = False
        # Evidence tracking (span_ids that proved each capability)
        self._evidence: Dict[str, List[str]] = {
            "calls_llm": [],
            "handles_user_text": [],
            "produces_user_text": [],
            "uses_tools": [],
            "has_external_actions": [],
        }

    def observe_span(self, attrs: Dict[str, Any], span_id: str | None = None) -> None:
        sid = span_id or ""
        if attrs.get(ATTR_LLM_MODEL):
            self.calls_llm = True
            self._evidence["calls_llm"].append(sid)

        if attrs.get(ATTR_LLM_PROMPT):
            self.handles_user_text = True
            self._evidence["handles_user_text"].append(sid)

        if attrs.get(ATTR_LLM_COMPLETION):
            self.produces_user_text = True
            self._evidence["produces_user_text"].append(sid)

        span_type = attrs.get(ATTR_SPAN_TYPE) or attrs.get(ATTR_AGENT_SPAN_TYPE, "")
        if span_type in ("tool", "function") or attrs.get(ATTR_AGENT_TOOL_NAME):
            self.uses_tools = True
            self._evidence["uses_tools"].append(sid)
            self.has_external_actions = True
            self._evidence["has_external_actions"].append(sid)

        if span_type == "handoff":
            self.has_external_actions = True
            self._evidence["has_external_actions"].append(sid)

    def to_list(self) -> List[str]:
        out: List[str] = []
        if self.calls_llm:
            out.append("calls_llm")
        if self.handles_user_text:
            out.append("handles_user_text")
        if self.produces_user_text:
            out.append("produces_user_text")
        if self.uses_tools:
            out.append("uses_tools")
        if self.has_external_actions:
            out.append("has_external_actions")
        return out

    def evidence_for(self, cap: str) -> List[str]:
        return self._evidence.get(cap, [])


# ---------------------------------------------------------------------------
# Required-guardrail matrix
# ---------------------------------------------------------------------------

def _required_categories(caps: _Capabilities) -> List[MissingGuardrail]:
    """Return all guardrail categories that *should* exist given capabilities."""
    required: List[MissingGuardrail] = []

    if caps.calls_llm and caps.handles_user_text:
        required.append(MissingGuardrail(
            category=GuardrailCategory.INPUT_VALIDATION,
            why_required="Agent makes LLM calls with prompt data (may be user-provided)",
            missing_confidence=Confidence.MEDIUM,
            evidence_ref=EvidenceRef(
                attribute_keys=["calls_llm", "handles_user_text"],
            ),
        ))
        required.append(MissingGuardrail(
            category=GuardrailCategory.PROMPT_INJECTION,
            why_required="Agent makes LLM calls with prompt data (may be user-provided)",
            missing_confidence=Confidence.MEDIUM,
            evidence_ref=EvidenceRef(
                attribute_keys=["calls_llm", "handles_user_text"],
            ),
        ))

    if caps.produces_user_text:
        required.append(MissingGuardrail(
            category=GuardrailCategory.OUTPUT_VALIDATION,
            why_required="Agent produces text output visible to users",
            missing_confidence=Confidence.MEDIUM,
            evidence_ref=EvidenceRef(
                attribute_keys=["produces_user_text"],
            ),
        ))
        required.append(MissingGuardrail(
            category=GuardrailCategory.MODERATION,
            why_required="Agent produces text output visible to users",
            missing_confidence=Confidence.MEDIUM,
            evidence_ref=EvidenceRef(
                attribute_keys=["produces_user_text"],
            ),
        ))

    if caps.uses_tools:
        required.append(MissingGuardrail(
            category=GuardrailCategory.TOOL_PERMISSION,
            why_required="Agent uses tool/function calls",
            missing_confidence=Confidence.HIGH,
            evidence_ref=EvidenceRef(
                attribute_keys=["uses_tools"],
            ),
        ))

    return required


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------


def evaluate_run(
    all_span_attrs: List[Dict[str, Any]],
    span_ids: List[str | None],
    findings: List[GuardrailFinding],
    trace_id: str | None = None,
    heuristics_enabled: bool = True,
) -> GuardrailSummary:
    """Produce a full GuardrailSummary for a single agent run.

    Parameters
    ----------
    all_span_attrs:
        List of attribute dicts, one per span in the run.
    span_ids:
        Parallel list of span ID strings (or None).
    findings:
        Already-detected GuardrailFinding list (from detectors).
    trace_id:
        Trace ID string for evidence references.
    heuristics_enabled:
        When False, heuristic (Tier C) findings are excluded from summary
        computation. The processor already omits them when disabled; this keeps
        direct calls to evaluate_run consistent.
    """
    # 1. Infer capabilities and collect suppression requests
    caps = _Capabilities()
    suppressed_cats: Set[str] = set()
    for attrs, sid in zip(all_span_attrs, span_ids):
        caps.observe_span(attrs, sid)
        suppress_val = attrs.get(ATTR_GUARDRAIL_SUPPRESS_MISSING)
        if suppress_val:
            # Accept either a list/tuple or a comma-separated string
            if isinstance(suppress_val, (list, tuple)):
                suppressed_cats.update(str(v) for v in suppress_val)
            else:
                suppressed_cats.update(
                    s.strip() for s in str(suppress_val).split(",") if s.strip()
                )

    # 2. Dedupe findings; optionally drop Tier C for summary (matches processor behavior)
    deduped = dedupe_findings(findings)
    if not heuristics_enabled:
        deduped = [f for f in deduped if f.source_type != SourceType.HEURISTIC]

    # 3. Compute detected and triggered categories.
    # Only HIGH/MEDIUM confidence findings count as "covered" for the purpose
    # of removing a category from the missing list.  A LOW-confidence heuristic
    # finding should appear as both a finding *and* a remaining missing category,
    # accurately communicating that we're not sure the category is covered.
    detected_cats: Set[str] = set()
    confident_detected_cats: Set[str] = set()
    triggered_cats: Set[str] = set()
    for f in deduped:
        detected_cats.add(f.category.value)
        if f.triggered is True:
            triggered_cats.add(f.category.value)
        if f.confidence != Confidence.LOW:
            confident_detected_cats.add(f.category.value)

    # 4. Compute missing categories (use confident_detected_cats, not all detected)
    required = _required_categories(caps)
    missing: List[MissingGuardrail] = []
    for req in required:
        cat = req.category.value
        if cat in suppressed_cats:
            continue
        if cat not in confident_detected_cats:
            missing.append(req)

    # 5. Compute aggregate coverage confidence
    if not deduped:
        cov = Confidence.LOW
    elif all(f.confidence == Confidence.HIGH for f in deduped):
        cov = Confidence.HIGH
    elif any(f.confidence == Confidence.HIGH for f in deduped):
        cov = Confidence.MEDIUM
    else:
        cov = Confidence.LOW

    # 6. Build limitations
    limitations: List[str] = []
    if not caps.calls_llm and not caps.uses_tools:
        limitations.append("No LLM or tool spans observed; capability inference may be incomplete.")
    if not deduped:
        limitations.append("No guardrail signals detected in trace; agent may have out-of-band guardrails not visible to tracing.")
    if any(f.confidence == Confidence.LOW for f in deduped):
        limitations.append("Some findings are heuristic-only (low confidence); verify independently.")
    if missing:
        limitations.append(
            f"{len(missing)} expected guardrail categor{'y' if len(missing) == 1 else 'ies'} not detected in trace."
        )

    return GuardrailSummary(
        detected_categories=sorted(detected_cats),
        triggered_categories=sorted(triggered_cats),
        missing_categories=missing,
        coverage_confidence=cov,
        capabilities_observed=caps.to_list(),
        limitations=limitations,
    )
