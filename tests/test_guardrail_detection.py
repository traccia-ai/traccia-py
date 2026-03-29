"""Tests for guardrail detection: schema, detectors, evaluator, and processor."""

from __future__ import annotations

import pytest

from traccia.guardrails.schema import (
    Confidence,
    EnforcementMode,
    EvidenceRef,
    FindingStatus,
    GuardrailCategory,
    GuardrailFinding,
    GuardrailSummary,
    MissingGuardrail,
    SourceType,
    dedupe_findings,
)
from traccia.guardrails.detectors import (
    detect_all,
    detect_explicit,
    detect_heuristic,
    detect_provider_native,
)
from traccia.guardrails.evaluator import evaluate_run
from traccia.guardrails.helpers import validate_guardrail_attributes


# ===================================================================
# Schema tests
# ===================================================================


class TestGuardrailFinding:
    def test_id_is_stable(self):
        f = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="content_filter",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
        )
        assert f.id == f.id
        assert len(f.id) == 16

    def test_id_changes_with_different_span(self):
        f1 = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="content_filter",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
        )
        f2 = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="content_filter",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            evidence_ref=EvidenceRef(trace_id="t1", span_id="s2"),
        )
        assert f1.id != f2.id

    def test_status_triggered(self):
        f = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="test",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            triggered=True,
        )
        assert f.status == FindingStatus.TRIGGERED

    def test_status_not_triggered(self):
        f = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="test",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            triggered=False,
        )
        assert f.status == FindingStatus.NOT_TRIGGERED

    def test_status_present_when_none(self):
        f = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="test",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            triggered=None,
        )
        assert f.status == FindingStatus.PRESENT

    def test_to_dict_round_trip(self):
        f = GuardrailFinding(
            category=GuardrailCategory.PII,
            name="pii_check",
            source_type=SourceType.HEURISTIC,
            confidence=Confidence.LOW,
            triggered=True,
            enforcement_mode=EnforcementMode.WARN,
            detection_reason="test_reason",
            evidence_ref=EvidenceRef(trace_id="t1", span_id="s1", integration="test"),
            raw_excerpt="some text",
        )
        d = f.to_dict()
        assert d["category"] == "pii"
        assert d["source_type"] == "heuristic"
        assert d["confidence"] == "low"
        assert d["triggered"] is True
        assert d["enforcement_mode"] == "warn"
        assert d["status"] == "triggered"
        assert d["evidence_ref"]["trace_id"] == "t1"
        assert d["raw_excerpt"] == "some text"
        assert d["id"] == f.id


class TestDedupe:
    def test_deduplicates_by_id(self):
        f1 = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="x",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.LOW,
            evidence_ref=EvidenceRef(trace_id="t", span_id="s"),
        )
        f2 = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="x",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            evidence_ref=EvidenceRef(trace_id="t", span_id="s"),
        )
        result = dedupe_findings([f1, f2])
        assert len(result) == 1
        assert result[0].confidence == Confidence.HIGH

    def test_keeps_distinct_findings(self):
        f1 = GuardrailFinding(
            category=GuardrailCategory.MODERATION,
            name="x",
            source_type=SourceType.EXPLICIT,
            confidence=Confidence.HIGH,
            evidence_ref=EvidenceRef(trace_id="t", span_id="s1"),
        )
        f2 = GuardrailFinding(
            category=GuardrailCategory.PII,
            name="y",
            source_type=SourceType.HEURISTIC,
            confidence=Confidence.LOW,
            evidence_ref=EvidenceRef(trace_id="t", span_id="s2"),
        )
        result = dedupe_findings([f1, f2])
        assert len(result) == 2


# ===================================================================
# Explicit detector tests
# ===================================================================


class TestExplicitDetector:
    def test_openai_agents_guardrail_span(self):
        attrs = {
            "agent.span.type": "guardrail",
            "agent.guardrail.name": "content_policy",
            "agent.guardrail.triggered": True,
        }
        findings = detect_explicit(attrs, trace_id="t1", span_id="s1")
        assert len(findings) == 1
        f = findings[0]
        assert f.source_type == SourceType.EXPLICIT
        assert f.confidence == Confidence.HIGH
        assert f.triggered is True
        assert f.name == "content_policy"
        assert f.detection_reason == "openai_agents_guardrail_span"

    def test_openai_agents_guardrail_not_triggered(self):
        attrs = {
            "agent.span.type": "guardrail",
            "agent.guardrail.name": "safety_check",
            "agent.guardrail.triggered": False,
        }
        findings = detect_explicit(attrs, trace_id="t1", span_id="s1")
        assert len(findings) == 1
        assert findings[0].triggered is False
        assert findings[0].status == FindingStatus.NOT_TRIGGERED

    def test_manual_observe_guardrail_full_attrs(self):
        attrs = {
            "span.type": "guardrail",
            "guardrail.name": "pii_scanner",
            "guardrail.category": "pii",
            "guardrail.triggered": False,
            "guardrail.enforcement_mode": "warn",
        }
        findings = detect_explicit(attrs, trace_id="t1", span_id="s2")
        assert len(findings) == 1
        f = findings[0]
        assert f.source_type == SourceType.EXPLICIT
        assert f.confidence == Confidence.HIGH
        assert f.category == GuardrailCategory.PII
        assert f.enforcement_mode == EnforcementMode.WARN
        assert f.detection_reason == "manual_observe_guardrail_span"

    def test_manual_observe_guardrail_missing_attrs_downgrades_confidence(self):
        attrs = {
            "span.type": "guardrail",
            "guardrail.name": "my_check",
        }
        findings = detect_explicit(attrs, trace_id="t1", span_id="s3")
        assert len(findings) == 1
        assert findings[0].confidence == Confidence.MEDIUM

    def test_no_findings_for_non_guardrail_span(self):
        attrs = {"span.type": "llm", "llm.model": "gpt-4"}
        findings = detect_explicit(attrs)
        assert findings == []


# ===================================================================
# Provider-native detector tests
# ===================================================================


class TestProviderNativeDetector:
    def test_content_filter_finish_reason(self):
        attrs = {"llm.finish_reason": "content_filter"}
        findings = detect_provider_native(attrs, trace_id="t1", span_id="s1")
        assert len(findings) == 1
        f = findings[0]
        assert f.category == GuardrailCategory.MODERATION
        assert f.source_type == SourceType.PROVIDER_NATIVE
        assert f.confidence == Confidence.HIGH
        assert f.triggered is True
        assert f.enforcement_mode == EnforcementMode.BLOCK

    def test_no_finding_for_normal_finish(self):
        attrs = {"llm.finish_reason": "stop"}
        findings = detect_provider_native(attrs)
        assert findings == []

    def test_refusal_text_with_incomplete_status(self):
        attrs = {
            "llm.response.status": "incomplete",
            "llm.completion": "I can't help with that request. It violates my usage policies.",
        }
        findings = detect_provider_native(attrs, trace_id="t1", span_id="s1")
        assert len(findings) == 1
        f = findings[0]
        assert f.confidence == Confidence.MEDIUM
        assert f.detection_reason == "llm_response_status_with_refusal_text"

    def test_no_finding_for_incomplete_without_refusal_text(self):
        attrs = {
            "llm.response.status": "incomplete",
            "llm.completion": "Here are three movies I recommend...",
        }
        findings = detect_provider_native(attrs)
        assert findings == []


# ===================================================================
# Heuristic detector tests
# ===================================================================


class TestHeuristicDetector:
    def test_tool_denial_error(self):
        attrs = {
            "span.type": "tool",
            "error.type": "PermissionError",
            "error.message": "Permission denied: cannot access file system",
        }
        findings = detect_heuristic(attrs, trace_id="t1", span_id="s1")
        assert len(findings) == 1
        f = findings[0]
        assert f.category == GuardrailCategory.TOOL_PERMISSION
        assert f.source_type == SourceType.HEURISTIC
        assert f.confidence == Confidence.LOW

    def test_no_finding_for_generic_error(self):
        attrs = {
            "span.type": "tool",
            "error.type": "ValueError",
            "error.message": "invalid argument for function",
        }
        findings = detect_heuristic(attrs)
        assert findings == []

    def test_disabled_returns_nothing(self):
        attrs = {
            "span.type": "tool",
            "error.type": "PermissionError",
            "error.message": "Permission denied",
        }
        findings = detect_heuristic(attrs, enabled=False)
        assert findings == []


# ===================================================================
# detect_all aggregation
# ===================================================================


class TestDetectAll:
    def test_combines_all_tiers(self):
        attrs = {
            "agent.span.type": "guardrail",
            "agent.guardrail.name": "safety",
            "agent.guardrail.triggered": True,
            "llm.finish_reason": "content_filter",
        }
        findings = detect_all(attrs, trace_id="t1", span_id="s1")
        assert len(findings) == 2
        source_types = {f.source_type for f in findings}
        assert SourceType.EXPLICIT in source_types
        assert SourceType.PROVIDER_NATIVE in source_types

    def test_heuristics_can_be_disabled(self):
        attrs = {
            "span.type": "tool",
            "error.type": "PermissionError",
            "error.message": "Permission denied",
        }
        with_h = detect_all(attrs, heuristics_enabled=True)
        without_h = detect_all(attrs, heuristics_enabled=False)
        assert len(with_h) == 1
        assert len(without_h) == 0


# ===================================================================
# Evaluator tests
# ===================================================================


class TestEvaluateRun:
    def _make_llm_span(self, span_id="s1"):
        return {
            "llm.model": "gpt-4",
            "llm.prompt": "Hello",
            "llm.completion": "Hi there",
        }

    def _make_tool_span(self, span_id="s2"):
        return {
            "span.type": "tool",
            "agent.tool.name": "web_search",
        }

    def test_basic_capabilities_detected(self):
        spans = [self._make_llm_span(), self._make_tool_span()]
        ids = ["s1", "s2"]
        summary = evaluate_run(spans, ids, [], trace_id="t1")
        assert "calls_llm" in summary.capabilities_observed
        assert "handles_user_text" in summary.capabilities_observed
        assert "produces_user_text" in summary.capabilities_observed
        assert "uses_tools" in summary.capabilities_observed

    def test_missing_guardrails_for_llm_with_tools(self):
        spans = [self._make_llm_span(), self._make_tool_span()]
        ids = ["s1", "s2"]
        summary = evaluate_run(spans, ids, [], trace_id="t1")
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.INPUT_VALIDATION in missing_cats
        assert GuardrailCategory.PROMPT_INJECTION in missing_cats
        assert GuardrailCategory.TOOL_PERMISSION in missing_cats
        assert GuardrailCategory.OUTPUT_VALIDATION in missing_cats

    def test_detected_guardrail_removes_from_missing(self):
        spans = [self._make_llm_span(), self._make_tool_span()]
        ids = ["s1", "s2"]
        findings = [
            GuardrailFinding(
                category=GuardrailCategory.INPUT_VALIDATION,
                name="input_check",
                source_type=SourceType.EXPLICIT,
                confidence=Confidence.HIGH,
                triggered=False,
                evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
            ),
            GuardrailFinding(
                category=GuardrailCategory.PROMPT_INJECTION,
                name="injection_guard",
                source_type=SourceType.EXPLICIT,
                confidence=Confidence.HIGH,
                triggered=False,
                evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
            ),
        ]
        summary = evaluate_run(spans, ids, findings, trace_id="t1")
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.INPUT_VALIDATION not in missing_cats
        assert GuardrailCategory.PROMPT_INJECTION not in missing_cats
        assert GuardrailCategory.TOOL_PERMISSION in missing_cats

    def test_coverage_confidence_high_when_all_explicit(self):
        spans = [self._make_llm_span()]
        findings = [
            GuardrailFinding(
                category=GuardrailCategory.MODERATION,
                name="mod",
                source_type=SourceType.EXPLICIT,
                confidence=Confidence.HIGH,
                evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
            ),
        ]
        summary = evaluate_run(spans, ["s1"], findings, trace_id="t1")
        assert summary.coverage_confidence == Confidence.HIGH

    def test_coverage_confidence_low_when_no_findings(self):
        summary = evaluate_run([self._make_llm_span()], ["s1"], [], trace_id="t1")
        assert summary.coverage_confidence == Confidence.LOW

    def test_heuristics_disabled_excludes_tier_c_from_summary(self):
        """evaluate_run(..., heuristics_enabled=False) drops heuristic findings."""
        spans = [self._make_llm_span(), self._make_tool_span()]
        ids = ["s1", "s2"]
        findings = [
            GuardrailFinding(
                category=GuardrailCategory.TOOL_PERMISSION,
                name="inferred_tool_denial",
                source_type=SourceType.HEURISTIC,
                confidence=Confidence.LOW,
                triggered=True,
                evidence_ref=EvidenceRef(trace_id="t1", span_id="s2"),
            ),
        ]
        with_h = evaluate_run(spans, ids, findings, trace_id="t1", heuristics_enabled=True)
        without_h = evaluate_run(spans, ids, findings, trace_id="t1", heuristics_enabled=False)
        assert "tool_permission" in with_h.detected_categories
        assert "tool_permission" not in without_h.detected_categories
        # Without heuristic, tool_permission still missing (LOW did not cover missing anyway)
        assert GuardrailCategory.TOOL_PERMISSION in {m.category for m in without_h.missing_categories}

    def test_limitations_present_when_no_findings(self):
        summary = evaluate_run([self._make_llm_span()], ["s1"], [], trace_id="t1")
        assert any("No guardrail signals" in l for l in summary.limitations)

    def test_triggered_categories_tracked(self):
        spans = [self._make_llm_span()]
        findings = [
            GuardrailFinding(
                category=GuardrailCategory.MODERATION,
                name="content_filter",
                source_type=SourceType.PROVIDER_NATIVE,
                confidence=Confidence.HIGH,
                triggered=True,
                evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
            ),
        ]
        summary = evaluate_run(spans, ["s1"], findings, trace_id="t1")
        assert "moderation" in summary.triggered_categories

    def test_summary_to_dict(self):
        summary = evaluate_run(
            [self._make_llm_span()], ["s1"], [], trace_id="t1"
        )
        d = summary.to_dict()
        assert isinstance(d["detected_categories"], list)
        assert isinstance(d["missing_categories"], list)
        assert isinstance(d["limitations"], list)
        assert d["coverage_confidence"] in ("high", "medium", "low")


# ===================================================================
# Helpers validation
# ===================================================================


class TestValidateGuardrailAttributes:
    def test_all_present(self):
        attrs = {
            "guardrail.name": "test",
            "guardrail.category": "pii",
            "guardrail.triggered": True,
        }
        assert validate_guardrail_attributes(attrs) == []

    def test_missing_fields(self):
        attrs = {"guardrail.name": "test"}
        warnings = validate_guardrail_attributes(attrs)
        assert len(warnings) == 2
        assert any("guardrail.category" in w for w in warnings)
        assert any("guardrail.triggered" in w for w in warnings)

    def test_empty_attrs(self):
        warnings = validate_guardrail_attributes({})
        assert len(warnings) == 3

    def test_require_triggered_false_skips_triggered_warning(self):
        attrs = {"guardrail.name": "x", "guardrail.category": "pii"}
        w = validate_guardrail_attributes(attrs, require_triggered=False)
        assert w == []
        w_all = validate_guardrail_attributes(attrs, require_triggered=True)
        assert any("guardrail.triggered" in x for x in w_all)


# ===================================================================
# Fixture-based integration scenarios
# ===================================================================


class TestFixtureScenarios:
    """End-to-end scenarios with realistic span attribute sets."""

    def test_openai_agents_run_with_guardrail(self):
        """Simulate an OpenAI Agents run with an explicit guardrail span."""
        spans = [
            {"agent.span.type": "agent", "agent.name": "movie_bot"},
            {"llm.model": "gpt-4o", "llm.prompt": "Recommend a movie", "llm.completion": "The Matrix"},
            {
                "agent.span.type": "guardrail",
                "agent.guardrail.name": "content_safety",
                "agent.guardrail.triggered": False,
            },
        ]
        ids = ["s1", "s2", "s3"]
        findings = []
        for attrs, sid in zip(spans, ids):
            findings.extend(detect_all(attrs, trace_id="t1", span_id=sid))
        summary = evaluate_run(spans, ids, findings, trace_id="t1")

        assert len(findings) == 1
        assert findings[0].source_type == SourceType.EXPLICIT
        assert "unknown" in summary.detected_categories
        assert summary.coverage_confidence in (Confidence.HIGH, Confidence.MEDIUM)

    def test_tool_agent_no_guardrails(self):
        """Simulate an agent with tools but no guardrails -- all missing."""
        spans = [
            {"llm.model": "gpt-4o", "llm.prompt": "Search for X", "llm.completion": "Found Y"},
            {"span.type": "tool", "agent.tool.name": "web_search"},
            {"llm.model": "gpt-4o", "llm.prompt": "Summarize", "llm.completion": "Summary"},
        ]
        ids = ["s1", "s2", "s3"]
        findings = []
        for attrs, sid in zip(spans, ids):
            findings.extend(detect_all(attrs, trace_id="t2", span_id=sid))
        summary = evaluate_run(spans, ids, findings, trace_id="t2")

        assert len(findings) == 0
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.INPUT_VALIDATION in missing_cats
        assert GuardrailCategory.PROMPT_INJECTION in missing_cats
        assert GuardrailCategory.TOOL_PERMISSION in missing_cats

    def test_provider_refusal_detected(self):
        """Simulate a provider content_filter response."""
        spans = [
            {
                "llm.model": "gpt-4o",
                "llm.prompt": "bad request",
                "llm.completion": "",
                "llm.finish_reason": "content_filter",
            },
        ]
        ids = ["s1"]
        findings = []
        for attrs, sid in zip(spans, ids):
            findings.extend(detect_all(attrs, trace_id="t3", span_id=sid))
        summary = evaluate_run(spans, ids, findings, trace_id="t3")

        assert len(findings) == 1
        assert findings[0].category == GuardrailCategory.MODERATION
        assert "moderation" in summary.detected_categories
        assert "moderation" in summary.triggered_categories

    def test_heuristic_tool_denial(self):
        """Simulate a tool denial that produces a low-confidence heuristic finding."""
        spans = [
            {
                "span.type": "tool",
                "agent.tool.name": "file_write",
                "error.type": "PermissionError",
                "error.message": "Permission denied: /etc/passwd",
            },
        ]
        ids = ["s1"]
        findings = []
        for attrs, sid in zip(spans, ids):
            findings.extend(detect_all(attrs, trace_id="t4", span_id=sid))

        assert len(findings) == 1
        assert findings[0].confidence == Confidence.LOW
        assert findings[0].source_type == SourceType.HEURISTIC


# ===================================================================
# Processor tests (trace-id scoping, span attribute writing)
# ===================================================================


class _FakeSpanContext:
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id


class _FakeSpan:
    """Minimal span mock for processor tests."""
    def __init__(self, attrs, trace_id, span_id, parent_span_id="parent"):
        self._attributes = dict(attrs)
        self.context = _FakeSpanContext(trace_id, span_id)
        self.parent_span_id = parent_span_id

    @property
    def attributes(self):
        return self._attributes

    def set_attribute(self, key, value):
        self._attributes[key] = value


class TestGuardrailDetectorProcessor:
    def _make_processor(self):
        from traccia.processors.guardrail_detector import GuardrailDetectorProcessor
        return GuardrailDetectorProcessor()

    def test_writes_findings_on_span_with_guardrail_signal(self):
        proc = self._make_processor()
        span = _FakeSpan(
            {"llm.finish_reason": "content_filter"},
            trace_id="t1", span_id="s1",
        )
        proc.on_end(span)
        assert span.attributes.get("guardrail.finding.count") == 1
        assert "guardrail.findings" in span.attributes

    def test_no_attributes_written_on_span_without_signal(self):
        proc = self._make_processor()
        span = _FakeSpan(
            {"llm.model": "gpt-4", "llm.finish_reason": "stop"},
            trace_id="t1", span_id="s1",
        )
        proc.on_end(span)
        assert "guardrail.finding.count" not in span.attributes

    def test_summary_written_on_root_span(self):
        proc = self._make_processor()
        # Child span first
        child = _FakeSpan(
            {"llm.model": "gpt-4", "llm.prompt": "Hello", "llm.completion": "Hi"},
            trace_id="t1", span_id="s1", parent_span_id="root_id",
        )
        proc.on_end(child)

        # Root span (parent_span_id=None)
        root = _FakeSpan(
            {"agent.id": "test_agent"},
            trace_id="t1", span_id="root_id", parent_span_id=None,
        )
        proc.on_end(root)

        assert "guardrail.summary" in root.attributes
        assert "guardrail.summary.coverage_confidence" in root.attributes
        assert "guardrail.summary.detected_categories" in root.attributes
        assert "guardrail.summary.missing_count" in root.attributes

        import json
        summary = json.loads(root.attributes["guardrail.summary"])
        assert "capabilities_observed" in summary
        assert "calls_llm" in summary["capabilities_observed"]

    def test_trace_id_isolation(self):
        """Two interleaved traces produce independent results."""
        proc = self._make_processor()

        # Trace A: has a guardrail
        child_a = _FakeSpan(
            {
                "agent.span.type": "guardrail",
                "agent.guardrail.name": "safety",
                "agent.guardrail.triggered": False,
            },
            trace_id="trace_a", span_id="a1", parent_span_id="a_root",
        )
        # Trace B: has an LLM span only, no guardrail
        child_b = _FakeSpan(
            {"llm.model": "gpt-4", "llm.prompt": "Hi", "llm.completion": "Hey"},
            trace_id="trace_b", span_id="b1", parent_span_id="b_root",
        )
        # Interleave: A child, then B child
        proc.on_end(child_a)
        proc.on_end(child_b)

        # End root A
        root_a = _FakeSpan(
            {}, trace_id="trace_a", span_id="a_root", parent_span_id=None,
        )
        proc.on_end(root_a)

        # End root B
        root_b = _FakeSpan(
            {}, trace_id="trace_b", span_id="b_root", parent_span_id=None,
        )
        proc.on_end(root_b)

        import json
        summary_a = json.loads(root_a.attributes["guardrail.summary"])
        summary_b = json.loads(root_b.attributes["guardrail.summary"])

        # Trace A detected a guardrail, Trace B did not
        assert len(summary_a["detected_categories"]) > 0
        assert len(summary_b["detected_categories"]) == 0

        # Trace B should have missing categories (LLM with user text)
        assert len(summary_b["missing_categories"]) > 0
        assert root_b.attributes["guardrail.summary.missing_count"] > 0

    def test_cleanup_after_root_span(self):
        """Internal state is cleaned up after root span ends."""
        proc = self._make_processor()
        child = _FakeSpan(
            {"llm.model": "gpt-4"},
            trace_id="t1", span_id="s1", parent_span_id="root",
        )
        proc.on_end(child)
        assert "t1" in proc._traces

        root = _FakeSpan(
            {}, trace_id="t1", span_id="root", parent_span_id=None,
        )
        proc.on_end(root)
        assert "t1" not in proc._traces


# ===================================================================
# New tests for Guardrail SDK Improvements
# ===================================================================


# -------------------------------------------------------------------
# 1. Auto-triggered from @observe(as_type="guardrail") bool return
# -------------------------------------------------------------------


class _RecordingSpan:
    """A minimal span mock that records set_attribute calls."""

    def __init__(self, initial_attrs=None):
        self._attributes = dict(initial_attrs or {})
        self.ended = False

    @property
    def attributes(self):
        return self._attributes

    def set_attribute(self, key, value):
        self._attributes[key] = value

    def record_exception(self, exc):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.ended = True
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.ended = True
        return False


class _RecordingTracer:
    """Minimal tracer mock that captures started spans."""

    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name, attributes=None):
        span = _RecordingSpan(attributes)
        self.spans.append(span)
        return span


class TestAutoTriggeredDecorator:
    """Test that decorator.py auto-sets guardrail.triggered from bool return.

    Uses a mock tracer to isolate the decorator logic from the global OTel state.
    """

    def _patched_observe(self, func, name, as_type, attributes, tracer):
        """Apply @observe and patch _get_tracer to use our recording tracer."""
        from unittest.mock import patch
        from traccia.instrumentation.decorator import observe

        decorated = observe(name=name, as_type=as_type, attributes=attributes)(func)
        with patch("traccia.instrumentation.decorator._get_tracer", return_value=tracer):
            return decorated, tracer

    def test_bool_true_return_sets_triggered(self):
        """Returning True from a guardrail function sets guardrail.triggered=True."""
        from unittest.mock import patch
        from traccia.instrumentation.decorator import observe

        tracer = _RecordingTracer()

        @observe(
            name="my_guardrail",
            as_type="guardrail",
            attributes={"guardrail.name": "my_guardrail", "guardrail.category": "pii",
                        "guardrail.triggered": None},
        )
        def pii_check(text: str) -> bool:
            return True

        with patch("traccia.instrumentation.decorator._get_tracer", return_value=tracer):
            result = pii_check("hello my ssn is 123-45-6789")

        assert result is True
        assert len(tracer.spans) == 1
        assert tracer.spans[0].attributes.get("guardrail.triggered") is True

    def test_bool_false_return_sets_triggered_false(self):
        """Returning False sets guardrail.triggered=False."""
        from unittest.mock import patch
        from traccia.instrumentation.decorator import observe

        tracer = _RecordingTracer()

        @observe(
            name="safe_check",
            as_type="guardrail",
            attributes={"guardrail.name": "safe_check", "guardrail.category": "moderation"},
        )
        def safety_check(text: str) -> bool:
            return False

        with patch("traccia.instrumentation.decorator._get_tracer", return_value=tracer):
            result = safety_check("hello world")

        assert result is False
        assert tracer.spans[0].attributes.get("guardrail.triggered") is False

    def test_pre_set_triggered_not_overridden(self):
        """Pre-setting guardrail.triggered=True in attributes= prevents auto-override by False return."""
        from unittest.mock import patch
        from traccia.instrumentation.decorator import observe

        tracer = _RecordingTracer()

        @observe(
            name="pre_set_check",
            as_type="guardrail",
            attributes={
                "guardrail.name": "pre_set_check",
                "guardrail.category": "pii",
                "guardrail.triggered": True,  # pre-set
            },
        )
        def always_triggered(text: str) -> bool:
            return False  # return disagrees, but pre-set should win

        with patch("traccia.instrumentation.decorator._get_tracer", return_value=tracer):
            result = always_triggered("test")

        assert result is False
        # Pre-set True should NOT be overridden by the False return
        assert tracer.spans[0].attributes.get("guardrail.triggered") is True

    def test_non_bool_return_does_not_set_triggered(self):
        """Non-bool returns don't auto-set guardrail.triggered."""
        from unittest.mock import patch
        from traccia.instrumentation.decorator import observe

        tracer = _RecordingTracer()

        @observe(
            name="string_return_check",
            as_type="guardrail",
            attributes={"guardrail.name": "string_return_check", "guardrail.category": "pii"},
        )
        def string_guardrail(text: str) -> str:
            return "clean"

        with patch("traccia.instrumentation.decorator._get_tracer", return_value=tracer):
            string_guardrail("test")

        # No guardrail.triggered should be auto-set for non-bool returns
        assert tracer.spans[0].attributes.get("guardrail.triggered") is None

    def test_async_bool_return_sets_triggered(self):
        """Async guardrail functions also auto-set triggered from bool return."""
        import asyncio
        from unittest.mock import patch
        from traccia.instrumentation.decorator import observe

        tracer = _RecordingTracer()

        @observe(
            name="async_guardrail",
            as_type="guardrail",
            attributes={"guardrail.name": "async_guardrail", "guardrail.category": "moderation"},
        )
        async def async_check(text: str) -> bool:
            return True

        with patch("traccia.instrumentation.decorator._get_tracer", return_value=tracer):
            result = asyncio.new_event_loop().run_until_complete(async_check("bad content"))

        assert result is True
        assert tracer.spans[0].attributes.get("guardrail.triggered") is True


# -------------------------------------------------------------------
# 2. Tier C: non-denial error types are excluded
# -------------------------------------------------------------------


class TestTierCNonDenialExclusion:
    def test_timeout_error_with_denial_keywords_is_excluded(self):
        """TimeoutError with 'denied' in message should NOT produce a finding."""
        attrs = {
            "span.type": "tool",
            "error.type": "TimeoutError",
            "error.message": "Connection denied: request timed out after 30s",
        }
        findings = detect_heuristic(attrs)
        assert findings == []

    def test_connection_error_is_excluded(self):
        attrs = {
            "span.type": "tool",
            "error.type": "ConnectionError",
            "error.message": "Connection refused: host not allowed",
        }
        findings = detect_heuristic(attrs)
        assert findings == []

    def test_os_error_is_excluded(self):
        attrs = {
            "span.type": "tool",
            "error.type": "OSError",
            "error.message": "Permission denied: cannot bind to port",
        }
        findings = detect_heuristic(attrs)
        assert findings == []

    def test_permission_error_is_not_excluded(self):
        """PermissionError is a genuine denial — should still produce a finding."""
        attrs = {
            "span.type": "tool",
            "error.type": "PermissionError",
            "error.message": "Permission denied: cannot write to file",
        }
        findings = detect_heuristic(attrs)
        assert len(findings) == 1
        assert findings[0].confidence == Confidence.LOW

    def test_requests_timeout_is_excluded(self):
        attrs = {
            "span.type": "function",
            "error.type": "requests.exceptions.Timeout",
            "error.message": "Request not allowed: timed out",
        }
        findings = detect_heuristic(attrs)
        assert findings == []


# -------------------------------------------------------------------
# 3. Tier C: low-confidence finding does NOT remove from missing list
# -------------------------------------------------------------------


class TestTierCLowConfidenceDoesNotCoverMissing:
    def test_low_confidence_finding_keeps_category_in_missing(self):
        """A LOW-confidence heuristic tool_permission finding should NOT remove
        tool_permission from the missing categories list."""
        spans = [
            {
                "span.type": "tool",
                "agent.tool.name": "file_write",
                "error.type": "PermissionError",
                "error.message": "Permission denied: /etc/passwd",
            },
            {"llm.model": "gpt-4", "llm.prompt": "Do something", "llm.completion": "Done"},
        ]
        ids = ["s1", "s2"]
        findings = []
        for attrs, sid in zip(spans, ids):
            findings.extend(detect_all(attrs, trace_id="t1", span_id=sid))

        # Should have one LOW-confidence heuristic finding
        assert any(f.confidence == Confidence.LOW for f in findings)

        summary = evaluate_run(spans, ids, findings, trace_id="t1")
        # tool_permission should still appear in missing (low confidence doesn't cover it)
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.TOOL_PERMISSION in missing_cats

        # But it should also appear in detected_categories (we did see something)
        assert "tool_permission" in summary.detected_categories

    def test_high_confidence_finding_removes_from_missing(self):
        """A HIGH-confidence explicit finding DOES remove the category from missing."""
        spans = [
            {"span.type": "tool", "agent.tool.name": "file_write"},
            {"llm.model": "gpt-4", "llm.prompt": "Do something", "llm.completion": "Done"},
        ]
        ids = ["s1", "s2"]
        findings = [
            GuardrailFinding(
                category=GuardrailCategory.TOOL_PERMISSION,
                name="explicit_tool_guard",
                source_type=SourceType.EXPLICIT,
                confidence=Confidence.HIGH,
                triggered=False,
                evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
            )
        ]
        summary = evaluate_run(spans, ids, findings, trace_id="t1")
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.TOOL_PERMISSION not in missing_cats


# -------------------------------------------------------------------
# 4. Tier B: new provider signals
# -------------------------------------------------------------------


class TestTierBExtended:
    def test_azure_content_filtered_finish_reason(self):
        """Azure OpenAI uses 'content_filtered' (not 'content_filter')."""
        attrs = {"llm.finish_reason": "content_filtered", "llm.vendor": "azure"}
        findings = detect_provider_native(attrs)
        assert len(findings) == 1
        assert findings[0].category == GuardrailCategory.MODERATION
        assert findings[0].confidence == Confidence.HIGH

    def test_google_safety_finish_reason(self):
        """Google GenAI direct uses 'SAFETY' as finish_reason."""
        attrs = {"llm.finish_reason": "SAFETY", "llm.vendor": "google"}
        findings = detect_provider_native(attrs)
        assert len(findings) == 1
        assert findings[0].category == GuardrailCategory.MODERATION

    def test_anthropic_stop_reason_content_filter(self):
        """Anthropic llm.stop_reason = 'content_filter' triggers moderation finding."""
        attrs = {"llm.stop_reason": "content_filter", "llm.vendor": "anthropic"}
        findings = detect_provider_native(attrs)
        assert len(findings) == 1
        assert findings[0].name == "anthropic_content_filter"
        assert findings[0].confidence == Confidence.HIGH
        assert findings[0].triggered is True

    def test_anthropic_stop_reason_content_filtered(self):
        attrs = {"llm.stop_reason": "content_filtered", "llm.vendor": "anthropic"}
        findings = detect_provider_native(attrs)
        assert len(findings) == 1
        assert findings[0].detection_reason == "llm_stop_reason_content_filter"

    def test_anthropic_stop_reason_without_vendor_no_finding(self):
        """stop_reason alone must not fire without Anthropic vendor (avoid false positives)."""
        attrs = {"llm.stop_reason": "content_filter", "llm.vendor": "openai"}
        assert detect_provider_native(attrs) == []
        attrs_no_vendor = {"llm.stop_reason": "content_filter"}
        assert detect_provider_native(attrs_no_vendor) == []

    def test_anthropic_vendor_claude_alias(self):
        attrs = {"llm.stop_reason": "content_filter", "llm.vendor": "claude"}
        findings = detect_provider_native(attrs)
        assert len(findings) == 1

    def test_anthropic_policy_violation_error_message(self):
        """Anthropic error.message with 'content policy' triggers medium finding on LLM spans."""
        attrs = {
            "llm.vendor": "anthropic",
            "llm.model": "claude-3-5-sonnet-20241022",
            "error.message": "Your request violates Anthropic's usage policy.",
        }
        findings = detect_provider_native(attrs)
        assert len(findings) == 1
        assert findings[0].confidence == Confidence.MEDIUM
        assert findings[0].detection_reason == "anthropic_error_message_policy_phrase"

    def test_anthropic_policy_not_fired_without_llm_span_signals(self):
        """Policy phrase on non-LLM span (no model/prompt/completion) is ignored."""
        attrs = {
            "llm.vendor": "anthropic",
            "error.message": "Your request violates Anthropic's usage policy.",
        }
        assert detect_provider_native(attrs) == []

    def test_anthropic_policy_phrase_not_triggered_for_other_vendors(self):
        """Policy phrase matching only fires for llm.vendor=anthropic."""
        attrs = {
            "llm.vendor": "openai",
            "error.message": "Your request violates Anthropic's usage policy.",
        }
        findings = detect_provider_native(attrs)
        assert findings == []

    def test_google_safety_ratings_blocked_true(self):
        """llm.safety_ratings with blocked=true triggers moderation finding."""
        import json
        ratings = [{"category": "HARM_CATEGORY_DANGEROUS", "probability": "LOW", "blocked": True}]
        attrs = {"llm.safety_ratings": json.dumps(ratings)}
        findings = detect_provider_native(attrs)
        assert len(findings) == 1
        assert findings[0].name == "google_safety_ratings_blocked"
        assert findings[0].confidence == Confidence.MEDIUM

    def test_google_safety_ratings_high_probability(self):
        """llm.safety_ratings with probability=HIGH triggers moderation finding."""
        import json
        ratings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH", "blocked": False},
        ]
        attrs = {"llm.safety_ratings": json.dumps(ratings)}
        findings = detect_provider_native(attrs)
        assert len(findings) == 1

    def test_google_safety_ratings_low_probability_not_triggered(self):
        """LOW probability without blocked=true produces no finding."""
        import json
        ratings = [{"category": "HARM_CATEGORY_DANGEROUS", "probability": "LOW", "blocked": False}]
        attrs = {"llm.safety_ratings": json.dumps(ratings)}
        findings = detect_provider_native(attrs)
        assert findings == []

    def test_malformed_safety_ratings_is_ignored(self):
        """Malformed JSON in llm.safety_ratings should not raise; just produce no finding."""
        attrs = {"llm.safety_ratings": "not valid json {{{"}
        findings = detect_provider_native(attrs)
        assert findings == []


# -------------------------------------------------------------------
# 5. Suppression mechanism
# -------------------------------------------------------------------


class TestSuppression:
    def _llm_tool_spans(self):
        return [
            {"llm.model": "gpt-4", "llm.prompt": "Hi", "llm.completion": "Hello"},
            {"span.type": "tool", "agent.tool.name": "calc"},
        ], ["s1", "s2"]

    def test_suppress_single_category(self):
        """suppress_missing on a span removes that category from missing list."""
        spans, ids = self._llm_tool_spans()
        # Add suppress attribute to a span
        spans[0]["traccia.guardrail.suppress_missing"] = ["prompt_injection"]

        summary = evaluate_run(spans, ids, [], trace_id="t1")
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.PROMPT_INJECTION not in missing_cats
        # input_validation should still be missing
        assert GuardrailCategory.INPUT_VALIDATION in missing_cats

    def test_suppress_multiple_categories(self):
        spans, ids = self._llm_tool_spans()
        spans[0]["traccia.guardrail.suppress_missing"] = ["prompt_injection", "input_validation"]

        summary = evaluate_run(spans, ids, [], trace_id="t1")
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.PROMPT_INJECTION not in missing_cats
        assert GuardrailCategory.INPUT_VALIDATION not in missing_cats
        # tool_permission not suppressed, should still be missing
        assert GuardrailCategory.TOOL_PERMISSION in missing_cats

    def test_suppress_on_any_span(self):
        """Suppression attribute can be on any span, not just root."""
        spans, ids = self._llm_tool_spans()
        spans[1]["traccia.guardrail.suppress_missing"] = ["tool_permission"]

        summary = evaluate_run(spans, ids, [], trace_id="t1")
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.TOOL_PERMISSION not in missing_cats

    def test_suppression_does_not_remove_detected_findings(self):
        """Suppressing a missing category does not affect detected findings."""
        spans, ids = self._llm_tool_spans()
        spans[0]["traccia.guardrail.suppress_missing"] = ["prompt_injection"]
        findings = [
            GuardrailFinding(
                category=GuardrailCategory.PROMPT_INJECTION,
                name="inj_guard",
                source_type=SourceType.EXPLICIT,
                confidence=Confidence.HIGH,
                triggered=True,
                evidence_ref=EvidenceRef(trace_id="t1", span_id="s1"),
            )
        ]
        summary = evaluate_run(spans, ids, findings, trace_id="t1")
        # Finding is still present
        assert "prompt_injection" in summary.detected_categories
        assert "prompt_injection" in summary.triggered_categories

    def test_no_suppression_on_empty_spans_has_no_effect(self):
        """Absence of suppress attribute leaves missing list intact."""
        spans, ids = self._llm_tool_spans()
        summary = evaluate_run(spans, ids, [], trace_id="t1")
        missing_cats = {m.category for m in summary.missing_categories}
        assert GuardrailCategory.PROMPT_INJECTION in missing_cats
        assert GuardrailCategory.INPUT_VALIDATION in missing_cats


# -------------------------------------------------------------------
# 6. Capability confidence downgrade
# -------------------------------------------------------------------


class TestCapabilityConfidence:
    def test_input_validation_missing_confidence_is_medium(self):
        """input_validation and prompt_injection should be MEDIUM confidence in missing list
        (not HIGH) because we can't confirm the LLM prompt is user-provided."""
        spans = [
            {"llm.model": "gpt-4", "llm.prompt": "Internal batch doc", "llm.completion": "Done"},
        ]
        summary = evaluate_run(spans, ["s1"], [], trace_id="t1")
        for m in summary.missing_categories:
            if m.category == GuardrailCategory.INPUT_VALIDATION:
                assert m.missing_confidence == Confidence.MEDIUM, (
                    f"Expected MEDIUM for input_validation, got {m.missing_confidence}"
                )
            if m.category == GuardrailCategory.PROMPT_INJECTION:
                assert m.missing_confidence == Confidence.MEDIUM, (
                    f"Expected MEDIUM for prompt_injection, got {m.missing_confidence}"
                )

    def test_why_required_text_reflects_uncertainty(self):
        """why_required text should mention 'may be user-provided' to reflect inference uncertainty."""
        spans = [
            {"llm.model": "gpt-4", "llm.prompt": "text", "llm.completion": "out"},
        ]
        summary = evaluate_run(spans, ["s1"], [], trace_id="t1")
        for m in summary.missing_categories:
            if m.category in (GuardrailCategory.INPUT_VALIDATION, GuardrailCategory.PROMPT_INJECTION):
                assert "may be user-provided" in m.why_required

    def test_tool_permission_missing_confidence_is_still_high(self):
        """tool_permission confidence should remain HIGH (tool use is unambiguous)."""
        spans = [
            {"span.type": "tool", "agent.tool.name": "calc"},
        ]
        summary = evaluate_run(spans, ["s1"], [], trace_id="t1")
        for m in summary.missing_categories:
            if m.category == GuardrailCategory.TOOL_PERMISSION:
                assert m.missing_confidence == Confidence.HIGH
