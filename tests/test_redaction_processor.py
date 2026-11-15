"""Tests for PII redaction helpers."""

from traccia.processors.redaction_processor import (
    RedactionSpanProcessor,
    apply_redaction_to_span,
    redact_attributes,
    redact_string,
)


class _FakeSpan:
    def __init__(self, attrs: dict):
        self._attributes = dict(attrs)

    @property
    def attributes(self):
        return self._attributes

    def set_attribute(self, key: str, value):
        self._attributes[key] = value


def test_redact_string_email_phone_ssn():
    raw = "Contact alice@corp.com or 555-123-4567; SSN 123-45-6789"
    out = redact_string(raw)
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_SSN]" in out
    assert "alice@corp.com" not in out


def test_redact_string_phi_heuristics():
    raw = "Patient MRN: A12B34; NPI 1234567890; DOB 01/15/1980"
    out = redact_string(raw)
    assert "[REDACTED_MRN]" in out
    assert "[REDACTED_NPI]" in out
    assert "[REDACTED_DOB]" in out
    assert "A12B34" not in out
    assert "1234567890" not in out


def test_redact_attributes_sensitive_keys():
    attrs = {
        "gen_ai.prompt": "Email me at bob@test.io",
        "http.url": "https://api.example.com",
    }
    out = redact_attributes(attrs)
    assert "[REDACTED_EMAIL]" in out["gen_ai.prompt"]
    assert out["http.url"] == attrs["http.url"]
    assert out["governance.redaction_applied"] is True


def test_apply_redaction_to_span_mutates():
    span = _FakeSpan({"gen_ai.completion": "Reach sue@example.org"})
    n = apply_redaction_to_span(span)
    assert n >= 1
    assert "[REDACTED_EMAIL]" in span.attributes["gen_ai.completion"]
    assert span.attributes["governance.redaction_applied"] is True


def test_redaction_allowlists_traccia_prompt_identity():
    attrs = {
        "traccia.prompt.name": "support-reply",
        "traccia.prompt.version": "3",
        "traccia.prompt.version_id": "uuid-1",
        "traccia.prompt.label": "production",
        "llm.prompt": "secret@example.com",
    }
    out = redact_attributes(attrs)
    assert out["traccia.prompt.name"] == "support-reply"
    assert out["traccia.prompt.version"] == "3"
    assert out["traccia.prompt.version_id"] == "uuid-1"
    assert out["traccia.prompt.label"] == "production"
    assert "[REDACTED_EMAIL]" in out["llm.prompt"]


def test_redaction_allowlists_traccia_policy_effect():
    attrs = {
        "traccia.policy.effect": "deny",
        "traccia.policy.matched": True,
        "llm.prompt": "secret@example.com",
    }
    out = redact_attributes(attrs)
    assert out["traccia.policy.effect"] == "deny"
    assert out["traccia.policy.matched"] is True
    assert "[REDACTED_EMAIL]" in out["llm.prompt"]


class _FakeEventSpan(_FakeSpan):
    def __init__(self, attrs: dict, events=None, status_description=None):
        super().__init__(attrs)
        self.events = events or []
        self.status_description = status_description


def test_processor_honors_redact_all_strings_for_opaque_keys():
    from traccia.processors.redaction_processor import RedactionSpanProcessor

    span = _FakeEventSpan({"opaque.field": "mail x@y.io", "count": 3})
    RedactionSpanProcessor(redact_all_strings=True).on_end(span)
    assert span.attributes["opaque.field"] == "mail [REDACTED_EMAIL]"
    assert span.attributes["count"] == 3


def test_default_redaction_covers_result_and_nested_values():
    span = _FakeEventSpan(
        {
            "result": "send to x@y.io",
            "llm.prompt": ["hi x@y.io", "ok"],
            "meta": {"text": "x@y.io", "nested": [{"content": "x@y.io"}]},
        }
    )
    apply_redaction_to_span(span)
    assert "x@y.io" not in repr(span.attributes)


def test_redaction_covers_events_exceptions_and_status():
    from traccia.processors.redaction_processor import RedactionSpanProcessor

    span = _FakeEventSpan(
        {"safe": "v"},
        events=[
            {"name": "custom", "attributes": {"payload": "x@y.io"}},
            {
                "name": "exception",
                "attributes": {
                    "exception.message": "bad x@y.io",
                    "exception.stacktrace": "Traceback ... ValueError: bad x@y.io",
                },
            },
        ],
        status_description="bad x@y.io",
    )
    RedactionSpanProcessor(redact_all_strings=True).on_end(span)
    assert "x@y.io" not in repr(span.events)
    assert span.status_description == "bad [REDACTED_EMAIL]"


def _provider_with_memory_exporter(**processor_kwargs):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from traccia.tracer.provider import TracerProvider

    provider = TracerProvider()
    provider.add_span_processor(RedactionSpanProcessor(**processor_kwargs))
    exporter = InMemorySpanExporter()
    provider._otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _dump(exported):
    return (
        repr(dict(exported.attributes))
        + repr([(e.name, dict(e.attributes or {})) for e in exported.events])
        + repr(exported.status.description)
    )


def test_end_to_end_export_redacts_events_and_exceptions():
    tracer, exporter = _provider_with_memory_exporter(redact_all_strings=True)
    try:
        with tracer.start_as_current_span("op") as span:
            span.set_attribute("opaque", "x@y.io")
            span.add_event("custom", {"payload": "x@y.io"})
            raise ValueError("bad x@y.io")
    except ValueError:
        pass

    (exported,) = exporter.get_finished_spans()
    assert "x@y.io" not in _dump(exported)
    assert [e.name for e in exported.events] == ["custom", "exception"]


def test_current_span_wrapper_events_are_exported_and_redacted():
    from traccia.context.context import get_current_span

    tracer, exporter = _provider_with_memory_exporter(redact_all_strings=True)
    with tracer.start_as_current_span("op"):
        current = get_current_span()
        current.add_event("custom", {"payload": "x@y.io"})
        current.record_exception(ValueError("bad x@y.io"))

    (exported,) = exporter.get_finished_spans()
    assert [e.name for e in exported.events] == ["custom", "exception"]
    assert "x@y.io" not in _dump(exported)


def test_native_otel_events_and_status_are_redacted():
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import StatusCode

    tracer, exporter = _provider_with_memory_exporter(redact_all_strings=True)
    with tracer.start_as_current_span("op"):
        native = otel_trace.get_current_span()
        native.record_exception(ValueError("bad x@y.io"))
        native.add_event("native", {"payload": "x@y.io"})
        native.set_status(StatusCode.ERROR, "bad x@y.io")

    (exported,) = exporter.get_finished_spans()
    assert [e.name for e in exported.events] == ["exception", "native"]
    assert "x@y.io" not in _dump(exported)


def test_nested_values_inherit_parent_key_sensitivity():
    out = redact_attributes({"result": {"answer": "x@y.io", "items": [{"v": "x@y.io"}]}, "opaque": {"v": "x@y.io"}})
    assert "x@y.io" not in repr(out["result"])
    assert out["opaque"] == {"v": "x@y.io"}


def test_otel_event_limits_are_preserved(monkeypatch):
    monkeypatch.setenv("OTEL_SPAN_EVENT_COUNT_LIMIT", "2")
    tracer, exporter = _provider_with_memory_exporter(redact_all_strings=True)
    with tracer.start_as_current_span("op") as span:
        for i in range(1000):
            span.add_event("e", {"payload": f"user{i}@y.io"})
        assert len(span.events) == 2

    (exported,) = exporter.get_finished_spans()
    assert len(exported.events) == 2
    assert exported.dropped_events == 998
    assert "@y.io" not in _dump(exported)


def test_init_span_logging_runs_after_redaction():
    from traccia import get_tracer_provider, init, stop_tracing
    from traccia.processors.logging_processor import LoggingSpanProcessor

    stop_tracing()  # other tests may leave tracing started, which makes init() a no-op
    try:
        init(
            enable_patching=False,
            auto_start_trace=False,
            redact_pii=True,
            enable_span_logging=True,
            enable_metrics=False,
        )
        procs = get_tracer_provider()._enrichment_processors
        redact_idx = next(i for i, p in enumerate(procs) if isinstance(p, RedactionSpanProcessor))
        log_idx = next(i for i, p in enumerate(procs) if isinstance(p, LoggingSpanProcessor))
        assert redact_idx < log_idx
    finally:
        stop_tracing()


def test_redact_attributes_scrubs_guardrail_finding_excerpts():
    """A finding's raw_excerpt copies span text, so it is scrubbed like that text (TRC-282)."""
    import json

    findings = [
        {
            "name": "provider_refusal",
            "detection_reason": "llm_response_status_with_refusal_text",
            "evidence_ref": {"attribute_keys": ["llm.response.status", "llm.completion"]},
            "raw_excerpt": "I can't help with that, jane@example.com",
        },
        {"name": "provider_content_filter", "raw_excerpt": "content_filter"},
    ]
    out = redact_attributes({"guardrail.findings": json.dumps(findings)})
    scrubbed = json.loads(out["guardrail.findings"])
    assert scrubbed[0]["raw_excerpt"] == "I can't help with that, [REDACTED_EMAIL]"
    # Everything else in the finding is kept as written.
    assert scrubbed[0]["evidence_ref"] == findings[0]["evidence_ref"]
    assert scrubbed[1] == findings[1]


def test_redact_attributes_keeps_guardrail_json_without_pii_byte_identical():
    import json

    summary = json.dumps({"detected_categories": ["moderation"], "limitations": []})
    findings = json.dumps([{"name": "x", "raw_excerpt": "nothing sensitive"}])
    out = redact_attributes({"guardrail.summary": summary, "guardrail.findings": findings})
    assert out["guardrail.summary"] == summary
    assert out["guardrail.findings"] == findings


def test_redact_attributes_guardrail_value_that_is_not_json_is_redacted_as_text():
    out = redact_attributes({"guardrail.findings": "raw jane@example.com"})
    assert out["guardrail.findings"] == "raw [REDACTED_EMAIL]"


def test_redaction_processor_scrubs_guardrail_findings_on_span():
    import json

    span = _FakeSpan(
        {"guardrail.findings": json.dumps([{"name": "x", "raw_excerpt": "SSN 123-45-6789"}])}
    )
    RedactionSpanProcessor().on_end(span)
    assert json.loads(span.attributes["guardrail.findings"])[0]["raw_excerpt"] == "SSN [REDACTED_SSN]"
