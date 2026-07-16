"""Tests for PII redaction helpers."""

from traccia.processors.redaction_processor import (
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
