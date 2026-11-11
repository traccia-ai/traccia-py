"""PII redaction — manual helpers and optional automatic span redaction before export."""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Iterable, Optional, Sequence

from traccia.governance.schema import REDACTION_APPLIED
from traccia.tracer.provider import SpanProcessor

_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Best-effort PHI-oriented patterns (not medical NER — document limits in docs)
_MRN = re.compile(r"\b(?:MRN|mrn)[:\s#=-]*([A-Za-z0-9-]{4,})\b")
_NPI = re.compile(r"\b(?:NPI|npi)[:\s#=-]*(\d{10})\b")
_DOB = re.compile(
    r"\b(?:DOB|dob|date of birth)[:\s=-]*("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)
_MEDICARE = re.compile(r"\b\d{4}-\d{4}-\d{4}\b")  # common MBI-like grouping (heuristic)


def redact_string(text: str) -> str:
    """Mask common PII/PHI-ish patterns (email, US phone, SSN-like, MRN/NPI/DOB heuristics).

    Best-effort regex only — not a guarantee that all PHI is removed.
    """
    if not text:
        return text
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    # Labeled PHI heuristics before generic phone/SSN so "NPI 1234567890" is not treated as a phone
    text = _MRN.sub("MRN:[REDACTED_MRN]", text)
    text = _NPI.sub("NPI:[REDACTED_NPI]", text)
    text = _DOB.sub("DOB:[REDACTED_DOB]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    text = _SSN.sub("[REDACTED_SSN]", text)
    text = _MEDICARE.sub("[REDACTED_ID]", text)
    return text


# Attribute key substrings that commonly hold user content (extend via RedactionSpanProcessor.extra_key_fragments)
DEFAULT_SENSITIVE_KEY_FRAGMENTS: FrozenSet[str] = frozenset(
    (
        "prompt",
        "completion",
        "input",
        "output",
        "message",
        "content",
        "text",
        "body",
        "query",
        "response",
        "result",
        "exception",
        "error",
        "user",
        "assistant",
        "phi",
        "clinical",
        "patient",
        "diagnosis",
        "medication",
    )
)

# Prompt / eval identity attrs contain substrings that would otherwise redact.
PROMPT_IDENTITY_KEY_PREFIX = "traccia.prompt."
_EVAL_ALLOWLIST_PREFIXES = (
    "traccia.prompt.",
    "traccia.policy.",
    "traccia.experiment.",
    "traccia.eval.",
    "traccia.dataset.",
)


# Attributes holding JSON-serialized guardrail findings or summaries. A finding's
# ``raw_excerpt`` copies text from the span (a completion, an error message), so these are
# decoded and their excerpts scrubbed like the attribute they came from. This covers the
# per-span findings and the root span's aggregate of every finding in the trace.
_GUARDRAIL_JSON_KEYS = frozenset({"guardrail.findings", "guardrail.summary"})
_GUARDRAIL_EXCERPT_FRAGMENTS = ("excerpt",)


def _redact_guardrail_json(
    value: Any,
    *,
    fragments: Iterable[str],
    redact_all_strings: bool,
) -> Any:
    """Scrub excerpts inside a JSON guardrail attribute; non-JSON values are redacted as text."""
    if not isinstance(value, str):
        return value
    import json

    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        # Not the JSON the detector writes: treat the whole value as content.
        return redact_string(value)
    scrubbed = redact_value(
        decoded,
        redact_all_strings=redact_all_strings,
        fragments=tuple(fragments) + _GUARDRAIL_EXCERPT_FRAGMENTS,
    )
    if scrubbed == decoded:
        return value
    return json.dumps(scrubbed)


def _key_is_allowlisted(key: str) -> bool:
    lower = key.lower()
    return any(lower.startswith(p) for p in _EVAL_ALLOWLIST_PREFIXES)


def _key_is_sensitive(key: str, fragments: Iterable[str]) -> bool:
    if _key_is_allowlisted(key):
        return False
    lower = key.lower()
    return any(fragment in lower for fragment in fragments)


def redact_value(
    value: Any,
    *,
    redact_all_strings: bool = False,
    key: str = "",
    fragments: Optional[Iterable[str]] = None,
    _sensitive: bool = False,
) -> Any:
    """Redact a value recursively.

    Strings are redacted when ``redact_all_strings`` is set or when their key, or any
    ancestor key, is sensitive (so ``{"result": {"answer": ...}}`` is covered).
    """
    frags = DEFAULT_SENSITIVE_KEY_FRAGMENTS if fragments is None else fragments
    sensitive = _sensitive or redact_all_strings or _key_is_sensitive(key, frags)
    if isinstance(value, str):
        return redact_string(value) if sensitive else value
    if isinstance(value, (list, tuple)):
        return type(value)(
            redact_value(v, redact_all_strings=redact_all_strings, key=key, fragments=frags, _sensitive=sensitive)
            for v in value
        )
    if isinstance(value, dict):
        return {
            k: redact_value(
                v, redact_all_strings=redact_all_strings, key=str(k), fragments=frags, _sensitive=sensitive
            )
            for k, v in value.items()
        }
    return value


def redact_attributes(
    attrs: Optional[Dict[str, Any]],
    *,
    extra_key_fragments: Optional[Sequence[str]] = None,
    redact_all_strings: bool = False,
) -> Dict[str, Any]:
    """Return a copy of span attributes with sensitive string values redacted."""
    if not attrs:
        return {}
    fragments = list(DEFAULT_SENSITIVE_KEY_FRAGMENTS)
    if extra_key_fragments:
        fragments.extend(extra_key_fragments)
    out: Dict[str, Any] = {}
    for key, value in attrs.items():
        if _key_is_allowlisted(key):
            out[key] = value
            continue
        if key in _GUARDRAIL_JSON_KEYS:
            out[key] = _redact_guardrail_json(
                value, fragments=fragments, redact_all_strings=redact_all_strings
            )
            continue
        out[key] = redact_value(value, redact_all_strings=redact_all_strings, key=key, fragments=fragments)
    out[REDACTION_APPLIED] = True
    return out


def apply_redaction_to_span(
    span: Any,
    *,
    extra_key_fragments: Optional[Sequence[str]] = None,
    redact_all_strings: bool = False,
) -> int:
    """
    Redact sensitive attributes, span events (including exceptions) and the error
    status description on a mutable Traccia span (call inside @observe or before span ends).

    Returns the number of attribute keys, events and status fields updated.
    """
    if not span or not hasattr(span, "set_attribute"):
        return 0
    changed = 0

    raw = dict(getattr(span, "attributes", None) or {})
    if raw:
        redacted = redact_attributes(
            raw, extra_key_fragments=extra_key_fragments, redact_all_strings=redact_all_strings
        )
        for key, value in redacted.items():
            if key == REDACTION_APPLIED:
                continue
            if raw.get(key) != value:
                span.set_attribute(key, value)
                changed += 1

    def _redact_event_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        out = redact_attributes(
            attrs, extra_key_fragments=extra_key_fragments, redact_all_strings=redact_all_strings
        )
        out.pop(REDACTION_APPLIED, None)
        return out

    if hasattr(span, "_rewrite_otel_events"):
        # Traccia span: rewrite events stored on the OTel span itself, so events added
        # through any wrapper or the raw OTel API are covered and OTel limits still apply.
        changed += span._rewrite_otel_events(_redact_event_attrs)
        # Error status descriptions are exception messages: always scrub them.
        if span._rewrite_otel_status_description(redact_string):
            changed += 1
    else:
        events = getattr(span, "events", None)
        if isinstance(events, list):
            for event in events:
                attrs = event.get("attributes") if isinstance(event, dict) else None
                if not attrs:
                    continue
                new_attrs = _redact_event_attrs(attrs)
                if new_attrs != attrs:
                    event["attributes"] = new_attrs
                    changed += 1

    description = getattr(span, "status_description", None)
    if isinstance(description, str) and description:
        new_description = redact_string(description)
        if new_description != description:
            span.status_description = new_description
            changed += 1

    if raw or changed:
        span.set_attribute(REDACTION_APPLIED, True)
    return changed


class RedactionSpanProcessor(SpanProcessor):
    """
    Traccia enrichment processor: redacts sensitive span attributes in on_end (before export).

    Enable via init(redact_pii=True) or TRACCIA_REDACT_PII=1.
    """

    def __init__(
        self,
        *,
        extra_key_fragments: Optional[Sequence[str]] = None,
        redact_all_strings: bool = False,
    ) -> None:
        self._extra_key_fragments = extra_key_fragments
        self._redact_all_strings = redact_all_strings

    def on_end(self, span: Any) -> None:
        apply_redaction_to_span(
            span,
            extra_key_fragments=self._extra_key_fragments,
            redact_all_strings=self._redact_all_strings,
        )


# Legacy exporter wrapper — prefer RedactionSpanProcessor via init(redact_pii=True)
class RedactionSpanExporter:
    """Deprecated: use RedactionSpanProcessor with init(redact_pii=True) instead."""

    def __init__(self, exporter: Any) -> None:
        self._exporter = exporter

    def export(self, spans: list) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        for span in spans:
            attrs = dict(getattr(span, "attributes", None) or {})
            if attrs and hasattr(span, "_attributes"):
                for key, value in redact_attributes(attrs).items():
                    span._attributes[key] = value
        return self._exporter.export(spans)

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)
