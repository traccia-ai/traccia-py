"""Keep pytest from opening exporter connections."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_policy_fail_mode():
    """No remembered Traccia setting or local override leaks between tests (TRC-342)."""
    from traccia.governance import fail_mode
    from traccia.governance.config import gov_config

    fail_mode.clear_for_tests()
    gov_config.policy_fail_mode = None
    yield
    fail_mode.clear_for_tests()
    gov_config.policy_fail_mode = None


@pytest.fixture(autouse=True)
def block_otlp_export(monkeypatch):
    from opentelemetry.sdk.trace.export import SpanExportResult

    def _export(self, spans):  # noqa: ARG001
        return SpanExportResult.SUCCESS

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter.export",
        _export,
        raising=False,
    )
    monkeypatch.setattr("traccia.exporter.otlp_exporter.OTLPExporter.export", lambda self, spans: True)
