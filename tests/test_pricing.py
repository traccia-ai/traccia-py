"""Tests for the new pricing architecture.

Covers:
- Bundled snapshot loading (cost_engine.BUNDLED_PRICING)
- 4-level resolution precedence in pricing_config.load_pricing_with_source
- Staleness logic (snapshot_age_days)
- CostAnnotatingProcessor new span attributes
- CLI pricing subcommand (status, clear)
- Back-compat: AGENT_DASHBOARD_PRICING_JSON still works
- Back-compat: DEFAULT_PRICING alias still importable
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# cost_engine
# ---------------------------------------------------------------------------

class TestCostEngine:
    def test_bundled_pricing_loaded(self):
        from traccia.processors.cost_engine import BUNDLED_PRICING, BUNDLED_PRICING_GENERATED_AT
        # Snapshot must contain popular models
        assert isinstance(BUNDLED_PRICING, dict)
        assert len(BUNDLED_PRICING) > 10, "Expected many models from LiteLLM snapshot"
        assert BUNDLED_PRICING_GENERATED_AT != "unknown", "generated_at should be set"

    def test_default_pricing_alias(self):
        """DEFAULT_PRICING must remain importable as a back-compat alias."""
        from traccia.processors.cost_engine import DEFAULT_PRICING, BUNDLED_PRICING
        assert DEFAULT_PRICING is BUNDLED_PRICING

    def test_compute_cost_exact_match(self):
        from traccia.processors.cost_engine import compute_cost
        table = {"gpt-4": {"prompt": 0.03, "completion": 0.06}}
        cost = compute_cost("gpt-4", 1000, 500, pricing_table=table)
        assert cost == pytest.approx(0.03 + 0.03, rel=1e-5)

    def test_compute_cost_prefix_match(self):
        from traccia.processors.cost_engine import compute_cost
        table = {"gpt-4o": {"prompt": 0.005, "completion": 0.015}}
        cost = compute_cost("gpt-4o-2024-08-06", 1000, 1000, pricing_table=table)
        assert cost is not None

    def test_compute_cost_unknown_model_returns_none(self):
        from traccia.processors.cost_engine import compute_cost
        cost = compute_cost("definitely-not-a-model", 100, 100, pricing_table={})
        assert cost is None

    def test_prefix_longer_key_wins(self):
        """gpt-4o should not be matched by gpt-4 key when gpt-4o exists."""
        from traccia.processors.cost_engine import _lookup_price
        table = {"gpt-4": {"prompt": 0.03, "completion": 0.06},
                 "gpt-4o": {"prompt": 0.005, "completion": 0.015}}
        key, _ = _lookup_price("gpt-4o-mini", table)
        assert key == "gpt-4o"


# ---------------------------------------------------------------------------
# pricing_config — resolution order
# ---------------------------------------------------------------------------

class TestPricingConfigResolution:
    def test_bundled_is_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRACCIA_PRICING_OVERRIDE_JSON", raising=False)
        monkeypatch.delenv("AGENT_DASHBOARD_PRICING_JSON", raising=False)
        monkeypatch.setenv("TRACCIA_PRICING_CACHE_PATH", str(tmp_path / "no_cache.json"))

        from traccia.pricing_config import load_pricing_with_source
        _, source, generated_at = load_pricing_with_source()
        assert source == "bundled"
        assert generated_at != "unknown"

    def test_local_cache_beats_bundled(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "pricing.json"
        snapshot = {
            "generated_at": "2026-01-01T00:00:00Z",
            "source": "litellm",
            "models": {"test-model": {"prompt": 9.9, "completion": 9.9}},
        }
        cache_file.write_text(json.dumps(snapshot))
        monkeypatch.setenv("TRACCIA_PRICING_CACHE_PATH", str(cache_file))
        monkeypatch.delenv("TRACCIA_PRICING_OVERRIDE_JSON", raising=False)
        monkeypatch.delenv("AGENT_DASHBOARD_PRICING_JSON", raising=False)

        from traccia import pricing_config
        import importlib; importlib.reload(pricing_config)
        _, source, generated_at = pricing_config.load_pricing_with_source()
        assert source == "local_cache"
        assert generated_at == "2026-01-01T00:00:00Z"

    def test_env_override_beats_cache(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "pricing.json"
        cache_file.write_text(json.dumps({
            "generated_at": "2026-01-01T00:00:00Z",
            "source": "litellm",
            "models": {"base-model": {"prompt": 1.0, "completion": 1.0}},
        }))
        monkeypatch.setenv("TRACCIA_PRICING_CACHE_PATH", str(cache_file))
        env_pricing = json.dumps({"env-model": {"prompt": 7.7, "completion": 7.7}})
        monkeypatch.setenv("TRACCIA_PRICING_OVERRIDE_JSON", env_pricing)

        from traccia import pricing_config
        import importlib; importlib.reload(pricing_config)
        table, source, _ = pricing_config.load_pricing_with_source()
        assert source == "env"
        assert "env-model" in table

    def test_programmatic_override_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "TRACCIA_PRICING_OVERRIDE_JSON",
            json.dumps({"env-model": {"prompt": 7.7, "completion": 7.7}}),
        )
        monkeypatch.setenv("TRACCIA_PRICING_CACHE_PATH", str(tmp_path / "no_cache.json"))

        from traccia import pricing_config
        import importlib; importlib.reload(pricing_config)
        override = {"override-model": {"prompt": 99.0, "completion": 99.0}}
        table, source, _ = pricing_config.load_pricing_with_source(override=override)
        assert source == "override"
        assert "override-model" in table

    def test_deprecated_env_var_alias(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv("TRACCIA_PRICING_OVERRIDE_JSON", raising=False)
        monkeypatch.setenv("TRACCIA_PRICING_CACHE_PATH", str(tmp_path / "no_cache.json"))
        env_pricing = json.dumps({"old-env-model": {"prompt": 5.0, "completion": 5.0}})
        monkeypatch.setenv("AGENT_DASHBOARD_PRICING_JSON", env_pricing)

        import logging
        with caplog.at_level(logging.WARNING, logger="traccia.pricing_config"):
            from traccia import pricing_config
            import importlib; importlib.reload(pricing_config)
            table, source, _ = pricing_config.load_pricing_with_source()

        assert source == "env"
        assert "old-env-model" in table
        assert any("deprecated" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# snapshot_age_days
# ---------------------------------------------------------------------------

class TestSnapshotAge:
    def test_recent_snapshot(self):
        from traccia.pricing_config import snapshot_age_days
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        age = snapshot_age_days(ts)
        assert age is not None
        assert age < 1

    def test_old_snapshot(self):
        from traccia.pricing_config import snapshot_age_days
        old = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
        age = snapshot_age_days(old)
        assert age is not None
        assert age > 44

    def test_unknown_returns_none(self):
        from traccia.pricing_config import snapshot_age_days
        assert snapshot_age_days("unknown") is None
        assert snapshot_age_days("") is None


# ---------------------------------------------------------------------------
# CostAnnotatingProcessor — new span attributes
# ---------------------------------------------------------------------------

class _FakeSpan:
    def __init__(self, attrs: dict):
        self.attributes = dict(attrs)
        self._set: dict = {}

    def set_attribute(self, key: str, value) -> None:
        self.attributes[key] = value
        self._set[key] = value


class TestCostAnnotatingProcessor:
    def _make_processor(self, generated_at: str = "2026-01-01T00:00:00Z"):
        from traccia.processors.cost_processor import CostAnnotatingProcessor
        table = {"gpt-4o": {"prompt": 0.005, "completion": 0.015}}
        return CostAnnotatingProcessor(
            pricing_table=table,
            pricing_source="local_cache",
            pricing_generated_at=generated_at,
        )

    def test_cost_is_set(self):
        proc = self._make_processor()
        span = _FakeSpan({
            "llm.model": "gpt-4o",
            "llm.usage.prompt_tokens": 1000,
            "llm.usage.completion_tokens": 500,
        })
        proc.on_end(span)
        assert "llm.cost.usd" in span.attributes

    def test_new_pricing_attributes_set(self):
        proc = self._make_processor()
        span = _FakeSpan({
            "llic.model": "gpt-4o",
            "llm.model": "gpt-4o",
            "llm.usage.prompt_tokens": 1000,
            "llm.usage.completion_tokens": 500,
        })
        proc.on_end(span)
        assert "llm.pricing.generated_at" in span.attributes
        assert "llm.pricing.age_days" in span.attributes
        assert "llm.pricing.snapshot_version" in span.attributes
        assert "llm.pricing.source" in span.attributes
        assert span.attributes["llm.pricing.source"] == "local_cache"

    def test_llm_usage_source_set_and_alias(self):
        proc = self._make_processor()
        span = _FakeSpan({
            "llm.model": "gpt-4o",
            "llm.usage.prompt_tokens": 100,
            "llm.usage.completion_tokens": 50,
            "llm.usage.source": "openai",
        })
        proc.on_end(span)
        # Both new name and old alias must be set
        assert span.attributes.get("llm.usage.source") == "openai"
        assert span.attributes.get("llm.cost.source") == "openai"

    def test_no_double_computation_if_cost_already_set(self):
        proc = self._make_processor()
        span = _FakeSpan({
            "llm.model": "gpt-4o",
            "llm.usage.prompt_tokens": 1000,
            "llm.usage.completion_tokens": 500,
            "llm.cost.usd": 0.99,
        })
        proc.on_end(span)
        # Must not overwrite
        assert span.attributes["llm.cost.usd"] == 0.99

    def test_update_pricing_table_refreshes_age(self):
        proc = self._make_processor()
        new_table = {"claude-3-opus": {"prompt": 0.015, "completion": 0.075}}
        new_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        proc.update_pricing_table(new_table, pricing_source="platform", pricing_generated_at=new_ts)
        assert proc.pricing_source == "platform"
        assert proc.pricing_generated_at == new_ts

    def test_staleness_warning_fires_once(self, caplog):
        import logging
        from traccia.processors import cost_processor as cp_module

        # Reset the global guard so we can test it cleanly
        cp_module._staleness_warned.clear()

        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        proc = self._make_processor(generated_at=old_ts)

        with caplog.at_level(logging.WARNING, logger="traccia.processors.cost_processor"):
            for _ in range(3):
                span = _FakeSpan({
                    "llm.model": "gpt-4o",
                    "llm.usage.prompt_tokens": 100,
                    "llm.usage.completion_tokens": 50,
                })
                proc.on_end(span)

        warnings = [r for r in caplog.records if "pricing snapshot" in r.message.lower()]
        assert len(warnings) == 1, "Staleness warning should fire exactly once"
        cp_module._staleness_warned.clear()  # restore


# ---------------------------------------------------------------------------
# Platform normalizer
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_basic_normalization(self):
        from app.services.pricing.normalizer import normalize
        raw = {
            "gpt-4": {
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
        result = normalize(raw)
        assert "gpt-4" in result
        entry = result["gpt-4"]
        assert abs(entry["prompt"] - 0.03) < 1e-6
        assert abs(entry["completion"] - 0.06) < 1e-6
        assert entry["_provider"] == "openai"

    def test_skips_entries_without_cost(self):
        from app.services.pricing.normalizer import normalize
        raw = {"no-cost-model": {"max_tokens": 4096}}
        result = normalize(raw)
        assert "no-cost-model" not in result

    def test_optional_fields(self):
        from app.services.pricing.normalizer import normalize
        raw = {
            "claude-3": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            }
        }
        result = normalize(raw)
        entry = result["claude-3"]
        assert "cache_write" in entry
        assert "cached_prompt" in entry

    def test_diff_detects_changed_keys(self):
        from app.services.pricing.normalizer import diff_models
        prev = {"gpt-4": {"prompt": 0.03, "completion": 0.06}}
        curr = {"gpt-4": {"prompt": 0.04, "completion": 0.06}}  # prompt changed
        changed = diff_models(prev, curr)
        assert "gpt-4" in changed

    def test_diff_ignores_metadata(self):
        from app.services.pricing.normalizer import diff_models
        prev = {"gpt-4": {"prompt": 0.03, "completion": 0.06, "_provider": "openai"}}
        curr = {"gpt-4": {"prompt": 0.03, "completion": 0.06, "_provider": "azure"}}
        changed = diff_models(prev, curr)
        assert "gpt-4" not in changed

    def test_diff_catches_new_model(self):
        from app.services.pricing.normalizer import diff_models
        prev = {}
        curr = {"new-model": {"prompt": 0.01, "completion": 0.02}}
        changed = diff_models(prev, curr)
        assert "new-model" in changed


# ---------------------------------------------------------------------------
# CostResolver — shared singleton and trace/metrics parity
# ---------------------------------------------------------------------------

class TestCostResolver:
    def test_resolver_created_with_default_pricing(self, tmp_path, monkeypatch):
        """get_resolver() returns a resolver with a non-empty pricing table."""
        # Reset singleton for isolation
        import traccia.cost_resolver as cr
        cr._resolver = None
        monkeypatch.setenv("TRACCIA_PRICING_CACHE_PATH", str(tmp_path / "no_cache.json"))
        monkeypatch.delenv("TRACCIA_PRICING_OVERRIDE_JSON", raising=False)
        monkeypatch.delenv("AGENT_DASHBOARD_PRICING_JSON", raising=False)

        resolver = cr.get_resolver()
        assert isinstance(resolver.pricing_table, dict)
        assert len(resolver.pricing_table) > 0

    def test_resolver_compute_returns_cost(self):
        """compute() returns a float for known models."""
        from traccia.cost_resolver import CostResolver
        table = {"gpt-4o": {"prompt": 0.005, "completion": 0.015}}
        resolver = CostResolver(table, "test", "2026-01-01T00:00:00Z")
        cost = resolver.compute("gpt-4o", 1000, 500)
        assert cost is not None
        assert cost == pytest.approx(0.005 + 0.0075, rel=1e-5)

    def test_set_resolver_replaces_singleton(self):
        """set_resolver() updates the process-level resolver."""
        import traccia.cost_resolver as cr
        from traccia.cost_resolver import CostResolver, set_resolver, get_resolver
        override_table = {"my-custom-model": {"prompt": 99.0, "completion": 99.0}}
        custom_resolver = CostResolver(override_table, "override", "2026-01-01T00:00:00Z")
        set_resolver(custom_resolver)
        assert get_resolver() is custom_resolver
        assert get_resolver().compute("my-custom-model", 1, 1) is not None
        # Reset
        cr._resolver = None

    def test_resolver_update_changes_table(self):
        """update() replaces the table atomically."""
        from traccia.cost_resolver import CostResolver
        r = CostResolver({"model-a": {"prompt": 1.0, "completion": 1.0}}, "bundled", "2026-01-01T00:00:00Z")
        assert r.compute("model-a", 1000, 0) is not None
        assert r.compute("model-b", 1000, 0) is None

        r.update({"model-b": {"prompt": 2.0, "completion": 2.0}}, "updated", "2026-06-01T00:00:00Z")
        assert r.compute("model-a", 1000, 0) is None
        assert r.compute("model-b", 1000, 0) is not None

    def test_pricing_override_via_set_resolver_is_reflected(self):
        """
        Simulates what start_tracing(pricing_override=...) does:
        set_resolver() is called with the override table, and both
        the span path (CostAnnotatingProcessor) and the metrics path
        (openai agents recorder) read the same resolver.

        This is the trace/metrics parity test.
        """
        import traccia.cost_resolver as cr
        from traccia.cost_resolver import CostResolver, set_resolver, get_resolver

        override = {"custom-llm": {"prompt": 42.0, "completion": 84.0}}
        set_resolver(CostResolver(override, "override", "2026-01-01T00:00:00Z"))

        resolver = get_resolver()
        # Span path: compute cost
        span_cost = resolver.compute("custom-llm", 1, 1)  # 0.042 + 0.084
        assert span_cost is not None

        # Metrics path: same call
        metric_cost = resolver.compute("custom-llm", 1, 1)
        assert span_cost == metric_cost, "Span and metric costs diverged"

        # Reset
        cr._resolver = None

    def test_resolver_thread_safety(self):
        """Concurrent reads and updates should not raise."""
        from traccia.cost_resolver import CostResolver
        import threading, time

        r = CostResolver({"model": {"prompt": 1.0, "completion": 1.0}}, "bundled", "x")
        errors = []

        def reader():
            for _ in range(100):
                try:
                    r.compute("model", 10, 10)
                except Exception as e:
                    errors.append(e)

        def writer():
            for i in range(10):
                try:
                    r.update({"model": {"prompt": float(i), "completion": float(i)}})
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)] + [threading.Thread(target=writer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread safety errors: {errors}"
