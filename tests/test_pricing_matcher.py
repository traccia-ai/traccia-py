"""
Tests for traccia.pricing_matcher.match_model.

The cases in fixtures/pricing_match_cases.json are shared with
traccia-dashboard-service, which keeps an identical copy of the matcher; both repos
must pass all of them. Tests below the fixture cover behaviour the fixture cannot
express (empty input, tables without _provider, index caching, the bundled snapshot).
"""

import json
from pathlib import Path

import pytest

from traccia.pricing_matcher import PriceMatch, match_model

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "pricing_match_cases.json").read_text(
        encoding="utf-8"
    )
)
TABLE = _FIXTURE["table"]


@pytest.mark.parametrize(
    "case", _FIXTURE["cases"], ids=lambda c: f"{c['model']}|{c['vendor']}"
)
def test_shared_case(case):
    result = match_model(case["model"], TABLE, case["vendor"])
    if case["key"] is None:
        assert result is None
    else:
        assert result is not None
        assert (result.key, result.kind) == (case["key"], case["kind"])


def test_match_reports_provider():
    assert match_model("openai/gpt-oss-120b", TABLE, "groq") == PriceMatch(
        "groq/openai/gpt-oss-120b", "alias", "groq"
    )


@pytest.mark.parametrize("model", [None, "", "   "])
def test_empty_model(model):
    assert match_model(model, TABLE) is None


def test_empty_table():
    assert match_model("gpt-4o", {}) is None


class TestTablesWithoutProvider:
    """Hand-written tables (tests, pricing_override) have no _provider metadata."""

    TABLE = {
        "gpt-4o": {"prompt": 1},
        "gpt-4o-mini": {"prompt": 2},
        "xai/grok-4.7": {"prompt": 3},
    }

    def test_longest_key_wins(self):
        assert match_model("gpt-4o-mini-2024-07-18", self.TABLE).key == "gpt-4o-mini"

    def test_first_segment_is_provider(self):
        assert match_model("grok-4.7", self.TABLE, "xai") == PriceMatch(
            "xai/grok-4.7", "alias", "xai"
        )


class TestIndexCache:
    def test_table_growth_is_seen(self):
        table = {"gpt-4o": {"prompt": 1}}
        assert match_model("o9-mega", table) is None
        table["o9-mega"] = {"prompt": 2}
        assert match_model("o9-mega", table).key == "o9-mega"

    def test_distinct_tables(self):
        a, b = {"model-a": {}}, {"model-b": {}}
        assert match_model("model-a", a).key == "model-a"
        assert match_model("model-a", b) is None


# Real model ids as each SDK reports them, against the bundled LiteLLM snapshot.
BUNDLED_CASES = [
    ("llama-3.3-70b-versatile", "groq", "groq/llama-3.3-70b-versatile"),
    ("llama-3.1-8b-instant", "groq", "groq/llama-3.1-8b-instant"),
    ("openai/gpt-oss-120b", "groq", "groq/openai/gpt-oss-120b"),
    ("openai/gpt-oss-20b", "groq", "groq/openai/gpt-oss-20b"),
    (
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "groq",
        "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    ),
    ("moonshotai/kimi-k2-instruct", "groq", "groq/moonshotai/kimi-k2-instruct-0905"),
    ("qwen/qwen3-32b", "groq", "groq/qwen/qwen3-32b"),
    ("gpt-4o", "openai", "gpt-4o"),
    ("gpt-4o-2099-01-01", "openai", "gpt-4o"),
    ("claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
    ("grok-4", "xai", "xai/grok-4"),
]


@pytest.mark.parametrize("model,vendor,expected", BUNDLED_CASES)
def test_bundled_snapshot(model, vendor, expected):
    from traccia.processors.cost_engine import BUNDLED_PRICING

    if not any(k.startswith("groq/") for k in BUNDLED_PRICING):
        pytest.skip("bundled snapshot not available")
    assert match_model(model, BUNDLED_PRICING, vendor).key == expected


def test_bundled_snapshot_gemini_sdk_prefix():
    from traccia.processors.cost_engine import BUNDLED_PRICING

    result = match_model("models/gemini-2.5-flash", BUNDLED_PRICING, "google_gemini")
    assert result is not None and result.key.rsplit("/", 1)[-1] == "gemini-2.5-flash"
