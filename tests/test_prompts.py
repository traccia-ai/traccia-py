"""Tests for prompt compile parity and load_prompt behavior."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traccia.prompts import (
    CompileError,
    LoadedPrompt,
    PromptFetchError,
    load_prompt,
    prefetch_prompts,
    reset_prompt_cache,
    configure_prompts,
)
import traccia.prompts as prompts_mod
from traccia.prompts.compile import compile_body
from traccia.processors.redaction_processor import redact_attributes


FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "implementation"
    / "prompt-management"
    / "scratch"
    / "f49-compile-fixtures.json"
)


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_prompt_cache()
    prompts_mod._reset_fetch_impl_for_tests()
    configure_prompts(cache_ttl_s=60)
    yield
    reset_prompt_cache()
    prompts_mod._reset_fetch_impl_for_tests()


def test_f49_fixtures_parity():
    data = json.loads(FIXTURES.read_text())
    for case in data["cases"]:
        if case.get("error"):
            with pytest.raises(CompileError) as exc:
                compile_body(case["type"], case["body"], case["variables"])
            assert case["error"] in str(exc.value)
            continue
        out, extras = compile_body(case["type"], case["body"], case["variables"])
        assert out == case["expected"]
        assert list(extras) == list(case.get("warn_extras") or [])


def test_loaded_prompt_compile_text_and_attrs():
    prompt = LoadedPrompt.from_payload(
        {
            "name": "greet",
            "type": "text",
            "version": 3,
            "version_id": "vid-1",
            "label": "production",
            "body": {"text": "Hi {{name}}"},
        }
    )
    span = MagicMock()
    with patch("traccia.context.get_current_span", return_value=span):
        assert prompt.compile(name="Ada") == "Hi Ada"
    span.set_attribute.assert_any_call("traccia.prompt.name", "greet")
    span.set_attribute.assert_any_call("traccia.prompt.version", "3")
    span.set_attribute.assert_any_call("traccia.prompt.label", "production")


def test_cache_hit_avoids_second_fetch():
    calls = {"n": 0}
    payload = {
        "name": "support-reply",
        "type": "chat",
        "version": 1,
        "version_id": "v1",
        "label": "production",
        "body": {"messages": [{"role": "system", "content": "hi"}]},
        "model_config": {},
    }

    def fake_fetch(name, **kwargs):
        calls["n"] += 1
        return payload, '"etag"'

    prompts_mod._set_fetch_impl_for_tests(fake_fetch)
    a = load_prompt("support-reply")
    b = load_prompt("support-reply")
    assert calls["n"] == 1
    assert a.version == b.version == 1
    assert not a.is_stale and not b.is_stale


def test_swr_serves_stale_then_refreshes():
    payloads = [
        {
            "name": "p",
            "type": "text",
            "version": 1,
            "version_id": "a",
            "label": "production",
            "body": {"text": "v1"},
        },
        {
            "name": "p",
            "type": "text",
            "version": 2,
            "version_id": "b",
            "label": "production",
            "body": {"text": "v2"},
        },
    ]
    idx = {"i": 0}

    def fake_fetch(name, **kwargs):
        i = min(idx["i"], len(payloads) - 1)
        idx["i"] += 1
        return payloads[i], None

    prompts_mod._set_fetch_impl_for_tests(fake_fetch)
    configure_prompts(cache_ttl_s=0.05)
    first = load_prompt("p")
    assert first.version == 1
    time.sleep(0.06)
    second = load_prompt("p")
    assert second.is_stale is True
    assert second.version == 1
    # After SWR kickoff, restore a normal TTL so the refreshed entry can be fresh
    configure_prompts(cache_ttl_s=60)
    deadline = time.time() + 2
    third = second
    while time.time() < deadline:
        time.sleep(0.05)
        third = load_prompt("p")
        if third.version == 2:
            break
    else:
        pytest.fail("SWR did not refresh cache")
    assert third.version == 2
    assert third.is_stale is False


def test_fallback_and_is_fallback():
    def boom(name, **kwargs):
        raise PromptFetchError("down")

    prompts_mod._set_fetch_impl_for_tests(boom)
    prompt = load_prompt(
        "missing",
        fallback={"type": "chat", "messages": [{"role": "system", "content": "offline"}]},
    )
    assert prompt.is_fallback is True
    assert prompt.messages[0]["content"] == "offline"
    attrs = prompt.span_attributes()
    assert attrs["traccia.prompt.is_fallback"] is True


def test_redaction_preserves_prompt_identity():
    attrs = {
        "gen_ai.prompt": "email me at a@b.com",
        "traccia.prompt.name": "support-reply",
        "traccia.prompt.version": "12",
        "traccia.prompt.label": "production",
        "traccia.prompt.is_fallback": True,
    }
    out = redact_attributes(attrs)
    assert "[REDACTED_EMAIL]" in out["gen_ai.prompt"]
    assert out["traccia.prompt.name"] == "support-reply"
    assert out["traccia.prompt.version"] == "12"
    assert out["traccia.prompt.label"] == "production"
    assert out["traccia.prompt.is_fallback"] is True


def test_prefetch_warms_cache():
    calls = []

    def fake_fetch(name, **kwargs):
        calls.append(name)
        return {
            "name": name,
            "type": "text",
            "version": 1,
            "version_id": name,
            "label": "production",
            "body": {"text": name},
        }, None

    prompts_mod._set_fetch_impl_for_tests(fake_fetch)
    prefetch_prompts(["a", "b"], jitter_s=0)
    assert calls == ["a", "b"]
    reset_calls = list(calls)
    load_prompt("a")
    assert calls == reset_calls  # cache hit
