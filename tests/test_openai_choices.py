"""OpenAI choice extraction through SDK decoding and in-memory OTel export."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import traccia
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traccia.instrumentation import openai as instrumentation
from traccia.processors.guardrail_detector import GuardrailDetectorProcessor
from traccia.tracer.provider import TracerProvider


@pytest.mark.parametrize(
    "value,path,expected",
    [
        (
            {"choices": [{"message": {"content": "answer"}}]},
            "choices.0.message.content",
            "answer",
        ),
        (
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]),
            "choices.0.finish_reason",
            "stop",
        ),
        (({"value": 0},), "0.value", 0),
        ([False], "0", False),
        ({"0": "mapping key"}, "0", "mapping key"),
        ([], "0", "fallback"),
        ([1], "1", "fallback"),
        ([1], "-1", "fallback"),
        ([1], "name", "fallback"),
        ([1], "9" * 5000, "fallback"),
        ([None], "0", "fallback"),
        ([{}], "0.missing.value", "fallback"),
    ],
)
def test_safe_get_indexed_paths(value: Any, path: str, expected: Any) -> None:
    """Preserve falsy values and mapping keys; unavailable paths use the default."""
    assert instrumentation._safe_get(value, path, "fallback") == expected


@pytest.mark.parametrize("mode", ["text", "filtered", "tool_only", "empty"])
def test_sdk_chat_choice_attributes(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Retain SDK choices through real HTTP decoding and OTel span export."""
    openai = pytest.importorskip("openai")
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    from openai.resources.chat.completions import Completions

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(GuardrailDetectorProcessor(heuristics_enabled=False))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(traccia, "_global_tracer_provider", provider)
    monkeypatch.setattr(instrumentation, "_patched", False)
    monkeypatch.setattr(Completions, "create", Completions.create)
    assert instrumentation.patch_openai()
    content = "Fixture response" if mode == "text" else None
    finish = {
        "text": "stop",
        "filtered": "content_filter",
        "tool_only": "tool_calls",
        "empty": None,
    }[mode]
    requests = []

    def respond(request: Any) -> Any:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        message = {"role": "assistant", "content": content}
        if mode == "tool_only":
            message["tool_calls"] = [
                {
                    "id": "call_fixture",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": []
                if mode == "empty"
                else [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                },
            },
        )

    try:
        with openai.OpenAI(
            api_key="fixture-key",
            base_url="https://provider.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Fixture prompt"}],
            )
        assert len(requests) == 1
        if mode == "empty":
            assert result.choices == []
        else:
            assert result.choices[0].message.content == content
            assert result.choices[0].finish_reason == finish
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attributes = spans[0].attributes
        assert attributes.get("llm.completion") == content
        assert attributes.get("llm.finish_reason") == finish
        assert attributes["llm.usage.prompt_tokens"] == 12
        assert attributes["llm.usage.completion_tokens"] == 5
        assert attributes["llm.usage.total_tokens"] == 17
        if mode == "filtered":
            assert attributes["guardrail.finding.count"] == 1
            assert any(
                f["name"] == "provider_content_filter"
                for f in json.loads(attributes["guardrail.findings"])
            )
    finally:
        provider.shutdown()
