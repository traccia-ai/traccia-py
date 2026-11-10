"""Responses API completion and finish metadata through SDK decoding and OTel export (TRC-282).

A real ``AsyncOpenAI`` client decodes each fixture into the SDK's typed ``Response`` over a
mock HTTP transport, so these tests exercise the same objects the instrumentation sees in
production: ``output`` holds ``message`` items whose ``content`` holds ``output_text`` parts.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
import traccia
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traccia.instrumentation import openai as instrumentation
from traccia.processors.guardrail_detector import GuardrailDetectorProcessor
from traccia.tracer.provider import TracerProvider


def _message(*parts: Dict[str, Any], msg_id: str = "msg_fixture") -> Dict[str, Any]:
    return {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "status": "completed",
        "content": list(parts),
    }


def _text(text: str) -> Dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


def _refusal(text: str) -> Dict[str, Any]:
    return {"type": "refusal", "refusal": text}


_FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_fixture",
    "call_id": "call_fixture",
    "name": "lookup",
    "arguments": "{}",
    "status": "completed",
}

_REASONING = {"type": "reasoning", "id": "rs_fixture", "summary": []}

_WEB_SEARCH = {
    "type": "web_search_call",
    "id": "ws_fixture",
    "status": "completed",
    "action": {"type": "search", "query": "fixture"},
}


def _body(
    output: List[Dict[str, Any]],
    status: str = "completed",
    incomplete_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 0,
        "model": "gpt-4o-mini",
        "status": status,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 5,
            "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


@pytest.fixture
def traced(monkeypatch: pytest.MonkeyPatch):
    """Patch AsyncResponses.create onto a fresh provider; yield the exporter."""
    pytest.importorskip("openai")
    from openai.resources.responses import AsyncResponses

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(GuardrailDetectorProcessor(heuristics_enabled=False))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(traccia, "_global_tracer_provider", provider)
    monkeypatch.setattr(instrumentation, "_responses_patched", False)
    # monkeypatch restores the unwrapped create after the test.
    monkeypatch.setattr(AsyncResponses, "create", AsyncResponses.create)
    assert instrumentation.patch_openai_responses()
    try:
        yield exporter
    finally:
        provider.shutdown()


def _call(body: Dict[str, Any]) -> Any:
    import openai

    try:
        import httpx2 as httpx
    except ImportError:
        import httpx

    requests = []

    def respond(request: Any) -> Any:
        requests.append(request)
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json=body)

    async def go() -> Any:
        async with openai.AsyncOpenAI(
            api_key="fixture-key",
            base_url="https://provider.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            return await client.responses.create(model="gpt-4o-mini", input="Fixture prompt")

    result = asyncio.run(go())
    assert len(requests) == 1
    return result


def _only_span(exporter: InMemorySpanExporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "llm.openai.responses"
    return spans[0].attributes


@pytest.mark.parametrize(
    "output,status,incomplete_reason,completion,finish_reason",
    [
        pytest.param([_message(_text("hello"))], "completed", None, "hello", "stop", id="text"),
        pytest.param(
            [_message(_text("Hello, "), _text("world"))],
            "completed",
            None,
            "Hello, world",
            "stop",
            id="multi_part",
        ),
        pytest.param(
            [_REASONING, _message(_text("first"), msg_id="m1"), _WEB_SEARCH, _message(_text(" second"), msg_id="m2")],
            "completed",
            None,
            "first second",
            "stop",
            id="mixed_items",
        ),
        pytest.param([_FUNCTION_CALL], "completed", None, None, "tool_calls", id="tool_only"),
        pytest.param(
            [_message(_text("Looking that up.")), _FUNCTION_CALL],
            "completed",
            None,
            "Looking that up.",
            "tool_calls",
            id="text_and_tool",
        ),
        pytest.param(
            [_message(_text("truncat"))],
            "incomplete",
            "max_output_tokens",
            "truncat",
            "length",
            id="max_output_tokens",
        ),
        pytest.param(
            [_message(_refusal("I can't help with that."))],
            "completed",
            None,
            "I can't help with that.",
            "stop",
            id="refusal",
        ),
        pytest.param([], "incomplete", "content_filter", None, "content_filter", id="content_filter"),
        pytest.param([], "failed", None, None, None, id="failed"),
        pytest.param([], "completed", None, None, "stop", id="empty"),
    ],
)
def test_responses_completion_and_finish_reason(
    traced: InMemorySpanExporter,
    output: List[Dict[str, Any]],
    status: str,
    incomplete_reason: Optional[str],
    completion: Optional[str],
    finish_reason: Optional[str],
) -> None:
    result = _call(_body(output, status=status, incomplete_reason=incomplete_reason))
    attributes = _only_span(traced)

    assert attributes.get("llm.completion") == completion
    assert attributes.get("llm.finish_reason") == finish_reason
    assert attributes["llm.response.status"] == status
    # Text spans carry exactly what the SDK itself reports as the response text.
    if output and any(p.get("type") == "output_text" for i in output for p in i.get("content", [])):
        assert attributes["llm.completion"] == result.output_text
    # Usage is unaffected.
    assert attributes["llm.usage.prompt_tokens"] == 12
    assert attributes["llm.usage.completion_tokens"] == 5
    assert attributes["llm.usage.total_tokens"] == 17


def test_responses_content_filter_reaches_guardrail_detector(traced: InMemorySpanExporter) -> None:
    """The derived finish reason feeds the same provider content-filter finding as Chat."""
    _call(_body([], status="incomplete", incomplete_reason="content_filter"))
    attributes = _only_span(traced)
    findings = json.loads(attributes["guardrail.findings"])
    assert any(f["name"] == "provider_content_filter" for f in findings)


def test_responses_completion_is_truncated(traced: InMemorySpanExporter) -> None:
    _call(_body([_message(_text("x" * 5000))]))
    assert _only_span(traced)["llm.completion"] == "x" * 2000


@pytest.mark.parametrize(
    "resp,completion,finish_reason",
    [
        # Plain dicts (e.g. a proxied or pre-parsed response) follow the same rules.
        (
            {"status": "completed", "output": [_message(_text("dict text"))]},
            "dict text",
            "stop",
        ),
        # The pre-fix shape (a top-level "text" part) is still accepted.
        (
            {"status": "completed", "output": [{"content": [{"type": "text", "text": "legacy"}]}]},
            "legacy",
            "stop",
        ),
        ({"status": "incomplete", "incomplete_details": {"reason": "something_new"}, "output": []}, None, "something_new"),
        ({"status": "incomplete", "output": []}, None, None),
        ({"status": "in_progress", "output": None}, None, None),
        ({"output": [_message(_text(""))]}, None, None),
        (None, None, None),
    ],
)
def test_responses_extractors_on_untyped_input(resp: Any, completion: Optional[str], finish_reason: Optional[str]) -> None:
    assert instrumentation._responses_completion(resp) == completion
    assert instrumentation._responses_finish_reason(resp) == finish_reason
