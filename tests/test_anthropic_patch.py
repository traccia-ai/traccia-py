"""Regression tests for direct Anthropic SDK resource instrumentation."""

import asyncio
import contextlib
import gc
import json
import sys
import types

import pytest

import instrumentation.anthropic as anthropic_mod


class FakeSpan:
    def __init__(self, attributes=None):
        self.attributes = dict(attributes or {})
        self.exception = None
        self.status = None
        self.exited = 0

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exception = exc

    def set_status(self, status, description=None):
        self.status = (status, description)


class FakeTracer:
    def __init__(self):
        self.spans = []

    @contextlib.contextmanager
    def start_as_current_span(self, name, attributes=None):
        span = FakeSpan(attributes)
        span.name = name
        self.spans.append(span)
        try:
            yield span
        finally:
            span.exited += 1


def _install_fake_sdk(monkeypatch, sync_cls, async_cls):
    """Install the concrete resource module paths used by patch_anthropic."""
    modules = {
        "anthropic": types.ModuleType("anthropic"),
        "anthropic.resources": types.ModuleType("anthropic.resources"),
        "anthropic.resources.messages": types.ModuleType("anthropic.resources.messages"),
        "anthropic.resources.messages.messages": types.ModuleType(
            "anthropic.resources.messages.messages"
        ),
        "anthropic.resources.beta": types.ModuleType("anthropic.resources.beta"),
        "anthropic.resources.beta.messages": types.ModuleType(
            "anthropic.resources.beta.messages"
        ),
        "anthropic.resources.beta.messages.messages": types.ModuleType(
            "anthropic.resources.beta.messages.messages"
        ),
    }
    modules["anthropic.resources.messages.messages"].Messages = sync_cls
    modules["anthropic.resources.messages.messages"].AsyncMessages = async_cls
    modules["anthropic.resources.beta.messages.messages"].Messages = sync_cls
    modules["anthropic.resources.beta.messages.messages"].AsyncMessages = async_cls
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(autouse=True)
def reset_patch_state():
    anthropic_mod._patched = False
    yield
    anthropic_mod._patched = False


def test_direct_messages_create_is_patched_and_reads_object_usage(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                model=kwargs["model"],
                stop_reason="end_turn",
                usage=types.SimpleNamespace(input_tokens=12, output_tokens=5),
            )

    class AsyncMessages:
        async def create(self, **kwargs):
            return types.SimpleNamespace(usage=None)

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)

    assert anthropic_mod.patch_anthropic() is True
    client = types.SimpleNamespace(messages=Messages())
    response = client.messages.create(model="claude-test")

    assert response.model == "claude-test"
    assert len(tracer.spans) == 1
    attrs = tracer.spans[0].attributes
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-test"
    assert attrs["llm.usage.prompt_tokens"] == 12
    assert attrs["llm.usage.completion_tokens"] == 5
    assert attrs["llm.stop_reason"] == "end_turn"


def test_async_messages_create_is_awaited_and_traced(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(usage=None)

    class AsyncMessages:
        async def create(self, **kwargs):
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(input_tokens=2, output_tokens=3)
            )

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)

    assert anthropic_mod.patch_anthropic() is True
    response = asyncio.run(AsyncMessages().create(model="claude-test"))

    assert response.usage.output_tokens == 3
    assert len(tracer.spans) == 1
    assert tracer.spans[0].attributes["llm.usage.input_tokens"] == 2


def test_patch_anthropic_is_idempotent(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(usage=None)

    class AsyncMessages:
        async def create(self, **kwargs):
            return types.SimpleNamespace(usage=None)

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    assert anthropic_mod.patch_anthropic() is True
    first = Messages.create
    assert anthropic_mod.patch_anthropic() is True
    assert Messages.create is first


def test_create_records_request_response_and_extended_usage(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return {
                "id": "msg_123",
                "model": "claude-response",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "I will call a tool."},
                    {"type": "tool_use", "name": "lookup", "id": "tool_123"},
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 2,
                    "output_tokens_details": {"thinking_tokens": 4},
                },
            }

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    metrics = []
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_compute_cost", lambda *args: 0.25)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: metrics.append((args, kwargs)))

    assert anthropic_mod.patch_anthropic() is True
    Messages().create(
        model="claude-request",
        system="You are concise.",
        messages=[{"role": "user", "content": "Find the weather"}],
        tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
    )

    attrs = tracer.spans[0].attributes
    assert attrs["llm.prompt"] == "user: Find the weather"
    assert '"You are concise."' == attrs["llm.anthropic.system"]
    assert '"lookup"' in attrs["llm.anthropic.tools"]
    assert attrs["llm.model"] == "claude-response"
    assert attrs["gen_ai.response.model"] == "claude-response"
    assert attrs["llm.response.id"] == "msg_123"
    assert attrs["gen_ai.response.finish_reasons"] == ["tool_use"]
    assert attrs["llm.completion"] == "I will call a tool.\n[tool_use lookup (tool_123)]"
    assert attrs["llm.usage.cache_creation_input_tokens"] == 3
    assert attrs["llm.usage.cache_read_input_tokens"] == 2
    assert attrs["llm.usage.thinking_tokens"] == 4
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert metrics[0][0][0:2] == ("claude-response", {
        "input_tokens": 12,
        "output_tokens": 5,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 2,
        "thinking_tokens": 4,
    })
    assert metrics[0][0][3] == 0.25


def test_request_payload_is_bounded_and_does_not_capture_credentials(monkeypatch):
    monkeypatch.setattr(anthropic_mod, "_limit", lambda: 30)
    attrs = anthropic_mod._start_attrs(
        "claude-test",
        (),
        {
            "messages": [{"role": "user", "content": "x" * 50}],
            "system": {"text": "safe", "api_key": "secret", "headers": {"Authorization": "secret"}},
            "tools": [{"input": b"binary", "authorization": "secret", "name": "tool"}],
        },
    )

    assert attrs["llm.prompt"] == "user: " + "x" * 24
    assert "secret" not in attrs["llm.anthropic.system"]
    assert "headers" not in attrs["llm.anthropic.system"]
    assert "secret" not in attrs["llm.anthropic.tools"]
    assert "[omitted]" in attrs["llm.anthropic.tools"]


def test_sync_create_error_records_error_span_and_metrics(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            raise RuntimeError("request rejected")

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer, metrics = FakeTracer(), []
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: metrics.append((args, kwargs)))

    anthropic_mod.patch_anthropic()
    with pytest.raises(RuntimeError, match="request rejected"):
        Messages().create(model="claude-test")

    span = tracer.spans[0]
    assert isinstance(span.exception, RuntimeError)
    assert span.status[1] == "request rejected"
    assert metrics[0][1] == {"error": True}


def test_sync_stream_create_error_records_error_span(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            raise RuntimeError("stream request rejected")

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer, metrics = FakeTracer(), []
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(
        anthropic_mod,
        "_record_metrics",
        lambda *args, **kwargs: metrics.append((args, kwargs)),
    )

    anthropic_mod.patch_anthropic()
    with pytest.raises(RuntimeError, match="stream request rejected"):
        Messages().create(model="claude-test", stream=True)

    assert isinstance(tracer.spans[0].exception, RuntimeError)
    assert tracer.spans[0].status[1] == "stream request rejected"
    assert tracer.spans[0].exited == 1
    assert metrics[0][1] == {"error": True}


def test_sync_stream_create_finalizes_once_and_uses_message_start_response(monkeypatch):
    response = {
        "id": "msg_stream",
        "model": "claude-stream",
        "content": [{"type": "text", "text": "complete response"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 9},
    }

    class Messages:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return iter([{"type": "message_start", "message": response}, {"type": "message_stop"}])

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    events = list(Messages().create(model="claude-request", stream=True))

    assert len(events) == 2
    span = tracer.spans[0]
    assert span.exited == 1
    assert span.attributes["llm.response.id"] == "msg_stream"
    assert span.attributes["llm.model"] == "claude-stream"
    assert span.attributes["llm.completion"] == "complete response"


def test_raw_stream_accumulates_text_and_message_delta_usage(monkeypatch):
    initial_message = {
        "id": "msg_raw_stream",
        "model": "claude-stream",
        "content": [],
        "usage": {"input_tokens": 7, "output_tokens": 0},
    }
    events = [
        {"type": "message_start", "message": initial_message},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]

    class Messages:
        def create(self, **kwargs):
            return iter(events)

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer, metrics = FakeTracer(), []
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(
        anthropic_mod,
        "_record_metrics",
        lambda *args, **kwargs: metrics.append((args, kwargs)),
    )

    anthropic_mod.patch_anthropic()
    list(Messages().create(model="claude-request", stream=True))

    span = tracer.spans[0]
    assert span.attributes["llm.completion"] == "hello world"
    assert span.attributes["llm.stop_reason"] == "end_turn"
    assert span.attributes["llm.usage.input_tokens"] == 7
    assert span.attributes["llm.usage.output_tokens"] == 9
    assert metrics[0][0][1]["output_tokens"] == 9


def test_helper_text_stream_finalizes_from_accumulated_snapshot(monkeypatch):
    final_message = {
        "id": "msg_helper_stream",
        "model": "claude-helper",
        "content": [{"type": "text", "text": "hello world"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 6},
    }

    class MessageStream:
        current_message_snapshot = final_message

        def __init__(self):
            self.text_stream = iter(["hello ", "world"])
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def close(self):
            self.closed = True

    class MessageStreamManager:
        def __init__(self):
            self.stream = MessageStream()
            self.exited = False

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            self.exited = True
            self.stream.close()

    manager = MessageStreamManager()

    class Messages:
        def stream(self, **kwargs):
            return manager

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    with Messages().stream(model="claude-request") as stream:
        assert "".join(stream.text_stream) == "hello world"

    span = tracer.spans[0]
    assert span.attributes["llm.completion"] == "hello world"
    assert span.attributes["llm.usage.output_tokens"] == 6
    assert manager.exited is True
    assert span.exited == 1


def test_async_helper_text_stream_finalizes_from_snapshot(monkeypatch):
    final_message = {
        "id": "msg_async_text_stream",
        "model": "claude-async-helper",
        "content": [{"type": "text", "text": "async hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }

    class AsyncMessageStream:
        current_message_snapshot = final_message

        def __init__(self):
            self.text_stream = self._text_stream()

        async def _text_stream(self):
            yield "async "
            yield "hello"

        async def close(self):
            pass

    class AsyncMessageStreamManager:
        def __init__(self):
            self.stream = AsyncMessageStream()

        async def __aenter__(self):
            return self.stream

        async def __aexit__(self, exc_type, exc, tb):
            await self.stream.close()

    class Messages:
        pass

    class AsyncMessages:
        def stream(self, **kwargs):
            return AsyncMessageStreamManager()

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()

    async def consume():
        async with AsyncMessages().stream(model="claude-request") as stream:
            return "".join([part async for part in stream.text_stream])

    assert asyncio.run(consume()) == "async hello"
    attrs = tracer.spans[0].attributes
    assert attrs["llm.completion"] == "async hello"
    assert attrs["llm.response.id"] == "msg_async_text_stream"
    assert attrs["llm.usage.output_tokens"] == 3


def test_async_helper_get_final_message_finalizes_from_snapshot(monkeypatch):
    final_message = {
        "id": "msg_async_helper",
        "model": "claude-async-helper",
        "content": [{"type": "text", "text": "async response"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }

    class AsyncMessageStream:
        current_message_snapshot = final_message

        async def get_final_message(self):
            return final_message

        async def close(self):
            pass

    class AsyncMessageStreamManager:
        def __init__(self):
            self.stream = AsyncMessageStream()
            self.exited = False

        async def __aenter__(self):
            return self.stream

        async def __aexit__(self, exc_type, exc, tb):
            self.exited = True
            await self.stream.close()

    manager = AsyncMessageStreamManager()

    class Messages:
        pass

    class AsyncMessages:
        def stream(self, **kwargs):
            return manager

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()

    async def consume():
        async with AsyncMessages().stream(model="claude-request") as stream:
            return await stream.get_final_message()

    response = asyncio.run(consume())
    assert response["id"] == "msg_async_helper"
    span = tracer.spans[0]
    assert span.attributes["llm.completion"] == "async response"
    assert span.attributes["llm.usage.output_tokens"] == 3
    assert manager.exited is True
    assert span.exited == 1


def test_direct_stream_method_and_close_finalize_span(monkeypatch):
    class ClosableStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            self.closed = True

    class Messages:
        def stream(self, **kwargs):
            return ClosableStream()

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    assert anthropic_mod.patch_anthropic() is True
    stream = Messages().stream(model="claude-test", messages=[{"role": "user", "content": "hi"}])
    stream.close()

    assert stream.closed is True
    assert tracer.spans[0].exited == 1
    assert tracer.spans[0].attributes["llm.prompt"] == "user: hi"


def test_direct_stream_method_error_records_error_span(monkeypatch):
    class Messages:
        def stream(self, **kwargs):
            raise RuntimeError("invalid stream request")

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer, metrics = FakeTracer(), []
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(
        anthropic_mod,
        "_record_metrics",
        lambda *args, **kwargs: metrics.append((args, kwargs)),
    )

    anthropic_mod.patch_anthropic()
    with pytest.raises(RuntimeError, match="invalid stream request"):
        Messages().stream(model="claude-test")

    assert isinstance(tracer.spans[0].exception, RuntimeError)
    assert tracer.spans[0].exited == 1
    assert metrics[0][1] == {"error": True}


def test_sync_stream_manager_returns_actual_stream_from_context(monkeypatch):
    response = {"model": "claude-manager", "usage": {"input_tokens": 3, "output_tokens": 4}}

    class MessageStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter([
                {"type": "message_start", "message": response},
                {"type": "message_stop"},
            ])

        def close(self):
            self.closed = True

    class MessageStreamManager:
        def __init__(self):
            self.stream = None
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            self.stream = MessageStream()
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            self.exited = True
            self.stream.close()

    manager = MessageStreamManager()

    class Messages:
        def stream(self, **kwargs):
            return manager

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    with Messages().stream(model="claude-request") as stream:
        events = list(stream)

    assert len(events) == 2
    assert manager.entered is True
    assert manager.exited is True
    assert manager.stream.closed is True
    assert tracer.spans[0].attributes["llm.usage.completion_tokens"] == 4
    assert tracer.spans[0].exited == 1


def test_async_stream_manager_returns_actual_stream_from_context(monkeypatch):
    response = {"model": "claude-async-manager", "usage": {"input_tokens": 5, "output_tokens": 6}}

    class AsyncMessageStream:
        def __init__(self):
            self.events = iter([
                {"type": "message_start", "message": response},
                {"type": "message_stop"},
            ])
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

        async def close(self):
            self.closed = True

    class AsyncMessageStreamManager:
        def __init__(self):
            self.stream = None
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            self.stream = AsyncMessageStream()
            return self.stream

        async def __aexit__(self, exc_type, exc, tb):
            self.exited = True
            await self.stream.close()

    manager = AsyncMessageStreamManager()

    class Messages:
        pass

    class AsyncMessages:
        def stream(self, **kwargs):
            return manager

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()

    async def consume():
        async with AsyncMessages().stream(model="claude-request") as stream:
            return [event async for event in stream]

    events = asyncio.run(consume())
    assert len(events) == 2
    assert manager.entered is True
    assert manager.exited is True
    assert manager.stream.closed is True
    assert tracer.spans[0].attributes["llm.usage.completion_tokens"] == 6
    assert tracer.spans[0].exited == 1


def test_stream_iteration_error_marks_span_as_failed(monkeypatch):
    class FailingStream:
        def __iter__(self):
            yield {"type": "message_start", "message": {"usage": None}}
            raise ValueError("stream failed")

    class Messages:
        def stream(self, **kwargs):
            return FailingStream()

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer, metrics = FakeTracer(), []
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: metrics.append((args, kwargs)))

    anthropic_mod.patch_anthropic()
    with pytest.raises(ValueError, match="stream failed"):
        list(Messages().stream(model="claude-test"))

    assert isinstance(tracer.spans[0].exception, ValueError)
    assert tracer.spans[0].exited == 1
    assert metrics[0][1] == {"error": True}


def test_async_stream_create_is_traced(monkeypatch):
    response = {
        "model": "claude-async-stream",
        "content": [{"type": "text", "text": "async response"}],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    class AsyncEventStream:
        def __init__(self):
            self.events = iter([{"type": "message_start", "message": response}, {"type": "message_stop"}])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration

    class Messages:
        pass

    class AsyncMessages:
        async def create(self, **kwargs):
            return AsyncEventStream()

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()

    async def consume():
        stream = await AsyncMessages().create(model="claude-request", stream=True)
        return [event async for event in stream]

    assert len(asyncio.run(consume())) == 2
    span = tracer.spans[0]
    assert span.exited == 1
    assert span.attributes["llm.model"] == "claude-async-stream"
    assert span.attributes["llm.usage.completion_tokens"] == 2


def test_sync_helper_get_final_text_uses_snapshot_for_response_metadata(monkeypatch):
    final_message = {
        "id": "msg_sync_final_text",
        "model": "claude-sync-final-text",
        "content": [{"type": "text", "text": "final answer"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 8, "output_tokens": 4},
    }

    class MessageStream:
        current_message_snapshot = final_message

        def get_final_text(self):
            return "final answer"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class MessageStreamManager:
        def __enter__(self):
            return MessageStream()

        def __exit__(self, exc_type, exc, tb):
            return None

    class Messages:
        def stream(self, **kwargs):
            return MessageStreamManager()

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    with Messages().stream(model="claude-request") as stream:
        assert stream.get_final_text() == "final answer"

    attrs = tracer.spans[0].attributes
    assert attrs["llm.completion"] == "final answer"
    assert attrs["llm.response.id"] == "msg_sync_final_text"
    assert attrs["llm.usage.output_tokens"] == 4


def test_sync_helper_until_done_uses_current_snapshot(monkeypatch):
    final_message = {
        "id": "msg_sync_until_done",
        "model": "claude-sync-until-done",
        "content": [{"type": "text", "text": "consumed"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }

    class MessageStream:
        current_message_snapshot = None

        def until_done(self):
            type(self).current_message_snapshot = final_message

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class MessageStreamManager:
        def __enter__(self):
            return MessageStream()

        def __exit__(self, exc_type, exc, tb):
            return None

    class Messages:
        def stream(self, **kwargs):
            return MessageStreamManager()

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    with Messages().stream(model="claude-request") as stream:
        stream.until_done()

    attrs = tracer.spans[0].attributes
    assert attrs["llm.completion"] == "consumed"
    assert attrs["llm.response.id"] == "msg_sync_until_done"
    assert attrs["llm.usage.output_tokens"] == 2


def test_raw_stream_tool_use_accumulates_tool_metadata(monkeypatch):
    initial_message = {
        "id": "msg_tool_stream",
        "model": "claude-tool-stream",
        "content": [],
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }
    class Stream:
        def __iter__(self):
            yield {"type": "message_start", "message": initial_message}
            yield {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "lookup", "id": "tool-1"},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 7},
            }
            yield {"type": "message_stop"}

    class Messages:
        def create(self, **kwargs):
            return Stream()

    class AsyncMessages:
        pass

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    list(Messages().create(model="claude-request", stream=True))

    attrs = tracer.spans[0].attributes
    assert attrs["llm.completion"] == "[tool_use lookup (tool-1)]"
    assert attrs["llm.response.id"] == "msg_tool_stream"
    assert attrs["llm.usage.output_tokens"] == 7


def test_async_close_waits_for_sdk_close_before_finalizing(monkeypatch):
    class AsyncStream:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    stream = AsyncStream()

    class Messages:
        pass

    class AsyncMessages:
        def stream(self, **kwargs):
            return stream

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()

    async def consume():
        traced_stream = AsyncMessages().stream(model="claude-request")
        await traced_stream.close()
        return traced_stream

    traced_stream = asyncio.run(consume())
    assert stream.closed is True
    assert tracer.spans[0].exited == 1
    assert traced_stream._done is True


def test_async_close_error_marks_span_as_failed(monkeypatch):
    class AsyncStream:
        async def close(self):
            raise RuntimeError("close failed")

    class Messages:
        pass

    class AsyncMessages:
        def stream(self, **kwargs):
            return AsyncStream()

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()

    async def consume():
        traced_stream = AsyncMessages().stream(model="claude-request")
        with pytest.raises(RuntimeError, match="close failed"):
            await traced_stream.close()

    asyncio.run(consume())
    assert isinstance(tracer.spans[0].exception, RuntimeError)
    assert tracer.spans[0].status[1] == "close failed"
    assert tracer.spans[0].exited == 1


def test_patch_returns_false_when_anthropic_cannot_be_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert anthropic_mod.patch_anthropic() is False


class _BrokenTracer:
    """Every entry point raises - stands in for a misconfigured tracer."""

    def start_span(self, *args, **kwargs):
        raise RuntimeError("tracer unavailable")

    def start_as_current_span(self, *args, **kwargs):
        raise RuntimeError("tracer unavailable")


def test_instrumentation_failure_does_not_break_sync_create(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return {
                "id": "ok",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: _BrokenTracer())
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    assert anthropic_mod.patch_anthropic() is True
    result = Messages().create(model="claude-x", messages=[{"role": "user", "content": "hi"}])
    assert result["content"][0]["text"] == "hi"


def test_instrumentation_failure_does_not_break_streaming(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return iter(
                [{"type": "message_start", "message": {"id": "m"}}, {"type": "message_stop"}]
            )

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: _BrokenTracer())
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    stream = Messages().create(model="claude-x", stream=True)
    assert [event["type"] for event in stream] == ["message_start", "message_stop"]


def test_finish_failure_does_not_break_sync_create(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return {
                "id": "ok",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)

    def boom(*args, **kwargs):
        raise RuntimeError("finish blew up")

    monkeypatch.setattr(anthropic_mod, "_finish", boom)

    anthropic_mod.patch_anthropic()
    result = Messages().create(model="claude-x", messages=[{"role": "user", "content": "hi"}])
    assert result["content"][0]["text"] == "done"
    assert tracer.spans[0].exited == 1


def test_usage_flattens_cache_creation_object(monkeypatch):
    class CacheCreation:
        ephemeral_5m_input_tokens = 3
        ephemeral_1h_input_tokens = 1

    class Usage:
        input_tokens = 10
        output_tokens = 4
        cache_creation = CacheCreation()

    class Response:
        id = "msg_cc"
        model = "claude-cc"
        stop_reason = "end_turn"
        content = [{"type": "text", "text": "ok"}]
        usage = Usage()

    class Messages:
        def create(self, **kwargs):
            return Response()

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    Messages().create(model="claude-cc", messages=[{"role": "user", "content": "hi"}])

    attrs = tracer.spans[0].attributes
    assert attrs["llm.usage.cache_creation_ephemeral_5m_input_tokens"] == 3
    assert attrs["llm.usage.cache_creation_ephemeral_1h_input_tokens"] == 1
    assert "llm.usage.cache_creation" not in attrs
    assert "gen_ai.usage.cache_creation" not in attrs
    for key, value in attrs.items():
        if key.startswith(("llm.usage.", "gen_ai.usage.")):
            assert isinstance(value, (int, float, str)) and not isinstance(value, bool)


def test_finish_skips_non_scalar_usage_values(monkeypatch):
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(anthropic_mod, "_compute_cost", lambda *args, **kwargs: None)

    span = FakeSpan()
    anthropic_mod._finish(
        span,
        {"model": "m", "usage": {"input_tokens": 5, "output_tokens": 2}},
        "m",
        0.0,
        usage={"input_tokens": 5, "output_tokens": 2, "junk": object(), "flag": True},
    )
    assert span.attributes["llm.usage.input_tokens"] == 5
    assert span.attributes["gen_ai.usage.output_tokens"] == 2
    assert "llm.usage.junk" not in span.attributes
    assert "llm.usage.flag" not in span.attributes


def test_oversized_json_request_attrs_stay_valid_json(monkeypatch):
    monkeypatch.setattr(anthropic_mod, "_limit", lambda: 120)
    attrs = anthropic_mod._start_attrs(
        "claude-x",
        (),
        {
            "messages": [{"role": "user", "content": "x" * 400}] * 5,
            "system": "s" * 400,
            "tools": [{"name": "t", "description": "d" * 400}],
        },
    )
    for key in ("llm.anthropic.messages", "llm.anthropic.system", "llm.anthropic.tools"):
        parsed = json.loads(attrs[key])  # must not raise
        assert isinstance(parsed, (dict, list))


def test_abandoned_stream_proxy_finalizes_span_on_del(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return iter(
                [{"type": "message_start", "message": {"id": "m"}}, {"type": "message_stop"}]
            )

    class AsyncMessages:
        async def create(self, **kwargs):
            return {"usage": None}

    _install_fake_sdk(monkeypatch, Messages, AsyncMessages)
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    monkeypatch.setattr(anthropic_mod, "_record_metrics", lambda *args, **kwargs: None)

    anthropic_mod.patch_anthropic()
    proxy = Messages().create(model="claude-x", stream=True)
    assert isinstance(proxy, anthropic_mod._StreamProxy)
    assert tracer.spans[0].exited == 0

    del proxy
    gc.collect()

    assert tracer.spans[0].exited == 1
