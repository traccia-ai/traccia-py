"""Regression tests for instrumentation/anthropic.py (TRC-281).

The fake SDK mirrors the real layout: ``Anthropic.messages`` is a
``cached_property`` and responses carry a typed, non-subscriptable ``Usage``.
"""

import asyncio
import contextlib
import functools
import sys
import types

import pytest

import instrumentation.anthropic as anthropic_mod
from instrumentation.anthropic import patch_anthropic


class FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exception = exc

    def set_status(self, status, message=None):
        self.status = status


class FakeTracer:
    def __init__(self):
        self.spans = []

    @contextlib.contextmanager
    def start_as_current_span(self, name, attributes=None):
        span = FakeSpan()
        span.attributes.update(attributes or {})
        self.spans.append(span)
        yield span


class Usage:
    """Typed usage object: attribute access only, like the SDK's pydantic model."""

    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class Message:
    def __init__(self, usage):
        self.usage = usage
        self.stop_reason = "end_turn"


@pytest.fixture
def fake_sdk(monkeypatch):
    class Messages:
        def create(self, **kwargs):
            return kwargs.pop("_response", Message(Usage(11, 7)))

    class AsyncMessages:
        async def create(self, **kwargs):
            return Message(Usage(11, 7))

    class Anthropic:
        @functools.cached_property
        def messages(self):
            return Messages()

    anthropic_pkg = types.ModuleType("anthropic")
    anthropic_pkg.Anthropic = Anthropic
    resources = types.ModuleType("anthropic.resources")
    messages_mod = types.ModuleType("anthropic.resources.messages")
    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_pkg)
    monkeypatch.setitem(sys.modules, "anthropic.resources", resources)
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages", messages_mod)
    monkeypatch.setattr(anthropic_mod, "_patched", False)
    return types.SimpleNamespace(Anthropic=Anthropic, Messages=Messages, AsyncMessages=AsyncMessages)


@pytest.fixture
def tracer(monkeypatch):
    tracer = FakeTracer()
    monkeypatch.setattr(anthropic_mod, "_get_tracer", lambda name: tracer)
    return tracer


def _assert_usage(span, input_tokens, output_tokens):
    assert span.attributes["llm.usage.input_tokens"] == input_tokens
    assert span.attributes["llm.usage.output_tokens"] == output_tokens
    assert span.attributes["llm.usage.prompt_tokens"] == input_tokens
    assert span.attributes["llm.usage.completion_tokens"] == output_tokens
    assert span.attributes["llm.usage.source"] == "provider_usage"


def test_patches_messages_resource_despite_cached_property_client(fake_sdk):
    assert patch_anthropic() is True
    assert fake_sdk.Messages.create._agent_trace_patched
    assert fake_sdk.AsyncMessages.create._agent_trace_patched


def test_sync_create_records_typed_usage(fake_sdk, tracer):
    patch_anthropic()
    resp = fake_sdk.Anthropic().messages.create(model="claude-x", max_tokens=5, messages=[])

    assert isinstance(resp, Message)
    (span,) = tracer.spans
    assert span.attributes["llm.model"] == "claude-x"
    assert span.attributes["llm.stop_reason"] == "end_turn"
    _assert_usage(span, 11, 7)


def test_sync_create_records_dict_usage(fake_sdk, tracer):
    patch_anthropic()
    response = {"usage": {"input_tokens": 3, "output_tokens": 2}}
    fake_sdk.Anthropic().messages.create(model="claude-x", messages=[], _response=response)

    _assert_usage(tracer.spans[0], 3, 2)


def test_async_create_records_typed_usage(fake_sdk, tracer):
    patch_anthropic()
    resp = asyncio.run(fake_sdk.AsyncMessages().create(model="claude-x", max_tokens=5, messages=[]))

    assert isinstance(resp, Message)
    _assert_usage(tracer.spans[0], 11, 7)
