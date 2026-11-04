"""Regression tests for instrumentation/anthropic.py (TRC-281).

The fake SDK mirrors the real layout: ``Anthropic.messages`` is a
``cached_property`` and responses carry a typed, non-subscriptable ``Usage``.
"""

import asyncio
import contextvars
import contextlib
import functools
import os
import signal
import sys
import threading
import time
import types

import pytest

import instrumentation.anthropic as anthropic_mod
from instrumentation.anthropic import patch_anthropic
from traccia import runtime_config


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


# -- governance I/O in the async wrapper ------------------------------------------------------

DECISION = {"effect": "allow", "obligations": {"reserved_usd": 0.5}}
TARGET = ("tr_key", "http://traccia.test", "agent-a")


@pytest.fixture
def slow_governance(monkeypatch):
    """A policy check and settle that block their thread for 250 ms, like a slow policy server."""
    import traccia.governance.pep as pep

    calls = types.SimpleNamespace(checks=[], settles=[], agent=[])

    def enforce_llm_call(kwargs, vendor=None, api=None):
        time.sleep(0.25)
        calls.checks.append(vendor)
        calls.agent.append(_AGENT.get())
        return DECISION

    def finish_llm_call(decision, **kwargs):
        if not decision:
            return  # as settle_policy(): nothing to settle
        time.sleep(0.25)
        calls.settles.append(kwargs)

    monkeypatch.setattr(pep, "enforce_llm_call", enforce_llm_call)
    monkeypatch.setattr(pep, "finish_llm_call", finish_llm_call)
    monkeypatch.setattr(pep, "settle_target", lambda: TARGET)
    monkeypatch.setattr(runtime_config, "pep_enabled", lambda: True)
    return calls


_AGENT = contextvars.ContextVar("test_agent", default=None)


def _timer_lateness(coro_factory):
    """Run a coroutine beside a 10 ms timer; return how late the timer fired, in seconds."""

    async def main():
        loop = asyncio.get_running_loop()
        fired = loop.create_future()
        start = loop.time()
        loop.call_later(0.01, lambda: fired.set_result(loop.time() - start))
        task = asyncio.ensure_future(coro_factory())
        elapsed = await fired
        await task
        return elapsed - 0.01

    return asyncio.run(main())


def _flush_settles():
    from traccia.governance.pep import flush_pending_settles

    flush_pending_settles()


def test_async_policy_check_and_settle_do_not_block_the_event_loop(fake_sdk, tracer, slow_governance):
    patch_anthropic()
    _AGENT.set("agent-a")

    lateness = _timer_lateness(lambda: fake_sdk.AsyncMessages().create(model="claude-x", messages=[]))
    _flush_settles()

    assert lateness < 0.1  # the check and settle each take 250 ms, off the loop
    assert slow_governance.checks == ["anthropic"]
    assert slow_governance.agent == ["agent-a"]  # the check ran in the caller's context
    (settle,) = slow_governance.settles
    assert settle["target"] == TARGET
    assert settle["actual_usd"] is None or settle["actual_usd"] >= 0
    _assert_usage(tracer.spans[0], 11, 7)


def test_async_policy_reshape_applies_to_the_request(fake_sdk, tracer, monkeypatch):
    import traccia.governance.pep as pep

    seen = {}

    def enforce_llm_call(kwargs, vendor=None, api=None):
        kwargs["model"] = "claude-cheaper"
        return None

    async def create(self, **kwargs):
        seen.update(kwargs)
        return Message(Usage(1, 1))

    monkeypatch.setattr(pep, "enforce_llm_call", enforce_llm_call)
    monkeypatch.setattr(runtime_config, "pep_enabled", lambda: True)
    monkeypatch.setattr(fake_sdk.AsyncMessages, "create", create)
    patch_anthropic()

    asyncio.run(fake_sdk.AsyncMessages().create(model="claude-x", messages=[]))

    assert seen["model"] == "claude-cheaper"


def test_async_cancelled_call_releases_its_reservation(fake_sdk, tracer, slow_governance, monkeypatch):
    started = threading.Event()

    async def create(self, **kwargs):
        started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr(fake_sdk.AsyncMessages, "create", create)
    patch_anthropic()

    async def main():
        task = asyncio.ensure_future(fake_sdk.AsyncMessages().create(model="claude-x", messages=[]))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    _flush_settles()

    assert slow_governance.settles == [{"release": True, "target": TARGET}]
    (span,) = tracer.spans
    assert isinstance(span.exception, asyncio.CancelledError)


def test_async_call_cancelled_during_its_policy_check_releases_the_reservation(
    fake_sdk, tracer, slow_governance
):
    patch_anthropic()

    async def main():
        task = asyncio.ensure_future(fake_sdk.AsyncMessages().create(model="claude-x", messages=[]))
        await asyncio.sleep(0.05)  # inside the 250 ms check
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    time.sleep(0.4)  # the check finishes on its thread and reserves
    _flush_settles()

    assert slow_governance.checks == ["anthropic"]
    assert slow_governance.settles == [{"release": True, "target": TARGET}]


def test_async_wait_for_timeout_releases_the_reservation(fake_sdk, tracer, slow_governance, monkeypatch):
    async def create(self, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(fake_sdk.AsyncMessages, "create", create)
    patch_anthropic()

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(fake_sdk.AsyncMessages().create(model="claude-x", messages=[]), 0.4)

    asyncio.run(main())
    _flush_settles()

    assert slow_governance.settles == [{"release": True, "target": TARGET}]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
def test_async_call_in_a_forked_child_does_not_deadlock(fake_sdk, tracer, monkeypatch):
    """A child forked while another thread held the policy-check pool's lock can still make an
    async call: the child gets a fresh lock and pool, not the parent's held lock."""
    import traccia.governance.pep as pep

    monkeypatch.setattr(pep, "enforce_llm_call", lambda kwargs, vendor=None, api=None: None)
    monkeypatch.setattr(runtime_config, "pep_enabled", lambda: True)
    patch_anthropic()
    asyncio.run(fake_sdk.AsyncMessages().create(model="claude-x", messages=[]))  # pool started

    held, release = threading.Event(), threading.Event()

    def hold_lock():
        with pep._offloop_lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert held.wait(5)
    try:
        pid = os.fork()
        if pid == 0:  # child: exit 0 if the call returns, killed by SIGALRM if it hangs
            code = 1
            try:
                signal.alarm(5)
                resp = asyncio.run(fake_sdk.AsyncMessages().create(model="claude-x", messages=[]))
                code = 0 if isinstance(resp, Message) else 1
            finally:
                os._exit(code)
    finally:
        release.set()
        holder.join(5)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
