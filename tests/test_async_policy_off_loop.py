"""The async OpenAI Responses and Gemini wrappers run their policy check and settle off the
event loop, and release the reservation of a cancelled call (TRC-281 follow-up).

Groq's async wrapper is covered in test_groq_patch.py, Anthropic's in test_anthropic_patch.py.
"""

import asyncio
import contextlib
import contextvars
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

import instrumentation.gemini as gemini_mod
import instrumentation.openai as openai_mod
import traccia.governance.pep as pep
from traccia import runtime_config

DECISION = {"effect": "allow", "obligations": {"reserved_usd": 0.5}}
TARGET = ("tr_key", "http://traccia.test", "agent-a")
RELEASE = {"release": True, "target": TARGET}

_AGENT = contextvars.ContextVar("test_agent", default=None)


class FakeSpan:
    def __init__(self):
        self.attributes = {}
        self.exception = None
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exception = exc

    def set_status(self, status, message=None):
        self.status = (status, message)


class FakeTracer:
    def __init__(self):
        self.spans = []

    @contextlib.contextmanager
    def start_as_current_span(self, name, attributes=None):
        span = FakeSpan()
        span.attributes.update(attributes or {})
        self.spans.append(span)
        yield span


def _openai_response():
    usage = types.SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18)
    return types.SimpleNamespace(model="gpt-x", usage=usage, output=[], status="completed")


def _gemini_response():
    usage = types.SimpleNamespace(
        total_input_tokens=11,
        total_output_tokens=7,
        total_thought_tokens=0,
        total_cached_tokens=0,
        total_tool_use_tokens=0,
        total_tokens=18,
    )
    return types.SimpleNamespace(
        id="i-1", model="gemini-x", status="completed", output_text="ok", usage=usage, agent=None
    )


def _install_openai(monkeypatch, tracer, create):
    responses_mod = types.ModuleType("openai.resources.responses")
    responses_mod.AsyncResponses = type("AsyncResponses", (), {"create": create})
    for name, module in (
        ("openai", types.ModuleType("openai")),
        ("openai.resources", types.ModuleType("openai.resources")),
        ("openai.resources.responses", responses_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(openai_mod, "_responses_patched", False)
    monkeypatch.setattr(openai_mod, "_get_tracer", lambda name: tracer)
    assert openai_mod.patch_openai_responses()
    return lambda: responses_mod.AsyncResponses().create(model="gpt-x", input="hi")


def _install_gemini(monkeypatch, tracer, create):
    cls = type("AsyncInteractions", (), {"create": gemini_mod._build_async_wrapper(create)})
    monkeypatch.setattr(gemini_mod, "_get_tracer", lambda name: tracer)
    return lambda: cls().create(model="gemini-x", input="hi")


PROVIDERS = {
    "openai": (_install_openai, _openai_response, "openai"),
    "gemini": (_install_gemini, _gemini_response, "google_gemini"),
}


@pytest.fixture(params=sorted(PROVIDERS))
def provider(request, monkeypatch):
    """install(create=None) patches a fake async SDK method and returns a factory for one call
    through the wrapper; create defaults to one that returns a response."""
    install, response, vendor = PROVIDERS[request.param]
    tracer = FakeTracer()

    def install_with(create=None):
        if create is None:

            async def create(self, **kwargs):
                return response()

        return install(monkeypatch, tracer, create)

    return types.SimpleNamespace(install=install_with, tracer=tracer, vendor=vendor)


@pytest.fixture(autouse=True)
def governed(monkeypatch):
    monkeypatch.setattr(runtime_config, "pep_enabled", lambda: True)


@pytest.fixture
def slow_governance(monkeypatch):
    """A policy check and settle that block their thread for 250 ms, like a slow policy server."""
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
    return calls


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


def _settles(calls):
    pep.flush_pending_settles()
    return calls.settles


def test_policy_check_and_settle_do_not_block_the_event_loop(provider, slow_governance):
    call = provider.install()
    _AGENT.set("agent-a")

    lateness = _timer_lateness(call)

    assert lateness < 0.1  # the check and settle each take 250 ms, off the loop
    assert slow_governance.checks == [provider.vendor]
    assert slow_governance.agent == ["agent-a"]  # the check ran in the caller's context
    (settle,) = _settles(slow_governance)
    assert settle["target"] == TARGET
    assert not settle.get("release")
    assert provider.tracer.spans[0].status is None


def test_policy_reshape_applies_to_the_request(provider, monkeypatch):
    seen = {}

    def enforce_llm_call(kwargs, vendor=None, api=None):
        kwargs["model"] = "cheaper-model"
        return None

    async def create(self, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(pep, "enforce_llm_call", enforce_llm_call)
    asyncio.run(provider.install(create)())

    assert seen["model"] == "cheaper-model"


def test_cancelled_call_releases_its_reservation(provider, slow_governance):
    started = threading.Event()

    async def create(self, **kwargs):
        started.set()
        await asyncio.sleep(10)

    call = provider.install(create)

    async def main():
        task = asyncio.ensure_future(call())
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())

    assert _settles(slow_governance) == [RELEASE]
    (span,) = provider.tracer.spans
    assert isinstance(span.exception, asyncio.CancelledError)
    assert span.status[1] == "CancelledError"


def test_call_cancelled_during_its_policy_check_releases_the_reservation(provider, slow_governance):
    created = []

    async def create(self, **kwargs):
        created.append(kwargs)

    call = provider.install(create)

    async def main():
        task = asyncio.ensure_future(call())
        await asyncio.sleep(0.05)  # inside the 250 ms check
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    time.sleep(0.4)  # the check finishes on its thread and reserves

    assert slow_governance.checks == [provider.vendor]
    assert _settles(slow_governance) == [RELEASE]
    assert created == []  # the provider is never called


def test_wait_for_timeout_releases_the_reservation(provider, slow_governance):
    async def create(self, **kwargs):
        await asyncio.sleep(10)

    call = provider.install(create)

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(call(), 0.4)

    asyncio.run(main())

    assert _settles(slow_governance) == [RELEASE]


def test_denied_call_settles_nothing(provider, monkeypatch, slow_governance):
    from traccia.governance.policy import AgentBlockedError

    def enforce_llm_call(kwargs, vendor=None, api=None):
        raise AgentBlockedError("denied")

    monkeypatch.setattr(pep, "enforce_llm_call", enforce_llm_call)
    with pytest.raises(AgentBlockedError):
        asyncio.run(provider.install()())

    assert _settles(slow_governance) == []
    assert isinstance(provider.tracer.spans[0].exception, AgentBlockedError)


def test_ungoverned_call_skips_the_worker_thread(provider, monkeypatch):
    """With policy checks off there is no I/O to move, so the call stays on the loop."""
    monkeypatch.setattr(runtime_config, "pep_enabled", lambda: False)
    monkeypatch.setattr(pep, "run_off_loop", None)  # would fail if called
    asyncio.run(provider.install()())
    assert provider.tracer.spans[0].status is None


# -- pep.run_off_loop() ------------------------------------------------------------------------


def test_run_off_loop_reports_a_call_cancelled_before_it_started(monkeypatch):
    """With every worker busy, a cancelled call's work never runs, and if_cancelled gets None."""
    monkeypatch.setattr(pep, "_offloop_executor", ThreadPoolExecutor(max_workers=1))
    busy = threading.Event()
    release = threading.Event()
    ran, reported = [], []

    async def main():
        blocker = asyncio.ensure_future(pep.run_off_loop(lambda: (busy.set(), release.wait(5))))
        while not busy.is_set():
            await asyncio.sleep(0.01)
        task = asyncio.ensure_future(pep.run_off_loop(ran.append, 1, if_cancelled=reported.append))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await blocker

    asyncio.run(main())
    pep._offloop_executor.shutdown(wait=True)

    assert ran == []
    assert reported == [None]


def test_run_off_loop_reports_a_late_result_in_the_callers_context():
    """Cancelled while the work runs: if_cancelled gets its finished Future, in the context the
    caller had."""
    started = threading.Event()
    reported = []

    def work():
        started.set()
        time.sleep(0.2)
        return "late"

    def if_cancelled(done):
        reported.append((done.result(), _AGENT.get()))

    async def main():
        _AGENT.set("agent-b")
        task = asyncio.ensure_future(pep.run_off_loop(work, if_cancelled=if_cancelled))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    deadline = time.monotonic() + 5
    while not reported and time.monotonic() < deadline:
        time.sleep(0.01)

    assert reported == [("late", "agent-b")]
