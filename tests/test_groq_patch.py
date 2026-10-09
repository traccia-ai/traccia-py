"""
Tests for instrumentation/groq.py's patch_groq() and span population.

The Groq SDK is faked at sys.modules level (groq.resources.chat.completions
.Completions / .AsyncCompletions) and the tracer is replaced with FakeTracer, so
no network or real exporter is needed. test_real_sdk_gets_patched runs against
the installed groq package when it is available.
"""

import asyncio
import gc
import sys
import types

import pytest
from opentelemetry.trace import INVALID_SPAN_CONTEXT, NonRecordingSpan

import instrumentation.groq as groq_mod
from traccia.tracer.span import SpanStatus
from instrumentation.groq import patch_groq


# Fakes: a tracer/span pair that records attributes and how often end() was called.
# FakeSpan is an OTel span so opentelemetry.trace.get_current_span() returns it
# while it is pushed as the current span.
class FakeSpan(NonRecordingSpan):
    def __init__(self):
        super().__init__(INVALID_SPAN_CONTEXT)
        self.attributes, self.exception, self.status = {}, None, None
        self.end_count = 0

    def set_attribute(self, k, v):
        self.attributes[k] = v

    def record_exception(self, exc):
        self.exception = exc

    def set_status(self, status, message=None):
        self.status = (status, message)

    def end(self):
        self.end_count += 1

    # Lets the span be pushed as the current OTel span.
    def get_span_context(self):
        return INVALID_SPAN_CONTEXT

    @property
    def ended(self):
        return self.end_count > 0


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes=None):
        span = FakeSpan()
        span.attributes.update(attributes or {})
        span.name = name
        self.spans.append(span)
        return span


@pytest.fixture(autouse=True)
def reset_state():
    groq_mod._patched = False
    yield
    groq_mod._patched = False


@pytest.fixture
def fake_tracer(monkeypatch):
    t = FakeTracer()
    monkeypatch.setattr(groq_mod, "_get_tracer", lambda name: t)
    return t


def _fake_response(**kw):
    d = dict(
        model="llama-3.3-70b-versatile",
        usage=types.SimpleNamespace(
            prompt_tokens=11, completion_tokens=5, total_tokens=16
        ),
        choices=[
            types.SimpleNamespace(
                finish_reason="stop", message=types.SimpleNamespace(content="Paris.")
            )
        ],
    )
    d.update(kw)
    return types.SimpleNamespace(**d)


def _make_classes():
    class Completions:
        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return _fake_response()

    class AsyncCompletions:
        async def create(self, **kwargs):
            self.last_kwargs = kwargs
            return _fake_response()

    return Completions, AsyncCompletions


def _install_sdk(monkeypatch, sync_cls, async_cls):
    for name in ("groq", "groq.resources", "groq.resources.chat"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("groq.resources.chat.completions")
    mod.Completions, mod.AsyncCompletions = sync_cls, async_cls
    monkeypatch.setitem(sys.modules, "groq.resources.chat.completions", mod)


MSGS = [{"role": "user", "content": "Capital of France?"}]


# Patching and span population
def test_patch_marks_both_classes(monkeypatch):
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    assert patch_groq() is True
    assert S.create._agent_trace_patched and A.create._agent_trace_patched


def test_patch_returns_false_when_groq_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", None)
    monkeypatch.setitem(sys.modules, "groq.resources.chat.completions", None)
    assert patch_groq() is False


def test_double_patch_does_not_double_wrap(monkeypatch):
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    first = S.create
    assert first._agent_trace_patched
    groq_mod._patched = False
    patch_groq()
    assert S.create is first


def test_sync_call_populates_span(monkeypatch, fake_tracer):
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    resp = S().create(model="llama-3.3-70b-versatile", messages=MSGS)
    assert resp.choices[0].message.content == "Paris."
    a = fake_tracer.spans[-1].attributes
    assert fake_tracer.spans[-1].name == "llm.groq.chat.completions"
    assert a["llm.vendor"] == "groq"
    assert a["llm.model"] == "llama-3.3-70b-versatile"
    assert a["llm.prompt"] == "user: Capital of France?"
    assert a["llm.usage.prompt_tokens"] == 11
    assert a["llm.usage.completion_tokens"] == 5
    assert a["llm.usage.total_tokens"] == 16
    assert a["llm.usage.source"] == "provider_usage"
    assert a["llm.finish_reason"] == "stop"
    assert a["llm.completion"] == "Paris."


def test_async_call_populates_span(monkeypatch, fake_tracer):
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    asyncio.run(A().create(model="llama-3.3-70b-versatile", messages=MSGS))
    a = fake_tracer.spans[-1].attributes
    assert a["llm.usage.total_tokens"] == 16
    assert a["llm.finish_reason"] == "stop"
    assert a["llm.completion"] == "Paris."


def test_positional_args_pass_through(monkeypatch, fake_tracer):
    S, A = _make_classes()
    S.create = lambda self, *args, **kw: (args, kw)
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    args, kw = S().create("pos", model="m", messages=MSGS)
    assert args == ("pos",) and kw["model"] == "m"


def test_exception_is_recorded_and_reraised(monkeypatch, fake_tracer):
    S, A = _make_classes()

    def boom(self, **kw):
        raise RuntimeError("rate limited")

    S.create = boom
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    with pytest.raises(RuntimeError):
        S().create(model="m", messages=MSGS)
    span = fake_tracer.spans[-1]
    assert isinstance(span.exception, RuntimeError)
    assert span.status == (SpanStatus.ERROR, "rate limited")


def test_async_exception_is_recorded_and_reraised(monkeypatch, fake_tracer):
    S, A = _make_classes()

    async def boom(self, **kw):
        raise RuntimeError("rate limited")

    A.create = boom
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    with pytest.raises(RuntimeError):
        asyncio.run(A().create(model="m", messages=MSGS))
    span = fake_tracer.spans[-1]
    assert isinstance(span.exception, RuntimeError)
    assert span.status[0] == SpanStatus.ERROR


def test_metrics_recorded(monkeypatch, fake_tracer):
    import traccia.metrics.recorder as rec_mod

    calls = {}

    class Rec:
        def record_token_usage(self, **kw):
            calls["tokens"] = kw

        def record_duration(self, d, **kw):
            calls["duration"] = d

        def record_cost(self, c, **kw):
            calls["cost"] = (c, kw)

        def record_exception(self, **kw):
            calls["exception"] = kw

    monkeypatch.setattr(rec_mod, "get_metrics_recorder", lambda: Rec())
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    S().create(model="llama-3.3-70b-versatile", messages=MSGS)
    assert calls["tokens"]["prompt_tokens"] == 11
    assert calls["tokens"]["completion_tokens"] == 5
    assert calls["tokens"]["attributes"]["gen_ai.system"] == "groq"
    assert calls["duration"] >= 0
    assert calls["cost"][0] > 0


def test_real_sdk_gets_patched(monkeypatch):
    pytest.importorskip("groq")
    from groq.resources.chat.completions import AsyncCompletions, Completions

    monkeypatch.setattr(Completions, "create", Completions.create)
    monkeypatch.setattr(AsyncCompletions, "create", AsyncCompletions.create)
    assert patch_groq() is True
    assert Completions.create._agent_trace_patched
    assert AsyncCompletions.create._agent_trace_patched


# Governance
@pytest.fixture
def pep(monkeypatch):
    """Enable PEP and capture settle calls."""
    from traccia import runtime_config
    import traccia.governance.pep as pep_mod

    calls = {"finish": []}
    monkeypatch.setattr(
        pep_mod,
        "finish_llm_call",
        lambda decision, **kw: calls["finish"].append((decision, kw)),
    )
    with runtime_config.run_identity(pep_enabled=True):
        yield pep_mod, calls


def test_deny_blocks_before_provider_call(monkeypatch, fake_tracer, pep):
    from traccia.governance.policy import AgentBlockedError

    pep_mod, _ = pep

    def deny(**kw):
        raise AgentBlockedError("budget exceeded")

    monkeypatch.setattr(pep_mod, "check_policy", deny)
    S, A = _make_classes()
    called = []
    orig = S.create
    S.create = lambda self, **kw: called.append(1) or orig(self, **kw)
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    with pytest.raises(AgentBlockedError):
        S().create(model="llama-3.3-70b-versatile", messages=MSGS)
    assert called == []


def test_reshape_swaps_model(monkeypatch, fake_tracer, pep):
    pep_mod, _ = pep
    monkeypatch.setattr(
        pep_mod,
        "check_policy",
        lambda **kw: {
            "effect": "reshape",
            "would_have": False,
            "obligations": {
                "cheaper_model": "llama-3.1-8b-instant",
                "clamp_max_tokens": 100,
            },
        },
    )
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    inst = S()
    inst.create(model="llama-3.3-70b-versatile", messages=MSGS)
    assert inst.last_kwargs["model"] == "llama-3.1-8b-instant"
    assert inst.last_kwargs["max_tokens"] == 100
    assert fake_tracer.spans[-1].attributes["llm.model"] == "llama-3.1-8b-instant"


def test_async_deny_blocks_before_provider_call(monkeypatch, fake_tracer, pep):
    from traccia.governance.policy import AgentBlockedError

    pep_mod, _ = pep

    def deny(**kw):
        raise AgentBlockedError("budget exceeded")

    monkeypatch.setattr(pep_mod, "check_policy", deny)
    S, A = _make_classes()
    called = []

    async def create(self, **kw):
        called.append(1)

    A.create = create
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    with pytest.raises(AgentBlockedError):
        asyncio.run(A().create(model="llama-3.3-70b-versatile", messages=MSGS))
    assert called == []


def test_async_reshape_swaps_model(monkeypatch, fake_tracer, pep):
    pep_mod, _ = pep
    monkeypatch.setattr(
        pep_mod,
        "check_policy",
        lambda **kw: {
            "effect": "reshape",
            "would_have": False,
            "obligations": {"cheaper_model": "llama-3.1-8b-instant"},
        },
    )
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    inst = A()
    asyncio.run(inst.create(model="llama-3.3-70b-versatile", messages=MSGS))
    assert inst.last_kwargs["model"] == "llama-3.1-8b-instant"


def test_failure_releases_reservation(monkeypatch, fake_tracer, pep):
    pep_mod, calls = pep
    monkeypatch.setattr(
        pep_mod, "check_policy", lambda **kw: {"effect": "allow", "id": "d1"}
    )
    S, A = _make_classes()

    def boom(self, **kw):
        raise RuntimeError("x")

    S.create = boom
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    with pytest.raises(RuntimeError):
        S().create(model="m", messages=MSGS)
    assert calls["finish"][-1][1].get("release") is True


def test_success_settles_with_actual_cost(monkeypatch, fake_tracer, pep):
    pep_mod, calls = pep
    monkeypatch.setattr(
        pep_mod, "check_policy", lambda **kw: {"effect": "allow", "id": "d1"}
    )
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    S().create(model="llama-3.3-70b-versatile", messages=MSGS)
    assert calls["finish"][-1][1]["actual_usd"] > 0


# Streaming
def _chunk(content=None, finish_reason=None, usage=None, x_groq_usage=None):
    return types.SimpleNamespace(
        model="llama-3.3-70b-versatile",
        choices=[
            types.SimpleNamespace(
                delta=types.SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        x_groq=types.SimpleNamespace(usage=x_groq_usage) if x_groq_usage else None,
    )


USAGE = types.SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16)
CHUNKS = [
    _chunk("Par"),
    _chunk("is."),
    _chunk(finish_reason="stop", x_groq_usage=USAGE),
]


class FakeStream:
    def __init__(self, chunks, error=None):
        self._chunks, self._error = list(chunks), error
        self.closed = False
        self.response = "raw-http-response"

    def __iter__(self):
        return self

    def __next__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error:
            raise self._error
        raise StopIteration

    def close(self):
        self.closed = True


class FakeAsyncStream:
    def __init__(self, chunks, error=None):
        self._chunks, self._error = list(chunks), error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error:
            raise self._error
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


def _install_streaming(monkeypatch, chunks=CHUNKS, error=None):
    S, A = _make_classes()
    S.create = lambda self, **kw: (
        FakeStream(chunks, error) if kw.get("stream") else _fake_response()
    )

    async def acreate(self, **kw):
        return FakeAsyncStream(chunks, error) if kw.get("stream") else _fake_response()

    A.create = acreate
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    return S, A


def _assert_stream_span(span):
    a = span.attributes
    assert span.end_count == 1
    assert a["llm.streaming"] is True
    assert a["llm.completion"] == "Paris."
    assert a["llm.finish_reason"] == "stop"
    assert a["llm.usage.prompt_tokens"] == 11
    assert a["llm.usage.completion_tokens"] == 5
    assert a["llm.usage.total_tokens"] == 16
    assert a["llm.usage.source"] == "provider_usage"


def test_non_streaming_span_ends_once(monkeypatch, fake_tracer):
    S, A = _make_classes()
    _install_sdk(monkeypatch, S, A)
    patch_groq()
    S().create(model="llama-3.3-70b-versatile", messages=MSGS)
    assert fake_tracer.spans[-1].end_count == 1


def test_sync_stream_records_span_when_exhausted(monkeypatch, fake_tracer):
    S, _ = _install_streaming(monkeypatch)
    stream = S().create(model="llama-3.3-70b-versatile", messages=MSGS, stream=True)
    span = fake_tracer.spans[-1]
    assert not span.ended  # still open until the caller consumes the stream
    text = "".join(c.choices[0].delta.content or "" for c in stream)
    assert text == "Paris."
    _assert_stream_span(span)


def test_async_stream_records_span_when_exhausted(monkeypatch, fake_tracer):
    _, A = _install_streaming(monkeypatch)

    async def run():
        stream = await A().create(
            model="llama-3.3-70b-versatile", messages=MSGS, stream=True
        )
        assert not fake_tracer.spans[-1].ended
        return "".join([c.choices[0].delta.content or "" async for c in stream])

    assert asyncio.run(run()) == "Paris."
    _assert_stream_span(fake_tracer.spans[-1])


def test_stream_usage_from_include_usage_chunk(monkeypatch, fake_tracer):
    chunks = [_chunk("Paris.", finish_reason="stop"), _chunk(usage=USAGE)]
    chunks[1].choices = []  # include_usage sends a final chunk with no choices
    S, _ = _install_streaming(monkeypatch, chunks)
    list(S().create(model="llama-3.3-70b-versatile", messages=MSGS, stream=True))
    _assert_stream_span(fake_tracer.spans[-1])


def test_stream_context_manager_closes_and_ends_span(monkeypatch, fake_tracer):
    S, _ = _install_streaming(monkeypatch)
    with S().create(
        model="llama-3.3-70b-versatile", messages=MSGS, stream=True
    ) as stream:
        next(stream)
        inner = stream._stream
    span = fake_tracer.spans[-1]
    assert inner.closed
    assert span.end_count == 1
    assert span.attributes["llm.completion"] == "Par"


def test_async_stream_context_manager_closes_and_ends_span(monkeypatch, fake_tracer):
    _, A = _install_streaming(monkeypatch)

    async def run():
        async with await A().create(model="m", messages=MSGS, stream=True) as stream:
            await stream.__anext__()
            return stream._stream

    inner = asyncio.run(run())
    assert inner.closed
    assert fake_tracer.spans[-1].end_count == 1


def test_abandoned_stream_ends_span_on_gc(monkeypatch, fake_tracer):
    S, _ = _install_streaming(monkeypatch)
    stream = S().create(model="llama-3.3-70b-versatile", messages=MSGS, stream=True)
    for _ in stream:
        break
    del stream
    gc.collect()
    assert fake_tracer.spans[-1].end_count == 1


def test_stream_error_mid_iteration(monkeypatch, fake_tracer):
    S, _ = _install_streaming(monkeypatch, CHUNKS[:1], RuntimeError("connection reset"))
    stream = S().create(model="m", messages=MSGS, stream=True)
    with pytest.raises(RuntimeError):
        list(stream)
    span = fake_tracer.spans[-1]
    assert span.status == (SpanStatus.ERROR, "connection reset")
    assert span.end_count == 1


def test_stream_proxy_forwards_attributes(monkeypatch, fake_tracer):
    S, _ = _install_streaming(monkeypatch)
    stream = S().create(model="m", messages=MSGS, stream=True)
    assert stream.response == "raw-http-response"
    stream.close()


def test_stream_settles_after_consumption(monkeypatch, fake_tracer, pep):
    pep_mod, calls = pep
    monkeypatch.setattr(
        pep_mod, "check_policy", lambda **kw: {"effect": "allow", "id": "d1"}
    )
    S, _ = _install_streaming(monkeypatch)
    stream = S().create(model="llama-3.3-70b-versatile", messages=MSGS, stream=True)
    assert calls["finish"] == []  # nothing settled until the stream is read
    list(stream)
    assert len(calls["finish"]) == 1
    assert calls["finish"][0][1]["actual_usd"] > 0


def test_stream_error_releases_reservation(monkeypatch, fake_tracer, pep):
    pep_mod, calls = pep
    monkeypatch.setattr(
        pep_mod, "check_policy", lambda **kw: {"effect": "allow", "id": "d1"}
    )
    S, _ = _install_streaming(monkeypatch, CHUNKS[:1], RuntimeError("x"))
    with pytest.raises(RuntimeError):
        list(S().create(model="m", messages=MSGS, stream=True))
    assert calls["finish"] == [({"effect": "allow", "id": "d1"}, {"release": True})]


# Streaming: settlement identity and cancellation
@pytest.fixture
def settle_spy(monkeypatch):
    """Allow every call and record the agent, PEP flag and current span each settle sees."""
    from opentelemetry import trace as otel_trace

    from traccia import runtime_config
    import traccia.governance.pep as pep_mod

    seen = []

    def spy(decision, **kw):
        seen.append(
            {
                "agent": runtime_config.get_agent_id(),
                "pep": runtime_config.pep_enabled(),
                "span": otel_trace.get_current_span(),
                "kw": kw,
            }
        )

    monkeypatch.setattr(pep_mod, "finish_llm_call", spy)
    monkeypatch.setattr(
        pep_mod, "check_policy", lambda **kw: {"effect": "allow", "id": "d1"}
    )
    return seen


def test_stream_settles_for_creating_agent(monkeypatch, fake_tracer, settle_spy):
    from traccia import runtime_config
    import traccia.metrics.recorder as rec_mod

    metrics = {}

    class Rec:
        def record_token_usage(self, **kw):
            metrics["agent"] = kw["attributes"].get("agent.id")

        def record_duration(self, d, **kw):
            pass

        def record_cost(self, c, **kw):
            pass

    monkeypatch.setattr(rec_mod, "get_metrics_recorder", lambda: Rec())
    S, _ = _install_streaming(monkeypatch)
    with runtime_config.run_identity(agent_id="agent-a", pep_enabled=True):
        stream = S().create(model="llama-3.3-70b-versatile", messages=MSGS, stream=True)
    with runtime_config.run_identity(agent_id="agent-b", pep_enabled=False):
        list(stream)

    assert len(settle_spy) == 1
    assert settle_spy[0]["agent"] == "agent-a"
    assert settle_spy[0]["pep"] is True
    assert settle_spy[0]["span"] is fake_tracer.spans[-1]
    assert metrics["agent"] == "agent-a"


def test_stream_consumed_after_context_exit_still_settles(
    monkeypatch, fake_tracer, settle_spy
):
    from traccia import runtime_config

    S, _ = _install_streaming(monkeypatch)
    with runtime_config.run_identity(agent_id="agent-a", pep_enabled=True):
        stream = S().create(model="llama-3.3-70b-versatile", messages=MSGS, stream=True)
    list(stream)  # no run identity is active here
    assert [s["agent"] for s in settle_spy] == ["agent-a"]
    assert settle_spy[0]["kw"]["actual_usd"] > 0


def test_async_stream_consumed_in_other_task_settles_for_creating_agent(
    monkeypatch, fake_tracer, settle_spy
):
    from traccia import runtime_config

    _, A = _install_streaming(monkeypatch)

    async def run():
        with runtime_config.run_identity(agent_id="agent-a", pep_enabled=True):
            stream = await A().create(model="m", messages=MSGS, stream=True)

        async def consume():
            with runtime_config.run_identity(agent_id="agent-b"):
                return [c async for c in stream]

        await asyncio.create_task(consume())

    asyncio.run(run())
    assert [s["agent"] for s in settle_spy] == ["agent-a"]
    assert settle_spy[0]["span"] is fake_tracer.spans[-1]


def test_async_cancel_during_create_ends_span_and_releases(
    monkeypatch, fake_tracer, pep
):
    pep_mod, calls = pep
    monkeypatch.setattr(
        pep_mod, "check_policy", lambda **kw: {"effect": "allow", "id": "d1"}
    )
    S, A = _make_classes()

    async def slow(self, **kw):
        await asyncio.sleep(10)

    A.create = slow
    _install_sdk(monkeypatch, S, A)
    patch_groq()

    async def run():
        task = asyncio.create_task(A().create(model="m", messages=MSGS))
        await asyncio.sleep(0)  # let the call start and block on the provider
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    span = fake_tracer.spans[-1]
    assert span.end_count == 1
    assert span.status == (SpanStatus.ERROR, "CancelledError")
    assert calls["finish"] == [({"effect": "allow", "id": "d1"}, {"release": True})]


def test_async_cancel_during_stream_iteration_ends_span_and_releases(
    monkeypatch, fake_tracer, pep
):
    pep_mod, calls = pep
    monkeypatch.setattr(
        pep_mod, "check_policy", lambda **kw: {"effect": "allow", "id": "d1"}
    )

    class HangingAsyncStream(FakeAsyncStream):
        async def __anext__(self):
            if self._chunks:
                return self._chunks.pop(0)
            await asyncio.sleep(10)  # the next chunk never arrives

    S, A = _make_classes()

    async def create(self, **kw):
        return HangingAsyncStream(CHUNKS[:1])

    A.create = create
    _install_sdk(monkeypatch, S, A)
    patch_groq()

    async def run():
        stream = await A().create(model="m", messages=MSGS, stream=True)

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    span = fake_tracer.spans[-1]
    assert span.end_count == 1
    assert span.status[0] == SpanStatus.ERROR
    assert calls["finish"] == [({"effect": "allow", "id": "d1"}, {"release": True})]


def test_keyboard_interrupt_mid_stream_ends_span(monkeypatch, fake_tracer):
    S, _ = _install_streaming(monkeypatch, CHUNKS[:1], KeyboardInterrupt())
    stream = S().create(model="m", messages=MSGS, stream=True)
    with pytest.raises(KeyboardInterrupt):
        list(stream)
    span = fake_tracer.spans[-1]
    assert span.end_count == 1
    assert span.status == (SpanStatus.ERROR, "KeyboardInterrupt")
