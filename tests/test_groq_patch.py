"""
Tests for instrumentation/groq.py, against the real groq SDK.

Only the HTTP layers are faked: Groq's API through an httpx.MockTransport, and the policy
server through traccia.governance.pep._http_session. Spans come from a recording tracer.
"""

import asyncio
import gc
import json
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

groq = pytest.importorskip("groq")
from groq import AsyncGroq, Groq  # noqa: E402
from groq.resources.chat.completions import AsyncCompletions, Completions  # noqa: E402
from opentelemetry.trace import INVALID_SPAN_CONTEXT  # noqa: E402

import traccia.instrumentation.groq as groq_mod  # noqa: E402
from traccia import runtime_config  # noqa: E402
from traccia.governance import pep  # noqa: E402
from traccia.governance.pep import estimate_prompt_tokens, flush_pending_settles, settle_open_calls  # noqa: E402
from traccia.governance.policy import AgentBlockedError  # noqa: E402
from traccia.instrumentation.groq import patch_groq  # noqa: E402
from traccia.tracer.span import SpanStatus  # noqa: E402

MODEL = "llama-3.3-70b-versatile"
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital of France?"}]
# The prompt estimate sent with the policy check, used when Groq reports no usage.
PROMPT_ESTIMATE = estimate_prompt_tokens({"messages": MESSAGES})
USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
RESERVED = 0.5
# Groq's bundled price for MODEL, USD per 1K input and output tokens.
GROQ_RATES = (0.00059, 0.00079)
COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": MODEL,
    "usage": USAGE,
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Paris."}}],
}


def chunk(delta, finish_reason=None, **extra):
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }


# Groq's stream shape: a role-only first chunk, and usage under x_groq on the last one.
CHUNKS = [
    chunk({"role": "assistant", "content": ""}, x_groq={"id": "req_1"}),
    chunk({"content": "Par"}),
    chunk({"content": "is."}),
    chunk({}, "stop", x_groq={"id": "req_1", "usage": USAGE}),
]


def sse(chunks, error=None):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    if error:
        body += f"data: {json.dumps({'error': {'message': error}})}\n\n"
    return body + "data: [DONE]\n\n"


# -- fakes -------------------------------------------------------------------------


class FakeSpan:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = dict(attributes or {})
        self.ended = 0
        self.status = None
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status, description=None):
        self.status = (status, description)

    def end(self):
        self.ended += 1

    # Lets OpenTelemetry's context treat it as a (non-recording) current span.
    def get_span_context(self):
        return INVALID_SPAN_CONTEXT


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes=None):
        span = FakeSpan(name, attributes)
        self.spans.append(span)
        return span


class PolicyServer:
    """Fake policy server: records checks and settles."""

    def __init__(self):
        self.decision = {"effect": "allow", "would_have": False, "obligations": {"reserved_usd": RESERVED}}
        self.checks = []
        self._settles = []
        self.session = MagicMock()
        self.session.post.side_effect = self._post

    @property
    def settles(self):
        """Settles received, after waiting for those still running in the background."""
        flush_pending_settles()
        return self._settles

    def _post(self, url, **kwargs):
        body = kwargs.get("json")
        response = MagicMock(status_code=200)
        if url.endswith("/check"):
            self.checks.append(body)
            response.json.return_value = self.decision
        else:
            self._settles.append(body)
            response.json.return_value = {}
        return response


class GroqApi:
    """Fake Groq HTTP API for httpx.MockTransport."""

    def __init__(self):
        self.status = 200
        self.completion = COMPLETION
        self.chunks = CHUNKS
        self.stream_error = None
        self.requests = []
        # Set to an Event to hold each request until it is set (a call still in flight).
        self.hold = None
        self.entered = threading.Event()

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        self.entered.set()
        if self.hold is not None:
            self.hold.wait(timeout=10)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "bad request"}})
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse(self.chunks, self.stream_error),
            )
        return httpx.Response(200, json=self.completion)


# -- fixtures ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_patch():
    """Patch the real groq classes for each test, and restore them afterwards."""
    originals = (Completions.create, AsyncCompletions.create)
    groq_mod._patched = False
    yield
    Completions.create, AsyncCompletions.create = originals
    groq_mod._patched = False


@pytest.fixture
def tracer(monkeypatch):
    tracer = FakeTracer()
    monkeypatch.setattr(groq_mod, "_get_tracer", lambda name: tracer)
    return tracer


@pytest.fixture
def server():
    server = PolicyServer()
    with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
        with patch("traccia.governance.pep._http_session", server.session):
            yield server


@pytest.fixture
def api():
    return GroqApi()


@pytest.fixture
def client(api, tracer, server):
    assert patch_groq()
    return Groq(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(api)), max_retries=0)


@pytest.fixture
def async_client(api, tracer, server):
    assert patch_groq()
    return AsyncGroq(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)), max_retries=0
    )


@pytest.fixture
def governed():
    with runtime_config.run_identity(agent_id="agent-a", pep_enabled=True):
        yield


def priced(cost=pytest.approx(0.0000095, rel=0.1)):
    return {
        "actual_usd": cost,
        "release": False,
        "model": MODEL,
        "vendor": "groq",
        "input_tokens": 12,
        "output_tokens": 3,
    }


def estimated(output_tokens, input_tokens=PROMPT_ESTIMATE, rates=GROQ_RATES):
    """An estimate priced locally, at *rates*, like provider-reported usage (to 6 decimals)."""
    return {
        "actual_usd": pytest.approx(round((input_tokens * rates[0] + output_tokens * rates[1]) / 1000, 6)),
        "release": False,
        "model": MODEL,
        "vendor": "groq",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def settle_fields(body):
    keys = ("actual_usd", "release", "model", "vendor", "input_tokens", "output_tokens")
    return {k: body.get(k) for k in keys}


def create(client, **extra):
    return client.chat.completions.create(model=MODEL, messages=MESSAGES, **extra)


# -- patching ----------------------------------------------------------------------


def test_patch_groq_wraps_sync_and_async_create_once():
    assert patch_groq() is True
    sync_create, async_create = Completions.create, AsyncCompletions.create
    assert sync_create._agent_trace_patched and async_create._agent_trace_patched
    assert asyncio.iscoroutinefunction(async_create)
    groq_mod._patched = False
    assert patch_groq() is True
    assert Completions.create is sync_create
    assert AsyncCompletions.create is async_create


def test_patch_groq_soft_fails_without_groq(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "groq.resources.chat.completions", None)
    assert patch_groq() is False


def test_auto_init_patches_groq():
    import traccia.auto as auto

    assert auto.patch_groq is patch_groq


# -- non-streaming -----------------------------------------------------------------


def test_completion_is_traced_checked_and_settled_with_its_cost(client, tracer, server, governed):
    response = create(client)

    assert response.choices[0].message.content == "Paris."
    [span] = tracer.spans
    assert span.name == "llm.groq.chat.completions"
    assert span.ended == 1
    a = span.attributes
    assert a["llm.vendor"] == "groq"
    assert a["llm.model"] == MODEL
    assert a["llm.prompt"] == "system: Be brief.\nuser: Capital of France?"
    assert json.loads(a["llm.groq.messages"]) == MESSAGES
    assert a["llm.usage.prompt_tokens"] == 12
    assert a["llm.usage.completion_tokens"] == 3
    assert a["llm.usage.total_tokens"] == 15
    assert a["llm.completion"] == "Paris."
    assert a["llm.finish_reason"] == "stop"
    assert "llm.streaming" not in a
    [check] = server.checks
    assert check["context"]["vendor"] == "groq"
    assert check["context"]["model"] == MODEL
    [settle] = server.settles
    assert settle["agent_id"] == "agent-a"
    assert settle["reserved_usd"] == RESERVED
    assert settle_fields(settle) == priced()


def test_unpriced_model_sends_tokens_for_the_server_to_price(client, server, governed, monkeypatch):
    monkeypatch.setattr(groq_mod, "_compute_cost", lambda *a: None)
    create(client)
    assert settle_fields(server.settles[0]) == priced(cost=None)


def test_reshaped_model_is_sent_recorded_and_settled(client, api, tracer, server, governed):
    server.decision = {
        "effect": "reshape",
        "would_have": False,
        "obligations": {"reserved_usd": RESERVED, "cheaper_model": "llama-3.1-8b-instant"},
    }
    api.completion = {**COMPLETION, "model": None}
    create(client)
    assert api.requests[0]["model"] == "llama-3.1-8b-instant"
    assert tracer.spans[0].attributes["llm.model"] == "llama-3.1-8b-instant"
    assert server.settles[0]["model"] == "llama-3.1-8b-instant"


def test_response_without_usage_settles_an_estimate(client, api, server, governed):
    api.completion = {**COMPLETION, "usage": None}
    create(client)
    # 'Paris.' is 6 characters: 2 output tokens.
    assert settle_fields(server.settles[0]) == estimated(2)


def test_estimate_always_carries_input_tokens(client, api, server, governed):
    # Without input_tokens the server would release the reservation instead of pricing it.
    api.completion = {**COMPLETION, "usage": None}
    client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": ""}])
    assert settle_fields(server.settles[0]) == estimated(2, input_tokens=0)


def test_tool_call_arguments_count_as_output(client, api, server, governed):
    arguments = '{"city":"Paris","unit":"celsius"}'  # 33 characters: 9 tokens
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "weather", "arguments": arguments}}],
    }
    api.completion = {**COMPLETION, "usage": None, "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]}
    create(client)
    assert settle_fields(server.settles[0]) == estimated(9)


def test_http_error_releases_records_and_propagates(client, api, tracer, server, governed):
    api.status = 400
    with pytest.raises(groq.BadRequestError):
        create(client)
    [span] = tracer.spans
    assert span.ended == 1
    assert span.status[0] == SpanStatus.ERROR
    assert isinstance(span.exceptions[0], groq.BadRequestError)
    assert settle_fields(server.settles[0])["release"] is True


def test_denied_call_never_reaches_groq(client, api, tracer, server, governed):
    server.decision = {"id": "d1", "effect": "deny", "would_have": False, "reasons": ["spend cap"], "obligations": {}}
    with pytest.raises(AgentBlockedError):
        create(client)
    assert api.requests == []
    assert tracer.spans[0].ended == 1
    assert tracer.spans[0].status[0] == SpanStatus.ERROR
    assert server.settles == []  # nothing was reserved


def test_ungoverned_call_is_traced_without_policy_calls(client, tracer, server):
    create(client)
    assert tracer.spans[0].ended == 1
    assert server.checks == [] and server.settles == []


def test_with_raw_response_keeps_parse_and_settles_once(client, tracer, server, governed):
    raw = client.chat.completions.with_raw_response.create(model=MODEL, messages=MESSAGES)
    parsed = raw.parse()
    assert parsed.choices[0].message.content == "Paris."
    assert raw.parse() is parsed
    assert tracer.spans[0].ended == 1
    assert [settle_fields(s) for s in server.settles] == [priced()]


def test_with_streaming_response_charges_the_reservation_and_leaves_the_body(client, server, governed):
    with client.chat.completions.with_streaming_response.create(model=MODEL, messages=MESSAGES) as response:
        body = b"".join(response.iter_bytes())
    assert json.loads(body)["choices"][0]["message"]["content"] == "Paris."
    assert [settle_fields(s)["actual_usd"] for s in server.settles] == [RESERVED]


def test_metrics_are_recorded_for_the_calling_agent(client, governed, monkeypatch):
    recorder = MagicMock()
    monkeypatch.setattr("traccia.metrics.recorder.get_metrics_recorder", lambda: recorder)
    create(client)
    attrs = recorder.record_token_usage.call_args.kwargs["attributes"]
    assert attrs["gen_ai.system"] == "groq"
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["agent.id"] == "agent-a"
    assert recorder.record_token_usage.call_args.kwargs["prompt_tokens"] == 12
    recorder.record_duration.assert_called_once()
    recorder.record_cost.assert_called_once()
    recorder.record_exception.assert_not_called()


def test_a_failing_metrics_recorder_does_not_affect_the_call(client, server, governed, monkeypatch):
    def broken():
        raise RuntimeError("meter gone")

    monkeypatch.setattr("traccia.metrics.recorder.get_metrics_recorder", broken)
    assert create(client).choices[0].message.content == "Paris."
    assert settle_fields(server.settles[0]) == priced()


# -- streaming ---------------------------------------------------------------------


def test_stream_read_to_the_end_is_priced_once(client, tracer, server, governed):
    stream = create(client, stream=True)
    assert tracer.spans[0].ended == 0
    assert server.settles == []
    text = "".join(c.choices[0].delta.content or "" for c in stream)

    assert text == "Paris."
    [span] = tracer.spans
    assert span.ended == 1
    assert span.attributes["llm.streaming"] is True
    assert span.attributes["llm.completion"] == "Paris."
    assert span.attributes["llm.finish_reason"] == "stop"
    assert span.attributes["llm.usage.total_tokens"] == 15
    assert span.status is None
    assert [settle_fields(s) for s in server.settles] == [priced()]


def test_stream_with_include_usage_is_priced(client, api, server, governed):
    api.chunks = [chunk({"content": "Paris."}, "stop"), {**chunk({}), "choices": [], "usage": USAGE}]
    # groq's Python SDK has no stream_options parameter; it is passed through extra_body.
    for _ in create(client, stream=True, extra_body={"stream_options": {"include_usage": True}}):
        pass
    assert [settle_fields(s) for s in server.settles] == [priced()]


def test_stream_next_is_traced(client, server, governed):
    stream = create(client, stream=True)
    while True:
        try:
            next(stream)
        except StopIteration:
            break
    assert [settle_fields(s) for s in server.settles] == [priced()]


def test_stream_closed_early_settles_what_was_streamed(client, tracer, server, governed):
    stream = create(client, stream=True)
    for c in stream:
        if c.choices[0].delta.content:
            break
    stream.close()
    assert tracer.spans[0].ended == 1
    assert tracer.spans[0].attributes["llm.completion"] == "Par"
    # 'Par' is 1 token; Groq bills what it generated.
    assert [settle_fields(s) for s in server.settles] == [estimated(1)]


def test_stream_context_manager_exit_finalizes(client, server, governed):
    with create(client, stream=True) as stream:
        next(iter(stream))
    assert len(server.settles) == 1
    assert server.settles[0]["release"] is False


def test_stream_dropped_unread_charges_the_reservation(client, tracer, server, governed):
    stream = create(client, stream=True)
    del stream
    gc.collect()
    assert tracer.spans[0].ended == 1
    assert [settle_fields(s)["actual_usd"] for s in server.settles] == [RESERVED]


def test_stream_dropped_mid_read_aborts_the_request_and_charges_the_reservation(client, tracer, server, governed):
    stream = create(client, stream=True)
    response = stream.response
    it = iter(stream)
    next(it)
    next(it)  # 'Par'
    del it, stream
    gc.collect()
    # Groq bills the whole completion unless the request stops; without usage the cost is
    # unknown, so the reservation is kept.
    assert response.is_closed
    assert [settle_fields(s)["actual_usd"] for s in server.settles] == [RESERVED]
    assert tracer.spans[0].attributes["llm.interrupted"] == "abandoned"
    assert tracer.spans[0].ended == 1


def test_interrupted_call_is_priced_with_the_pricing_override(client, server, governed, monkeypatch):
    import traccia.cost_resolver as cr
    from traccia.pricing_config import load_pricing_with_source

    monkeypatch.delenv("TRACCIA_PRICING_OVERRIDE_JSON", raising=False)
    monkeypatch.delenv("AGENT_DASHBOARD_PRICING_JSON", raising=False)
    table, source, generated_at = load_pricing_with_source({MODEL: {"prompt": 1.0, "completion": 2.0}})
    monkeypatch.setattr(cr, "_resolver", cr.CostResolver(table, source, generated_at))
    stream = create(client, stream=True)
    for c in stream:
        if c.choices[0].delta.content:
            break
    stream.close()
    # The same resolver as provider-reported usage, so the override applies, not server pricing.
    assert [settle_fields(s) for s in server.settles] == [estimated(1, rates=(1.0, 2.0))]


def test_unpriced_model_estimate_is_left_to_the_server(client, server, governed, monkeypatch):
    monkeypatch.setattr(groq_mod, "_compute_cost", lambda *a: None)
    stream = create(client, stream=True)
    for c in stream:
        if c.choices[0].delta.content:
            break
    stream.close()
    assert settle_fields(server.settles[0])["actual_usd"] is None
    assert settle_fields(server.settles[0])["output_tokens"] == 1


def test_open_stream_at_shutdown_is_aborted_and_charged_the_reservation(client, tracer, server, governed):
    stream = create(client, stream=True)
    response = stream.response
    it = iter(stream)
    next(it)
    next(it)
    settle_open_calls(timeout=5)
    assert response.is_closed
    assert tracer.spans[0].ended == 1
    assert tracer.spans[0].attributes["llm.interrupted"] == "shutdown"
    assert [settle_fields(s)["actual_usd"] for s in server.settles] == [RESERVED]
    stream.close()
    assert len(server.settles) == 1


def test_request_in_flight_at_shutdown_is_charged_the_reservation(client, api, tracer, server):
    api.hold = threading.Event()

    def call():
        with runtime_config.run_identity(agent_id="agent-a", pep_enabled=True):
            create(client)

    caller = threading.Thread(target=call)
    caller.start()
    assert api.entered.wait(timeout=5)
    settle_open_calls(timeout=5)
    assert [settle_fields(s)["actual_usd"] for s in server.settles] == [RESERVED]
    assert tracer.spans[0].attributes["llm.interrupted"] == "shutdown"

    api.hold.set()
    caller.join(timeout=5)
    # The response that arrives afterwards settles nothing more.
    assert len(server.settles) == 1
    assert tracer.spans[0].ended == 1


def test_settle_open_calls_stops_at_its_time_limit():
    class SlowCall:
        def __init__(self):
            self.ran = False

        def settle_at_shutdown(self):
            self.ran = True
            time.sleep(0.3)

    calls = [SlowCall() for _ in range(5)]
    for call in calls:
        pep.track_open_call(call)
    try:
        started = time.monotonic()
        settle_open_calls(timeout=0.5)
        assert time.monotonic() - started < 1.0
        assert 0 < sum(call.ran for call in calls) < 5
    finally:
        for call in calls:
            pep.untrack_open_call(call)


def test_stop_tracing_settles_open_calls(monkeypatch):
    from traccia import auto

    timeouts = []
    monkeypatch.setattr(pep, "settle_open_calls", lambda timeout=10.0: timeouts.append(timeout))
    auto.stop_tracing(flush_timeout=3.0)
    assert timeouts == [3.0]


def test_error_mid_stream_settles_what_was_streamed_and_propagates(client, api, tracer, server, governed):
    api.chunks = CHUNKS[:2]
    api.stream_error = "connection reset"
    with pytest.raises(groq.APIError):
        for _ in create(client, stream=True):
            pass
    [span] = tracer.spans
    assert span.ended == 1
    assert span.status[0] == SpanStatus.ERROR
    assert span.attributes["llm.completion"] == "Par"
    assert [settle_fields(s) for s in server.settles] == [estimated(1)]


def test_error_before_any_chunk_releases(client, api, server, governed):
    api.chunks = []
    api.stream_error = "overloaded"
    with pytest.raises(groq.APIError):
        for _ in create(client, stream=True):
            pass
    assert settle_fields(server.settles[0])["release"] is True


def test_error_after_the_role_chunk_settles_a_one_token_estimate(client, api, server, governed):
    # Groq had started the completion, so it is billed: never release it.
    api.chunks = CHUNKS[:1]
    api.stream_error = "overloaded"
    with pytest.raises(groq.APIError):
        for _ in create(client, stream=True):
            pass
    assert [settle_fields(s) for s in server.settles] == [estimated(1)]


def test_stream_stopped_on_the_role_chunk_settles_an_estimate_not_the_reservation(client, server, governed):
    stream = create(client, stream=True)
    for _ in stream:
        break
    stream.close()
    assert [settle_fields(s) for s in server.settles] == [estimated(1)]


def test_stream_that_ended_without_output_or_usage_settles_a_one_token_estimate(client, api, server, governed):
    api.chunks = [chunk({}, "stop")]
    for _ in create(client, stream=True):
        pass
    assert [settle_fields(s) for s in server.settles] == [estimated(1)]


# A reasoning model streams its reasoning in delta.reasoning before any content.
REASONING = "x" * 42_500  # 10,625 output tokens at 4 characters a token
REASONING_CHUNKS = [
    chunk({"role": "assistant", "content": ""}, x_groq={"id": "req_1"}),
    chunk({"reasoning": REASONING[:20_000]}),
    chunk({"reasoning": REASONING[20_000:]}),
    chunk({"content": "Paris."}),
    chunk({}, "stop", x_groq={"id": "req_1", "usage": USAGE}),
]


def test_stream_stopped_during_reasoning_settles_the_reasoning_estimate(client, api, server, governed):
    api.chunks = REASONING_CHUNKS
    stream = create(client, stream=True)
    for i, _ in enumerate(stream):
        if i == 2:  # both reasoning chunks read, no content or usage yet
            break
    stream.close()
    assert [settle_fields(s) for s in server.settles] == [estimated(10_625)]


def test_error_during_reasoning_settles_the_reasoning_estimate(client, api, server, governed):
    api.chunks = REASONING_CHUNKS[:3]
    api.stream_error = "overloaded"
    with pytest.raises(groq.APIError):
        for _ in create(client, stream=True):
            pass
    assert [settle_fields(s) for s in server.settles] == [estimated(10_625)]


def test_response_without_usage_counts_reasoning_as_output(client, api, server, governed):
    message = {"role": "assistant", "content": "Paris.", "reasoning": REASONING}
    api.completion = {**COMPLETION, "usage": None, "choices": [{"index": 0, "finish_reason": "stop", "message": message}]}
    create(client)
    # 42,500 + 6 characters: 10,627 output tokens.
    assert settle_fields(server.settles[0]) == estimated(10_627)


def test_settle_runs_in_the_background(client, server, governed, monkeypatch):
    import threading

    from traccia.governance import pep

    gate = threading.Event()
    original = pep.finish_llm_call

    def slow_finish(*args, **kwargs):
        gate.wait(5)
        original(*args, **kwargs)

    monkeypatch.setattr(pep, "finish_llm_call", slow_finish)
    assert create(client).choices[0].message.content == "Paris."
    assert server._settles == []  # the caller got its data before the settle finished
    gate.set()
    assert [settle_fields(s) for s in server.settles] == [priced()]


def use_pricing(monkeypatch, table):
    """Make *table* the process pricing table, as start_tracing() does."""
    from traccia.cost_resolver import CostResolver

    monkeypatch.setattr("traccia.cost_resolver._resolver", CostResolver(table, "override"))


def test_groq_price_is_never_another_providers(monkeypatch):
    use_pricing(monkeypatch, {
        "groq/llama-3.3-70b-versatile": {"prompt": 0.00059, "completion": 0.00079, "_provider": "groq"},
        "groq/openai/gpt-oss-120b": {"prompt": 0.00015, "completion": 0.0006, "_provider": "groq"},
        "fireworks_ai/accounts/fireworks/models/gemma2-9b-it": {"prompt": 0.0002, "completion": 0.0002, "_provider": "fireworks_ai"},
        "openai/gpt-oss-120b": {"prompt": 0.01, "completion": 0.01, "_provider": "openrouter"},
        "ovhcloud/Qwen3-32B": {"prompt": 0.00008, "completion": 0.00023, "_provider": "ovhcloud"},
        "groq/qwen/qwen3-32b": {"prompt": 0.00029, "completion": 0.00059, "_provider": "groq"},
    })
    assert groq_mod._compute_cost("llama-3.3-70b-versatile", 1000, 1000) == pytest.approx(0.00138)
    assert groq_mod._compute_cost("openai/gpt-oss-120b", 1000, 1000) == pytest.approx(0.00075)
    assert groq_mod._compute_cost("qwen/qwen3-32b", 1000, 1000) == pytest.approx(0.00088)
    assert groq_mod._compute_cost("gemma2-9b-it", 1000, 1000) is None


def test_groq_price_needs_the_same_base_name(monkeypatch):
    use_pricing(monkeypatch, {
        "groq/llama-3.3-70b-versatile": {"prompt": 0.00059, "completion": 0.00079, "_provider": "groq"},
        "groq/openai/gpt-oss-120b": {"prompt": 0.00015, "completion": 0.0006, "_provider": "groq"},
    })
    # A date suffix is not part of the base name, as in the platform's matcher.
    assert groq_mod._compute_cost("openai/gpt-oss-120b-2025-08-05", 1000, 1000) == pytest.approx(0.00075)
    # Another model whose name starts with a Groq model's name is not priced as that model.
    assert groq_mod._compute_cost("llama-3.3-70b-versatile-0915", 1000, 1000) is None
    assert groq_mod._compute_cost("openai/gpt-oss-120b-long", 1000, 1000) is None
    assert groq_mod._compute_cost("llama-3.3-70b", 1000, 1000) is None


def test_groq_price_uses_the_process_pricing_table(monkeypatch):
    # A pricing_override given to start_tracing() reaches the settled cost.
    use_pricing(monkeypatch, {"groq/llama-3.3-70b-versatile": {"prompt": 0.01, "completion": 0.02, "_provider": "groq"}})
    assert groq_mod._compute_cost("llama-3.3-70b-versatile", 1000, 1000) == pytest.approx(0.03)


def test_cached_prompt_tokens_are_priced_at_the_cached_rate(client, api, tracer, server, governed, monkeypatch):
    use_pricing(monkeypatch, {
        f"groq/{MODEL}": {"prompt": 0.001, "completion": 0.003, "cached_prompt": 0.0005, "_provider": "groq"},
    })
    usage = {**USAGE, "prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 800}}
    api.completion = {**COMPLETION, "usage": usage}
    create(client)
    a = tracer.spans[0].attributes
    assert a["llm.usage.prompt_tokens"] == 200
    assert a["llm.usage.cache_read_tokens"] == 800
    # 200 uncached at 0.001/1K + 800 cached at 0.0005/1K + 3 output at 0.003/1K
    expected = 0.0002 + 0.0004 + 0.000009
    assert settle_fields(server.settles[0]) == {
        **priced(cost=pytest.approx(expected)),
        "input_tokens": 1000,
    }


def test_cached_tokens_without_a_cached_rate_use_the_input_rate(monkeypatch):
    use_pricing(monkeypatch, {f"groq/{MODEL}": {"prompt": 0.001, "completion": 0.003, "_provider": "groq"}})
    assert groq_mod._compute_cost(MODEL, 1000, 0, 800) == pytest.approx(0.001)


def test_span_cost_matches_the_settled_cost(client, api, tracer, server, governed, monkeypatch):
    from traccia.processors.cost_processor import CostAnnotatingProcessor

    table = {
        "groq/qwen/qwen3-32b": {"prompt": 0.00029, "completion": 0.00059, "_provider": "groq"},
        "ovhcloud/Qwen3-32B": {"prompt": 0.00008, "completion": 0.00023, "_provider": "ovhcloud"},
    }
    use_pricing(monkeypatch, table)
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000}
    # Requested under one name, run and billed by Groq under another.
    api.completion = {**COMPLETION, "model": "qwen/qwen3-32b", "usage": usage}
    create(client)
    span = tracer.spans[0]
    assert span.attributes["llm.model"] == "qwen/qwen3-32b"
    assert span.attributes["llm.request.model"] == MODEL
    CostAnnotatingProcessor(pricing_table=table).on_end(span)
    assert span.attributes["llm.cost.usd"] == pytest.approx(0.00088)
    assert span.attributes["llm.pricing.model_key"] == "groq/qwen/qwen3-32b"
    assert server.settles[0]["actual_usd"] == pytest.approx(0.00088)


def test_stream_settles_for_the_agent_that_made_the_call(client, server):
    with runtime_config.run_identity(agent_id="agent-a", pep_enabled=True):
        stream = create(client, stream=True)
    with runtime_config.run_identity(agent_id="agent-b", pep_enabled=False):
        for _ in stream:
            pass
    assert [s["agent_id"] for s in server.settles] == ["agent-a"]


def test_raw_stream_response_traces_the_parsed_stream(client, server, governed):
    raw = client.chat.completions.with_raw_response.create(model=MODEL, messages=MESSAGES, stream=True)
    assert server.settles == []
    text = "".join(c.choices[0].delta.content or "" for c in raw.parse())
    assert text == "Paris."
    assert [settle_fields(s) for s in server.settles] == [priced()]


# -- async -------------------------------------------------------------------------


def test_async_completion_is_settled_with_its_cost(async_client, tracer, server, governed):
    response = asyncio.run(create(async_client))
    assert response.choices[0].message.content == "Paris."
    assert tracer.spans[0].ended == 1
    assert server.checks[0]["context"]["vendor"] == "groq"
    assert [settle_fields(s) for s in server.settles] == [priced()]


def test_async_stream_read_to_the_end_is_priced(async_client, tracer, server, governed):
    async def run():
        stream = await create(async_client, stream=True)
        return "".join([c.choices[0].delta.content or "" async for c in stream])

    assert asyncio.run(run()) == "Paris."
    assert tracer.spans[0].ended == 1
    assert [settle_fields(s) for s in server.settles] == [priced()]


def test_async_stream_closed_early_settles_what_was_streamed(async_client, server, governed):
    async def run():
        stream = await create(async_client, stream=True)
        async for c in stream:
            if c.choices[0].delta.content:
                break
        await stream.close()

    asyncio.run(run())
    assert [settle_fields(s) for s in server.settles] == [estimated(1)]


def test_async_stream_dropped_mid_read_charges_the_reservation(async_client, server, governed):
    closed = []

    async def run():
        stream = await create(async_client, stream=True)
        response = stream.response
        it = stream.__aiter__()
        await it.__anext__()
        await it.__anext__()
        del it, stream
        gc.collect()
        await asyncio.sleep(0.05)
        closed.append(response.is_closed)

    asyncio.run(run())
    assert closed == [True]
    assert [settle_fields(s)["actual_usd"] for s in server.settles] == [RESERVED]


def test_async_http_error_releases(async_client, api, server, governed):
    api.status = 400
    with pytest.raises(groq.BadRequestError):
        asyncio.run(create(async_client))
    assert settle_fields(server.settles[0])["release"] is True


def test_async_stream_settles_for_the_creating_agent(async_client, server):
    async def run():
        with runtime_config.run_identity(agent_id="agent-a", pep_enabled=True):
            stream = await create(async_client, stream=True)
        with runtime_config.run_identity(agent_id="agent-b", pep_enabled=False):
            async for _ in stream:
                pass

    asyncio.run(run())
    assert [s["agent_id"] for s in server.settles] == ["agent-a"]


# -- regression: no billed Groq call may settle for $0 ----------------------------


SCENARIOS = {
    "completion": lambda c: create(c),
    "stream": lambda c: [x for x in create(c, stream=True)],
    "stream_closed_early": lambda c: (lambda s: (next(iter(s)), next(iter(s)), s.close()))(create(c, stream=True)),
    "raw_response": lambda c: c.chat.completions.with_raw_response.create(model=MODEL, messages=MESSAGES).parse(),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_billed_call_is_never_settled_without_a_cost_or_tokens(name, client, server, governed):
    SCENARIOS[name](client)
    gc.collect()
    [settle] = server.settles
    assert settle["release"] is False
    assert settle["actual_usd"] is not None or (
        settle["input_tokens"] is not None and settle["output_tokens"] is not None
    ), f"{name} would be refunded by the server: {settle}"


def test_reshape_never_changes_the_callers_params(client, api, server, governed):
    server.decision = {
        "effect": "reshape",
        "would_have": False,
        "obligations": {"reserved_usd": RESERVED, "cheaper_model": "llama-3.1-8b-instant", "clamp_max_tokens": 10},
    }
    params = {"model": MODEL, "messages": MESSAGES, "max_tokens": 4000}
    client.chat.completions.create(**params)
    client.chat.completions.create(**params)
    assert params == {"model": MODEL, "messages": MESSAGES, "max_tokens": 4000}
    # Each call is checked as written, and sends the reshaped request.
    assert [c["context"]["model"] for c in server.checks] == [MODEL, MODEL]
    assert [(r["model"], r["max_tokens"]) for r in api.requests] == [("llama-3.1-8b-instant", 10)] * 2
