import copy
import json
import logging
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.context import attach as otel_context_attach, detach as otel_context_detach

from traccia.governance.pep import check_policy, enforce_llm_call, enforce_tool_call
from traccia.governance.policy import AgentBlockedError, ApprovalPending, pending_tool_result
from traccia.instrumentation.requests import _should_skip_http_instrumentation
from traccia import runtime_config


def test_skips_policy_check_url():
    assert _should_skip_http_instrumentation("https://app.traccia.ai/api/v1/policy/check")
    assert _should_skip_http_instrumentation("http://localhost:8001/api/v1/policy/settle")


def test_enforce_llm_noop_without_pep_flag():
    kwargs = {"model": "gpt-4o"}
    assert enforce_llm_call(kwargs) is None
    assert kwargs["model"] == "gpt-4o"


def test_before_llm_deny_raises_readable_error():
    decision = {
        "id": "dec-1",
        "effect": "deny",
        "would_have": False,
        "reasons": ["spend exceeded cap"],
        "remaining_budget_usd": 0.0,
        "obligations": {},
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                with pytest.raises(AgentBlockedError) as exc:
                    check_policy(action={"type": "llm_call", "model": "gpt-4o"})
                assert "spend exceeded cap" in str(exc.value)
                assert exc.value.decision_id == "dec-1"


def test_before_llm_reshape_swaps_model():
    decision = {
        "id": "dec-2",
        "effect": "reshape",
        "would_have": False,
        "reasons": ["use cheaper model"],
        "obligations": {"cheaper_model": "gpt-4o-mini"},
        "remaining_budget_usd": 1.0,
    }
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                out = enforce_llm_call(kwargs)
                assert out["effect"] == "reshape"
                assert kwargs["model"] == "gpt-4o-mini"


@pytest.mark.parametrize(
    "cap, vendor, api, sent_cap, clamped",
    [
        ({"max_output_tokens": 100}, "openai", "responses", 100, {"max_output_tokens": 5}),
        ({"max_completion_tokens": 100}, "openai", None, 100, {"max_completion_tokens": 5}),
        ({"max_tokens": 100}, "openai", None, 100, {"max_tokens": 5}),
        ({}, "openai", None, None, {"max_completion_tokens": 5}),
        ({"max_tokens": 100}, "anthropic", None, 100, {"max_tokens": 5}),
        (
            {"generation_config": {"max_output_tokens": 100, "temperature": 0.2}},
            "google_gemini",
            None,
            100,
            {"generation_config": {"max_output_tokens": 5, "temperature": 0.2}},
        ),
    ],
)
def test_before_llm_reshape_clamps_the_api_token_cap(cap, vendor, api, sent_cap, clamped):
    kwargs = {"model": "m", **cap}
    session = _reshape(kwargs, {"clamp_max_tokens": 5}, vendor=vendor, api=api)
    assert session.post.call_args.kwargs["json"]["context"]["max_tokens"] == sent_cap
    assert kwargs == {"model": "m", **clamped}


def _reshape(kwargs, obligations, **enforce_kwargs):
    decision = {"id": "dec-c", "effect": "reshape", "would_have": False, "obligations": obligations}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                enforce_llm_call(kwargs, **enforce_kwargs)
    return session


try:
    from openai import NOT_GIVEN, omit
except ImportError:  # stand-ins for the SDK's omission sentinels: falsy instances of these classes

    class NotGiven:
        def __bool__(self):
            return False

    class Omit(NotGiven):
        pass

    NOT_GIVEN, omit = NotGiven(), Omit()


@pytest.mark.parametrize("sentinel", [NOT_GIVEN, omit])
def test_reshape_ignores_openai_omission_sentinels(sentinel):
    # chat.completions.stream() forwards every parameter it was not given as a sentinel.
    kwargs = {"model": "gpt-4o", "max_completion_tokens": sentinel, "max_tokens": 100, "stream": True}
    session = _reshape(kwargs, {"clamp_max_tokens": 5}, vendor="openai")
    payload = session.post.call_args.kwargs["json"]
    json.dumps(payload)
    assert payload["context"]["max_tokens"] == 100
    assert kwargs["max_tokens"] == 5
    assert kwargs["max_completion_tokens"] is sentinel


@pytest.mark.parametrize("vendor, api, key", [("openai", "responses", "max_output_tokens"), ("anthropic", None, "max_tokens")])
def test_policy_check_never_sees_an_omission_sentinel(vendor, api, key):
    kwargs = {"model": "m", key: NOT_GIVEN}
    session = _reshape(kwargs, {}, vendor=vendor, api=api)
    payload = session.post.call_args.kwargs["json"]
    json.dumps(payload)
    assert payload["context"]["max_tokens"] is None


try:
    from google.genai._gaos.models.createinteraction import CreateInteractionRequest
except ImportError:  # stand-in with the SDK's shape: body.generation_config.max_output_tokens
    from pydantic import BaseModel

    class _GenerationConfig(BaseModel):
        max_output_tokens: Optional[int] = None
        temperature: Optional[float] = None

    class _Body(BaseModel):
        model: str
        input: Any = None
        generation_config: Optional[_GenerationConfig] = None

    class CreateInteractionRequest(BaseModel):
        body: _Body


def _gemini_request(form):
    body = {"model": "gemini-2.5-flash", "input": "hello there", "generation_config": {"max_output_tokens": 100, "temperature": 0.2}}
    return {"body": body} if form == "dict" else CreateInteractionRequest(body=body)


def _gemini_body(request):
    body = request["body"] if isinstance(request, dict) else request.body
    return body if isinstance(body, dict) else body.model_dump(exclude_none=True)


@pytest.mark.parametrize("form", ["dict", "model"])
def test_reshape_rewrites_a_gemini_request_copy(form):
    request = _gemini_request(form)
    original = copy.deepcopy(request)
    kwargs = {"request": request}
    session = _reshape(kwargs, {"clamp_max_tokens": 5, "cheaper_model": "gemini-2.5-flash-lite"}, vendor="google_gemini")
    context = session.post.call_args.kwargs["json"]["context"]
    assert context["model"] == "gemini-2.5-flash"
    assert context["max_tokens"] == 100
    assert context["input_tokens"]
    assert set(kwargs) == {"request"}  # the SDK rejects request= mixed with body fields
    assert type(kwargs["request"]) is type(request)
    assert _gemini_body(kwargs["request"]) == {
        "model": "gemini-2.5-flash-lite",
        "input": "hello there",
        "generation_config": {"max_output_tokens": 5, "temperature": 0.2},
    }
    assert request == original


class _UnrebuildableRequest:
    """A request whose model rejects the rewritten body."""

    body = {"model": "gemini-2.5-flash", "input": "hi", "generation_config": {"max_output_tokens": 100}}

    def model_dump(self, **_kwargs):
        return {"body": dict(self.body)}

    @classmethod
    def model_validate(cls, _data):
        raise ValueError("invalid request")


@pytest.mark.parametrize(
    "request_", [object(), {"body": {"agent": "deep-research", "input": "hi"}}, _UnrebuildableRequest()]
)
def test_reshape_blocks_a_gemini_request_it_cannot_clamp(request_):
    kwargs = {"request": request_}
    decision = {"id": "dec-c", "effect": "reshape", "would_have": False, "obligations": {"clamp_max_tokens": 5, "reserved_usd": 0.01}}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                with pytest.raises(AgentBlockedError) as exc:
                    enforce_llm_call(kwargs, vendor="google_gemini")
    assert "max_output_tokens" in str(exc.value)
    assert exc.value.decision_id == "dec-c"
    assert kwargs == {"request": request_}
    settle = session.post.call_args
    assert settle.args[0].endswith("/api/v1/policy/settle") and settle.kwargs["json"]["release"] is True


def test_before_llm_warn_proceeds():
    decision = {
        "id": "dec-3",
        "effect": "deny",
        "would_have": True,
        "reasons": ["would deny"],
        "obligations": {},
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                assert check_policy(action={"type": "llm_call"})["would_have"] is True


def test_stamp_span_sets_matched_activation():
    from traccia.governance.pep import _stamp_span

    span = MagicMock()
    decision = {
        "id": "dec-obs",
        "effect": "deny",
        "would_have": True,
        "activation": "observe",
        "reasons": ["would deny"],
    }
    with patch("traccia.context.get_current_span", return_value=span):
        _stamp_span(decision)
    span.set_attribute.assert_any_call("traccia.policy.activation", "observe")


def test_stamp_span_omits_activation_when_unmatched():
    from traccia.governance.pep import _stamp_span

    span = MagicMock()
    decision = {"id": "dec-allow", "effect": "allow", "would_have": False, "reasons": []}
    with patch("traccia.context.get_current_span", return_value=span):
        _stamp_span(decision)
    keys = [c.args[0] for c in span.set_attribute.call_args_list]
    assert "traccia.policy.activation" not in keys


def test_before_tool_calls_check():
    decision = {"id": "dec-4", "effect": "allow", "would_have": False, "reasons": []}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                out = enforce_tool_call("search", {"q": "x"})
                assert out["effect"] == "allow"
                body = session.post.call_args.kwargs["json"]
                assert body["action"]["type"] == "tool_call"
                assert body["action"]["name"] == "search"


def test_before_tool_deny_sends_amount_payload():
    decision = {
        "id": "dec-refund",
        "effect": "deny",
        "would_have": False,
        "reasons": ["Refund Guard amount $80.000 is above $50.000"],
        "obligations": {},
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                with pytest.raises(AgentBlockedError) as exc:
                    enforce_tool_call("issue_refund", {"amount": 80})
                assert "above" in str(exc.value)
                body = session.post.call_args.kwargs["json"]
                assert body["action"] == {"type": "tool_call", "name": "issue_refund"}
                assert body["context"]["input"] == {"amount": 80}
                assert body["context"]["tool_name"] == "issue_refund"


def test_before_tool_queue_raises_approval_pending_and_does_not_look_like_a_block():
    decision = {
        "id": "dec-queue",
        "effect": "queue",
        "would_have": False,
        "approval_id": "appr-1",
        "approval_expires_at": "2026-10-06T00:15:00+00:00",
        "reasons": ["needs approval"],
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                with pytest.raises(ApprovalPending) as exc:
                    enforce_tool_call("issue_refund", {"amount": 40})
                assert not isinstance(exc.value, AgentBlockedError)
                assert exc.value.approval_id == "appr-1"
                assert pending_tool_result(exc.value)["status"] == "pending_approval"


def test_before_tool_would_have_queue_still_returns():
    decision = {"id": "dec-wh", "effect": "queue", "would_have": True, "reasons": ["needs approval"]}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                out = enforce_tool_call("issue_refund", {"amount": 40})
                assert out["would_have"] is True


def test_before_tool_deny_sends_high_risk_name():
    decision = {
        "id": "dec-risk",
        "effect": "deny",
        "would_have": False,
        "reasons": ["High-Risk Tool delete_account is not allowed unsupervised"],
        "obligations": {},
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                with pytest.raises(AgentBlockedError) as exc:
                    enforce_tool_call("delete_account", {})
                assert "delete_account" in str(exc.value)
                body = session.post.call_args.kwargs["json"]
                assert body["action"]["name"] == "delete_account"


def test_before_tool_deny_sends_shell_pattern():
    decision = {
        "id": "dec-shell",
        "effect": "deny",
        "would_have": False,
        "reasons": ["Dangerous Shell Commands matched rm -rf"],
        "obligations": {},
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                with pytest.raises(AgentBlockedError) as exc:
                    enforce_tool_call("shell", {"command": "rm -rf /"})
                assert "rm -rf" in str(exc.value)
                body = session.post.call_args.kwargs["json"]
                assert body["action"]["name"] == "shell"
                assert body["context"]["input"]["command"] == "rm -rf /"


def test_trace_ids_format_ints_as_hex():
    from traccia.governance.pep import _as_otel_hex

    assert _as_otel_hex(1, 32) == ("0" * 31) + "1"
    assert _as_otel_hex("abc", 32) == "abc"
    assert _as_otel_hex(None, 32) is None


def test_check_http_error_blocks_by_default():
    # TRC-342: with no setting from Traccia and no override, an outage blocks.
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=503, text="down")
                with pytest.raises(AgentBlockedError) as exc:
                    check_policy(action={"type": "llm_call"})
                assert exc.value.reasons == ["check_http_error"]


def test_settle_includes_trace_id():
    from traccia.governance.pep import settle_policy

    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200)
                settle_policy({"obligations": {"reserved_usd": 0.25}}, actual_usd=0.1)
                body = session.post.call_args.kwargs["json"]
                assert body["agent_id"] == "support"
                assert body["reserved_usd"] == 0.25
                assert body["actual_usd"] == 0.1
                assert "trace_id" in body


def test_write_check_carries_last_read_timestamp_and_customer_id():
    from traccia.governance import pep

    pep._last_reads.clear()
    span = MagicMock()
    span.context.trace_id = 5
    span.context.span_id = 6
    span.attributes = {}
    decision = {"id": "dec-fresh", "effect": "allow", "would_have": False, "reasons": []}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.context.get_current_span", return_value=span):
            pep.remember_tool_result("search", {"q": "x"})
            pep.remember_tool_result("get_refund_policy", {"as_of": "2026-09-24T11:40:00Z", "text": "policy"})
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")), \
             patch("traccia.context.get_current_span", return_value=span):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                enforce_tool_call("issue_refund", {"amount": 10, "customer": {"id": "alice"}})
                body = session.post.call_args.kwargs["json"]
                freshness = body["context"]["freshness"]
                assert {"tool": "get_refund_policy", "read_at": "2026-09-24T11:40:00Z", "field": "as_of"} in freshness
                assert body["context"]["customer_id"] == "alice"
    pep._last_reads.clear()


def test_llm_check_sends_prompt_and_real_retrieval_count():
    from traccia.governance import pep

    pep._retrieval_by_trace.clear()
    span = MagicMock()
    span.context.trace_id = 1
    span.context.span_id = 2
    span.attributes = {
        "traccia.prompt.name": "refund-policy",
        "traccia.prompt.label": "production",
        "traccia.prompt.version_id": "ver-1",
    }
    pep.note_retrieval_attributes("00000000000000000000000000000001", {
        "traccia.retrieval.chunk_count": 4,
    })
    decision = {"id": "dec-pin", "effect": "allow", "would_have": False, "reasons": []}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                with patch("traccia.context.get_current_span", return_value=span):
                    session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                    enforce_llm_call({"model": "gpt-4o-mini"})
                    body = session.post.call_args.kwargs["json"]
                    assert body["context"]["prompt"] == {
                        "name": "refund-policy",
                        "label": "production",
                        "version_id": "ver-1",
                    }
                    assert body["context"]["retrieval"]["chunk_count"] == 4
                    assert body["context"]["retrieval"]["present"] is True
    pep._retrieval_by_trace.clear()


def test_llm_check_uses_prompt_compiled_before_the_provider_span():
    from traccia.governance import pep

    pep._prompt_by_trace.clear()
    span = MagicMock()
    span.context.trace_id = 1
    span.context.span_id = 9
    span.attributes = {}
    pep.note_prompt_attributes(1, {
        "traccia.prompt.name": "refund-policy",
        "traccia.prompt.label": "production",
    })
    decision = {"id": "dec-parent", "effect": "allow", "would_have": False, "reasons": []}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                with patch("traccia.context.get_current_span", return_value=span):
                    session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                    enforce_llm_call({"model": "gpt-4o-mini"})
                    body = session.post.call_args.kwargs["json"]
                    assert body["context"]["prompt"] == {
                        "name": "refund-policy",
                        "label": "production",
                    }
    pep._prompt_by_trace.clear()


def test_llm_check_omits_prompt_and_does_not_invent_chunk_count():
    from traccia.governance import pep

    pep._retrieval_by_trace.clear()
    span = MagicMock()
    span.context.trace_id = 3
    span.context.span_id = 4
    span.attributes = {}
    decision = {"id": "dec-plain", "effect": "allow", "would_have": False, "reasons": []}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                with patch("traccia.context.get_current_span", return_value=span):
                    session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                    enforce_llm_call({"model": "gpt-4o-mini"})
                    body = session.post.call_args.kwargs["json"]
                    assert "prompt" not in body["context"]
                    assert body["context"]["retrieval"] == {"present": False}
                    assert "chunk_count" not in body["context"]["retrieval"]


def test_before_llm_sends_vendor_for_spend_cap():
    decision = {"id": "dec-v", "effect": "allow", "would_have": False, "obligations": {}}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                enforce_llm_call({"model": "gemini-2.0-flash-001"}, vendor="google_gemini")
                body = session.post.call_args.kwargs["json"]
                assert body["context"]["vendor"] == "google_gemini"


def _fail_closed():
    return runtime_config.run_identity(agent_id="support", pep_enabled=True, pep_fail_mode="closed")


def test_fail_closed_blocks_on_http_error():
    with _fail_closed():
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=503, text="down")
                with pytest.raises(AgentBlockedError) as exc:
                    enforce_tool_call("issue_refund", {"amount": 80})
                assert exc.value.reasons == ["check_http_error"]
                assert "policy fail mode is closed" in str(exc.value)


def test_fail_closed_blocks_on_transport_error():
    with _fail_closed():
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.side_effect = ConnectionError("refused")
                with pytest.raises(AgentBlockedError) as exc:
                    enforce_llm_call({"model": "gpt-4o"})
                assert exc.value.reasons == ["check_error"]


def test_missing_credentials_are_a_configuration_error():
    from traccia.governance.policy import GovernanceConfigError

    with _fail_closed():
        with patch("traccia.governance.pep._credentials", return_value=(None, None)):
            with pytest.raises(GovernanceConfigError):
                enforce_tool_call("search", {"q": "x"})


def test_fail_closed_is_the_default_on_transport_error():
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.side_effect = ConnectionError("refused")
                with pytest.raises(AgentBlockedError) as exc:
                    enforce_tool_call("search", {"q": "x"})
                assert exc.value.reasons == ["check_error"]


def test_fail_closed_still_passes_a_real_allow():
    decision = {"id": "dec-ok", "effect": "allow", "would_have": False, "reasons": []}
    with _fail_closed():
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                assert enforce_tool_call("search", {"q": "x"})["id"] == "dec-ok"


# TRC-289: freshness evidence must not cross agent/run boundaries.

@pytest.fixture
def governed_tracing():
    from traccia import get_tracer_provider, set_tracer_provider
    from traccia.governance import pep
    from traccia.tracer.provider import TracerProvider

    previous = get_tracer_provider()
    set_tracer_provider(TracerProvider())
    for store in (pep._last_reads, pep._retrieval_by_trace, pep._prompt_by_trace):
        store.clear()
    sent = []

    def _post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return MagicMock(status_code=200, json=lambda: {"effect": "allow", "would_have": False, "reasons": []})

    with patch("traccia.governance.govern.check_agent_status"), \
         patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")), \
         patch.object(pep._http_session, "post", side_effect=_post):
        yield sent
    set_tracer_provider(previous)


def _customer_tools():
    from traccia import observe

    @observe(name="read_customer", as_type="tool")
    def read_customer(customer_id):
        return {"id": customer_id, "as_of": "2026-10-14T10:00:00Z"}

    @observe(name="update_customer", as_type="tool")
    def update_customer(customer_id):
        return {"ok": True}

    return read_customer, update_customer


def _write_freshness(sent):
    writes = [p for p in sent if p["action"]["name"] == "update_customer"]
    return [w["context"].get("freshness") for w in writes]


def test_freshness_does_not_leak_to_another_agent_or_run(governed_tracing):
    from traccia.governance.govern import govern

    read_customer, update_customer = _customer_tools()

    @govern(agent_id="agent-a", name="run_a")
    def run_a():
        read_customer("cust-1")

    @govern(agent_id="agent-b", name="run_b")
    def run_b():
        update_customer("cust-1")

    @govern(agent_id="agent-a", name="run_a_again")
    def run_a_again():
        update_customer("cust-1")

    run_a()
    run_b()
    run_a_again()
    assert _write_freshness(governed_tracing) == [None, None]


def test_freshness_kept_within_the_same_run_for_the_same_entity(governed_tracing):
    from traccia.governance.govern import govern

    read_customer, update_customer = _customer_tools()

    @govern(agent_id="agent-a", name="run")
    def run():
        read_customer("cust-1")
        update_customer("cust-1")
        update_customer("cust-2")

    run()
    same, other = _write_freshness(governed_tracing)
    assert same == [{
        "tool": "read_customer",
        "read_at": "2026-10-14T10:00:00Z",
        "field": "as_of",
        "resource_id": "cust-1",
    }]
    assert other is None


def test_evidence_is_dropped_when_the_run_ends(governed_tracing):
    from traccia import observe
    from traccia.context import get_current_span
    from traccia.governance import pep

    read_customer, _update = _customer_tools()

    @observe(name="retrieve")
    def retrieve():
        get_current_span().set_attribute("traccia.retrieval.chunk_count", 3)

    @observe(name="rag_step")
    def rag_step():
        span = get_current_span()
        pep.note_prompt_attributes(span.context.trace_id, {"traccia.prompt.name": "p"})
        retrieve()
        read_customer("cust-1")
        assert len(pep._retrieval_by_trace) == 1
        assert len(pep._prompt_by_trace) == 1
        assert len(pep._last_reads) == 1

    for _ in range(50):
        rag_step()
    assert len(pep._retrieval_by_trace) == 0
    assert len(pep._prompt_by_trace) == 0
    assert len(pep._last_reads) == 0


def test_evidence_store_is_bounded_by_size_and_expiry():
    from traccia.governance.evidence import EvidenceStore

    store = EvidenceStore(max_entries=3, ttl_s=60)
    for i in range(10):
        store.update(f"t{i}", f"t{i}", lambda _old, i=i: i)
    assert len(store) == 3
    assert store.get("t0") is None
    assert store.get("t9") == 9

    import time

    with patch("traccia.governance.evidence.time.monotonic", return_value=time.monotonic() + 61):
        assert store.get("t9") is None


@pytest.fixture
def auto_started_trace(governed_tracing):
    """What init(auto_start_trace=True) does: one root span for the whole process."""
    from opentelemetry import context
    from traccia import auto, get_tracer_provider

    ctx = auto._start_auto_trace(get_tracer_provider(), "root")
    auto._auto_trace_context = ctx
    yield governed_tracing
    auto._auto_trace_context = None
    context.detach(ctx["token"])
    ctx["span"].end()


def test_each_govern_call_is_its_own_run_under_the_auto_started_trace(auto_started_trace):
    from traccia.context import get_current_span
    from traccia.governance import pep
    from traccia.governance.govern import govern

    read_customer, update_customer = _customer_tools()

    def first_run():
        read_customer("cust-1")
        pep.note_prompt_attributes(get_current_span().context.trace_id, {"traccia.prompt.name": "old"})
        update_customer("cust-1")

    def second_run():
        update_customer("cust-1")
        pep.enforce_llm_call({"model": "gpt-4o-mini"})

    govern(agent_id="agent-a", name="first")(first_run)()
    assert len(pep._last_reads) == 0
    assert len(pep._prompt_by_trace) == 0
    govern(agent_id="agent-a", name="second")(second_run)()

    first, second = _write_freshness(auto_started_trace)
    assert first[0]["resource_id"] == "cust-1"
    assert second is None
    llm = [p for p in auto_started_trace if p["action"]["type"] == "llm_call"]
    assert "prompt" not in llm[0]["context"]


def test_concurrent_async_runs_keep_their_own_freshness(governed_tracing):
    import asyncio

    from traccia import observe
    from traccia.governance.govern import govern

    @observe(name="read_customer", as_type="tool")
    async def read_customer(customer_id):
        await asyncio.sleep(0)
        return {"id": customer_id, "as_of": "2026-10-14T10:00:00Z"}

    @observe(name="update_customer", as_type="tool")
    async def update_customer(customer_id):
        await asyncio.sleep(0)
        return {"ok": True}

    async def run_a():
        await read_customer("cust-1")
        await asyncio.sleep(0)
        await update_customer("cust-1")

    async def run_b():
        await asyncio.sleep(0)
        await update_customer("cust-1")

    async def main():
        await asyncio.gather(
            govern(agent_id="agent-a", name="run_a")(run_a)(),
            govern(agent_id="agent-b", name="run_b")(run_b)(),
        )

    asyncio.run(main())
    by_agent = {
        p["agent_id"]: p["context"].get("freshness")
        for p in governed_tracing
        if p["action"]["name"] == "update_customer"
    }
    assert by_agent["agent-a"][0]["tool"] == "read_customer"
    assert by_agent["agent-b"] is None


def test_ending_a_child_through_get_current_span_keeps_run_evidence(governed_tracing):
    from opentelemetry import context
    from opentelemetry.trace import set_span_in_context
    from traccia import get_tracer
    from traccia.context import get_current_span
    from traccia.governance import pep

    tracer = get_tracer("t")
    with tracer.start_as_current_span("root") as root:
        pep.note_retrieval_attributes(root.context.trace_id, {"traccia.retrieval.chunk_count": 2})
        child = tracer.start_span("child")
        token = context.attach(set_span_in_context(child._otel_span))
        get_current_span().end()
        context.detach(token)
        assert len(pep._retrieval_by_trace) == 1
    assert len(pep._retrieval_by_trace) == 0


def test_govern_runs_under_one_parent_trace_do_not_share_reads(governed_tracing):
    from traccia import observe
    from traccia.governance.govern import govern

    read_customer, update_customer = _customer_tools()
    first = govern(agent_id="agent-a", name="first")(lambda: read_customer("cust-1"))
    second = govern(agent_id="agent-a", name="second")(lambda: update_customer("cust-1"))

    @observe(name="request")
    def request():
        first()
        second()

    request()
    assert _write_freshness(governed_tracing) == [None]


def test_tool_level_reads_survive_customer_filtering(governed_tracing):
    from traccia import observe
    from traccia.governance.govern import govern

    @observe(name="get_refund_policy", as_type="tool")
    def get_refund_policy():
        return {"id": "policy-v1", "as_of": "2026-01-01T00:00:00Z"}

    @observe(name="read_customer", as_type="tool")
    def read_customer(customer_id):
        return {"customer_id": customer_id, "as_of": "2026-10-14T10:00:00Z"}

    @observe(name="issue_refund", as_type="tool")
    def issue_refund(customer, amount):
        return {"ok": True}

    def run():
        get_refund_policy()
        read_customer("bob")
        issue_refund({"id": "alice"}, 10)

    govern(agent_id="agent-a", name="run")(run)()
    refund = [p for p in governed_tracing if p["action"]["name"] == "issue_refund"][0]
    assert refund["context"]["freshness"] == [
        {"tool": "get_refund_policy", "read_at": "2026-01-01T00:00:00Z", "field": "as_of"}
    ]


def _stamped_keys(span):
    """The policy attributes on the exported (OTel) span."""
    return {k for k in (span._otel_span.attributes or {}) if k.startswith("traccia.policy.")}


def test_stamp_span_skips_a_span_that_has_ended(caplog):
    """A check that returns after its call was cancelled (its span ended) stamps nothing, and
    OTel logs no "Setting attribute on ended span" warning."""
    from opentelemetry import trace as otel_trace

    from traccia import get_tracer
    from traccia.context import get_current_span
    from traccia.governance.pep import _stamp_span

    decision = {"id": "dec-late", "effect": "allow", "would_have": False}
    with get_tracer("pep-test").start_as_current_span("llm.call") as span:
        ctx = otel_trace.set_span_in_context(span._otel_span)
    # As a worker thread sees it: a copy of the call's context, whose span has since ended.
    token = otel_context_attach(ctx)
    try:
        assert get_current_span() is not None
        with caplog.at_level(logging.WARNING):
            _stamp_span(decision)
    finally:
        otel_context_detach(token)
    assert "ended span" not in caplog.text
    assert not _stamped_keys(span)


def test_stamp_span_still_stamps_live_and_unsampled_spans():
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    from traccia import get_tracer, get_tracer_provider, set_tracer_provider
    from traccia.governance.pep import _stamp_span
    from traccia.tracer.provider import TracerProvider

    decision = {"id": "dec-live", "effect": "allow", "would_have": False}
    with get_tracer("pep-test").start_as_current_span("llm.call") as span:
        _stamp_span(decision)
    assert span._otel_span.attributes["traccia.policy.decision_id"] == "dec-live"

    previous = get_tracer_provider()
    provider = TracerProvider()
    provider.set_sampler(ALWAYS_OFF)
    set_tracer_provider(provider)
    try:
        with get_tracer("pep-test").start_as_current_span("llm.call") as unsampled:
            _stamp_span(decision)
        assert unsampled.attributes["traccia.policy.decision_id"] == "dec-live"
    finally:
        set_tracer_provider(previous)
