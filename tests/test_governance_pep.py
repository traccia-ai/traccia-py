import asyncio
import collections.abc
import contextvars
import copy
import functools
import gc
import hashlib
import hmac
import importlib
import json
import logging
import random
import sys
import threading
import types
from concurrent.futures import Future, ThreadPoolExecutor
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.context import attach as otel_context_attach, detach as otel_context_detach

from traccia import DefiniteFailure, IdempotencyRepeat, idempotency_key, observe
from traccia.governance.idempotency import Unkeyable, derive_key
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
    # With no setting from Traccia and no override, an outage blocks.
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


# Freshness evidence must not cross agent/run boundaries.

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

    # patch.object on the module: on Python < 3.11 the dotted path resolves to the govern() function.
    with patch.object(importlib.import_module("traccia.governance.govern"), "check_agent_status"), \
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


# Idempotent tools (check -> run -> settle commit/release/unconfirmed/abandon).

IDEM_TOKEN = "6f1c2b8e-3d4a-4f5b-9c6d-7e8f9a0b1c2d"
API_KEY = "key"
ENDPOINT = "https://app.traccia.ai/v1/traces"
DOWNSTREAM = "idk1_" + "b" * 64
IN_FLIGHT, REPEAT = ["idempotency_in_flight"], ["idempotency_repeat"]
_JSON = json


def _idem_denied(reasons):
    """What the dashboard adds when a Block idempotency policy's own deny stopped the call."""
    reason = next((r for r in reasons if r.startswith("idempotency_") and "unkeyable" not in r), None)
    return {"idempotency_denied": reason} if reason else {}


def _reply(status, body):
    return MagicMock(status_code=status, json=lambda: body)


class IdemServer:
    """The policy server for idempotent tools.

    A check reserves the call's key (handing back ``downstream_key`` when set), unless the next
    ``script`` entry or the tool's ``deny`` entry is a list of reasons, which denies (a
    ``(reasons, obligations)`` pair sets the obligations). ``unreachable``: every check raises.
    Settles are recorded and answered from ``settle_replies`` in turn ((status, body), or an
    exception to raise), then with 200 ok. ``on_check`` / ``on_settle`` run before each answer, in
    the thread that sent it. ``serialize``: round-trip bodies through JSON, as requests does.
    """

    def __init__(self):
        self.checks, self.settles, self.slept = [], [], []
        self.script, self.deny, self.settle_replies = [], {}, []
        self.downstream_key = None
        self.unreachable = False
        self.serialize = False
        self.on_check = self.on_settle = None

    def post(self, url, headers=None, json=None, timeout=None):
        if self.serialize:
            json = _JSON.loads(_JSON.dumps(json, allow_nan=False))
        if url.endswith("/check"):
            self.checks.append(json)
            if self.unreachable:
                raise ConnectionError("unreachable")
            if self.on_check:
                self.on_check(json)
            denial = self.script.pop(0) if self.script else self.deny.get(json["action"]["name"])
            if denial is not None:
                reasons, obligations = denial if isinstance(denial, tuple) else (denial, _idem_denied(denial))
                return _reply(200, {"id": "dec-deny", "effect": "deny", "would_have": False,
                                    "reasons": reasons, "obligations": obligations})
            obligations = {}
            idem = (json.get("context") or {}).get("idempotency") or {}
            if idem.get("key"):
                obligations["idempotency"] = {"key": idem["key"], "token": IDEM_TOKEN, "window_s": 3600}
            if self.downstream_key is not None:
                obligations["idempotency_downstream_key"] = self.downstream_key
            return _reply(200, {"id": "dec-idem", "effect": "allow", "would_have": False, "reasons": [],
                                "obligations": obligations})
        if self.on_settle:
            self.on_settle(json)
        self.settles.append(json)
        reply = self.settle_replies.pop(0) if self.settle_replies else (200, {"ok": True})
        if isinstance(reply, BaseException):
            raise reply
        return _reply(*reply)


@pytest.fixture
def idem_server(monkeypatch):
    from traccia import get_tracer_provider, set_tracer_provider
    from traccia.governance import tool_call
    from traccia.tracer.provider import TracerProvider

    server = IdemServer()
    # A fake clock for in-flight waits: sleeps are recorded, not slept; jitter is off (own test).
    now = [0.0]

    def sleep(seconds):
        server.slept.append(seconds)
        now[0] += seconds

    async def async_sleep(seconds):
        sleep(seconds)

    monkeypatch.setattr(tool_call, "_monotonic", lambda: now[0])
    monkeypatch.setattr(tool_call, "_sleep", sleep)
    monkeypatch.setattr(tool_call, "_async_sleep", async_sleep)
    monkeypatch.setattr(tool_call, "_jittered", lambda delay: delay)
    monkeypatch.setattr("traccia.governance.pep._credentials", lambda: (API_KEY, ENDPOINT))
    session = MagicMock()
    session.post.side_effect = server.post
    monkeypatch.setattr("traccia.governance.pep._http_session", session)
    previous = get_tracer_provider()
    set_tracer_provider(TracerProvider())
    yield server
    set_tracer_provider(previous)


def _identity(pep=True, **kwargs):
    return runtime_config.run_identity(pep_enabled=pep, agent_id="a1", **kwargs)


def idem_tool(name, **opts):
    return observe(name=name, as_type="tool", idempotent=True, **opts)


def _current_span():
    from traccia.context import get_current_span

    return get_current_span()


def _keyed(call_key, api_key=API_KEY):
    """What a tool passes downstream with no per-attempt key: the call key, keyed by the API key."""
    return "idk1_" + hmac.new(api_key.encode(), call_key.encode(), hashlib.sha256).hexdigest()


def _outcomes(server):
    return [s["idempotency"]["outcome"] for s in server.settles]


def _run(is_async, coro_fn):
    return asyncio.run(coro_fn()) if is_async else coro_fn()


# --- the call lifecycle ----------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_successful_idempotent_call_commits(idem_server, is_async):
    seen = {}

    def body():
        seen["key"], seen["span"] = idempotency_key(), _current_span()
        return {"sent": True}

    if is_async:
        @idem_tool("send_email", key_fields=["to"])
        async def send_email(to, text):
            await asyncio.sleep(0)
            return body()

        async def call():
            with _identity():
                return await send_email("x@y.z", "hello")
        assert asyncio.run(call()) == {"sent": True}
    else:
        @idem_tool("send_email", key_fields=["to"])
        def send_email(to, text):
            return body()

        with _identity():
            assert send_email("x@y.z", "hello") == {"sent": True}

    [check] = idem_server.checks
    key = derive_key("a1", "send_email", {"to": "x@y.z"})
    assert check["action"]["idempotent"] is True
    # the check carries the offline key, so a fresh attempt after a lost answer adopts it
    assert check["context"]["idempotency"] == {"key": key, "fallback_key": _keyed(key)}
    assert seen["key"] == _keyed(key)
    [settle] = idem_server.settles
    assert settle["idempotency"] == {"key": key, "token": IDEM_TOKEN, "window_s": 3600, "outcome": "commit"}
    assert settle["release"] is False and settle["agent_id"] == "a1"
    assert "reserved_usd" not in settle
    assert "hello" not in json.dumps(settle)
    # spans carry the downstream key, never the call key
    assert seen["span"].attributes["traccia.idempotency.downstream_key"] == _keyed(key)
    assert "traccia.idempotency.key" not in seen["span"].attributes


def test_an_async_tool_checks_and_settles_off_the_event_loop(idem_server):
    """Each request waits until the event loop has run again: it can only if the request is
    answered on another thread. The settle is still awaited before the call returns."""
    tick = threading.Event()
    loop_ran = []

    def wait_for_the_loop(_body):
        tick.clear()
        loop_ran.append(tick.wait(5))

    idem_server.on_check = idem_server.on_settle = wait_for_the_loop

    @idem_tool("charge_async")
    async def charge_async(amount):
        return "ok"

    async def ticker():
        while True:
            tick.set()
            await asyncio.sleep(0.001)

    async def call():
        with _identity():
            t = asyncio.ensure_future(ticker())
            result = await charge_async(10)
            settled_at_return = list(idem_server.settles)
            t.cancel()
            return result, settled_at_return

    result, settled_at_return = asyncio.run(call())
    assert result == "ok" and loop_ran == [True, True]
    assert [s["idempotency"]["outcome"] for s in settled_at_return] == ["commit"]


class ReadTimeout(Exception):
    """Named like requests/httpx ReadTimeout: the request was sent, the reply never came."""


class ConnectTimeout(TimeoutError):
    """Named like requests/httpx ConnectTimeout: nothing was sent, even though it is a TimeoutError."""


class ProviderUnavailable(Exception):
    pass


def _wrapped_read_timeout():
    try:
        raise ReadTimeout("read timed out")
    except ReadTimeout as exc:
        raise RuntimeError("payment provider unreachable") from exc


@pytest.mark.parametrize("is_async, opts, fail, outcome", [
    pytest.param(False, {}, DefiniteFailure("card declined"), "release", id="a definite failure"),
    pytest.param(True, {}, DefiniteFailure("card declined"), "release", id="a definite failure, async"),
    # anything not known to be definite is held, the tool's own errors included
    pytest.param(False, {}, ValueError("card declined"), "unconfirmed", id="the tool's own error"),
    pytest.param(False, {"release_on": [ProviderUnavailable]}, ProviderUnavailable("rejected"), "release",
                 id="an error the author says is definite"),
    pytest.param(False, {}, TimeoutError("read timed out"), "unconfirmed", id="TimeoutError"),
    pytest.param(False, {}, ReadTimeout("read timed out"), "unconfirmed", id="ReadTimeout"),
    # even a refused connection: the client may have retried after an attempt that did the work
    pytest.param(False, {}, ConnectTimeout("connect timed out"), "unconfirmed", id="ConnectTimeout"),
    pytest.param(False, {}, ConnectionRefusedError("refused"), "unconfirmed", id="ConnectionRefused"),
    pytest.param(False, {}, _wrapped_read_timeout, "unconfirmed", id="a timeout the tool wrapped"),
    # a cancelled call may already have done its side effect
    pytest.param(True, {}, asyncio.CancelledError(), "unconfirmed", id="cancelled in the body, async"),
    pytest.param(False, {"unconfirmed_on": [ProviderUnavailable]}, ProviderUnavailable("503 after sending"),
                 "unconfirmed", id="an error the author lists"),
    # a downstream that dedupes by the key (Stripe): let the retry through with the same key
    pytest.param(False, {"release_unconfirmed": True}, TimeoutError("read timed out"), "abandon",
                 id="release_unconfirmed"),
])
def test_settle_by_outcome(idem_server, is_async, opts, fail, outcome):
    def raise_it():
        if callable(fail):
            fail()
        raise fail

    if is_async:
        @idem_tool("charge", **opts)
        async def charge(amount):
            raise_it()
    else:
        @idem_tool("charge", **opts)
        def charge(amount):
            raise_it()

    async def call_async():
        with _identity():
            await charge(10)

    def call():
        if is_async:
            return asyncio.run(call_async())
        with _identity():
            charge(10)

    with pytest.raises(RuntimeError if callable(fail) else type(fail)):
        call()
    [settle] = idem_server.settles
    assert settle["idempotency"]["outcome"] == outcome
    assert settle["idempotency"]["key"] == derive_key("a1", "charge", {"amount": 10})
    assert settle["release"] is True  # not a commit: no USD is charged


def test_cancelling_an_async_tool_during_its_check_abandons_once_the_check_returns(idem_server):
    from traccia.governance.pep import flush_pending_settles

    check_started, release_check, release_settle = threading.Event(), threading.Event(), threading.Event()

    def hold(_body):
        check_started.set()
        release_check.wait(5)

    idem_server.on_check = hold
    idem_server.on_settle = lambda _body: release_settle.wait(5)
    ran = []

    @idem_tool("charge_async")
    async def charge_async(amount):
        ran.append(True)

    async def call():
        with _identity():
            task = asyncio.ensure_future(charge_async(10))
            await asyncio.to_thread(check_started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert idem_server.settles == []  # the check is still pending
        release_check.set()

    asyncio.run(call())  # returns once the check has, which starts the abandon in the background
    threading.Timer(0.05, release_settle.set).start()
    flush_pending_settles()  # waits for that settle
    assert ran == []
    assert _outcomes(idem_server) == ["abandon"]  # the tool never ran: keep its attempt


def test_cancelling_an_async_tool_during_its_body_is_unconfirmed(idem_server):
    from traccia.governance.pep import flush_pending_settles

    @idem_tool("charge_async", key_fields=["amount"])
    async def charge_async(amount, started):
        started.set()
        await asyncio.sleep(10)

    async def call():
        with _identity():
            started = asyncio.Event()
            task = asyncio.ensure_future(charge_async(10, started))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(call())
    flush_pending_settles()
    assert _outcomes(idem_server) == ["unconfirmed"]  # the body may have done its work


@pytest.mark.parametrize("replies, attempts, stamped", [
    pytest.param([], 1, False, id="applied"),
    pytest.param([ConnectionError("down"), ConnectionError("down")], 2, True, id="unreachable twice"),
    pytest.param([(503, None)], 2, False, id="a 503, then applied"),
    pytest.param([(500, None)], 2, False, id="a 500, then applied"),
    # a 4xx or {"ok": false} is lost: a retry would not help
    pytest.param([(409, {"detail": "bad token"})], 1, True, id="409"),
    pytest.param([(400, None)], 1, True, id="400"),
    pytest.param([(200, {"ok": False, "reason": "redis_unavailable"})], 1, True, id="not applied"),
])
def test_settle_delivery(idem_server, replies, attempts, stamped):
    """A settle that raises or gets a 5xx is tried once more; one that is lost is stamped on the span."""
    idem_server.settle_replies = list(replies)
    seen = {}

    @idem_tool("send_email")
    def send_email(to):
        seen["span"] = _current_span()
        return "ok"

    with _identity():
        assert send_email("x@y.z") == "ok"
    assert _outcomes(idem_server) == ["commit"] * attempts
    assert seen["span"].attributes.get("traccia.idempotency.settle_failed") is (True if stamped else None)


def test_a_failed_settle_never_masks_the_tool_error(idem_server):
    idem_server.settle_replies = [ConnectionError("down"), ConnectionError("down")]

    @idem_tool("charge")
    def charge(amount):
        raise KeyError("boom")

    with _identity():
        with pytest.raises(KeyError, match="boom"):
            charge(1)
    assert _outcomes(idem_server) == ["unconfirmed", "unconfirmed"]


@pytest.mark.parametrize("release, outcome", [(False, "commit"), (True, "release")])
def test_settle_policy_settles_an_idempotency_obligation_without_an_outcome(release, outcome):
    from traccia.governance.pep import settle_policy

    obligation = {"key": "idk1_" + "0" * 64, "token": IDEM_TOKEN, "window_s": 60}
    with runtime_config.run_identity(agent_id="a1", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=(API_KEY, ENDPOINT)):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.side_effect = [MagicMock(status_code=503), MagicMock(status_code=200)]
                settle_policy({"obligations": {"idempotency": obligation}}, release=release)
                assert session.post.call_count == 2  # a 5xx is tried once more
                body = session.post.call_args.kwargs["json"]
    assert body["idempotency"] == {**obligation, "outcome": outcome}
    assert body["release"] is release and "reserved_usd" not in body


def test_llm_settle_unchanged():
    from traccia.governance.pep import settle_policy

    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=(API_KEY, ENDPOINT)):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.side_effect = ConnectionError("down")
                settle_policy({"obligations": {"reserved_usd": 0.25}}, release=True)
                assert session.post.call_count == 1
                body = session.post.call_args.kwargs["json"]
                assert set(body) == {
                    "agent_id", "trace_id", "reserved_usd", "actual_usd", "release",
                    "model", "vendor", "input_tokens", "output_tokens",
                }
                assert body["reserved_usd"] == 0.25 and body["release"] is True
                session.post.reset_mock()
                settle_policy({"obligations": {}})
                settle_policy(None)
                assert session.post.call_count == 0


def test_non_idempotent_tool_unchanged(idem_server):
    @observe(name="search", as_type="tool")
    def search(q):
        with pytest.raises(RuntimeError):
            idempotency_key()
        return "r"

    with _identity():
        assert search("x") == "r"
    [check] = idem_server.checks
    assert check["action"] == {"type": "tool_call", "name": "search"}
    assert "idempotency" not in check["context"]
    assert check["context"]["input"] == {"q": "x"}
    assert idem_server.settles == []


def test_remember_tool_result_gets_the_same_arguments(idem_server):
    class Store:
        @observe(name="read", as_type="tool")
        def read(self, customer_id):
            return {"as_of": "2026-01-01T00:00:00Z"}

    with patch("traccia.governance.pep.remember_tool_result") as remember:
        with _identity():
            Store().read("c1")
    assert remember.call_args.args == ("read", {"as_of": "2026-01-01T00:00:00Z"}, {"customer_id": "c1"})
    assert idem_server.checks[0]["context"]["input"] == {"customer_id": "c1"}


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_an_unreachable_dashboard_blocks_an_idempotent_tool_even_with_open_fail_mode(idem_server, is_async):
    idem_server.unreachable = True
    ran = []

    if is_async:
        @idem_tool("send_email")
        async def send_email(to):
            ran.append(to)

        async def call():
            with _identity(pep_fail_mode="open"):
                await send_email("x@y.z")
    else:
        @idem_tool("send_email")
        def send_email(to):
            ran.append(to)

        def call():
            with _identity(pep_fail_mode="open"):
                send_email("x@y.z")
    with pytest.raises(AgentBlockedError) as blocked:
        _run(is_async, call)
    assert "idempotency_downstream_key_unknown" in blocked.value.reasons
    assert ran == [] and len(idem_server.checks) == 1 and idem_server.settles == []


def test_a_lost_check_after_the_key_rotated_keeps_the_rotated_key(idem_server):
    """After the downstream key rotated, a check whose answer is lost does not fall back to the
    offline key: the call is blocked or keeps the key its retry gets."""
    rotated = "idk1_" + "ab" * 32

    @idem_tool("send_email")
    def send_email(to):
        return idempotency_key()

    idem_server.downstream_key = _keyed(derive_key("a1", "send_email", {"to": "x@y.z"}))
    with _identity(pep_fail_mode="open"):
        send_email("x@y.z")  # first attempt completes
    idem_server.downstream_key = rotated  # the dashboard assigns a new key for the next invocation

    idem_server.unreachable = True  # that invocation's check is lost
    try:
        with _identity(pep_fail_mode="open"):
            lost = send_email("x@y.z")
    except AgentBlockedError:
        return  # the current key is unknown: blocked
    idem_server.unreachable = False
    with _identity(pep_fail_mode="open"):
        retried = send_email("x@y.z")  # retry of the same operation after recovery
    assert retried == rotated
    assert lost == retried, "one operation reached the downstream under two keys"


def test_an_idempotent_tool_with_non_json_arguments_is_checked(idem_server):
    import dataclasses
    import enum
    import uuid
    from datetime import date, datetime, timezone

    idem_server.serialize = True

    class Color(enum.Enum):
        RED = "red"

    @dataclasses.dataclass
    class Line:
        sku: str
        qty: Decimal

    class Model:
        def model_dump(self, mode="python"):
            return {"mode": mode}

    @idem_tool("refund", key_fields=["amount", "at"])
    def refund(amount, at, client, ref, day, color, line, model, tags, secret):
        return idempotency_key()

    secret = type("ApiClient", (), {"__repr__": lambda self: "ApiClient(token=sk_live_x)"})()
    with _identity(pep_fail_mode="closed"):
        key = refund(
            Decimal("10.5"), datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc), object(),
            uuid.UUID("6f1c2b8e-3d4a-4f5b-9c6d-7e8f9a0b1c2d"), date(2026, 1, 2), Color.RED,
            Line("a", Decimal("2")), Model(), ("x", {1}), secret,
        )

    [check] = idem_server.checks
    assert key == _keyed(check["context"]["idempotency"]["key"])
    assert check["context"]["input"] == {
        "amount": "10.5",
        "at": "2026-01-02T03:04:05+00:00",
        "client": "<object>",
        "ref": "6f1c2b8e-3d4a-4f5b-9c6d-7e8f9a0b1c2d",
        "day": "2026-01-02",
        "color": "red",
        "line": {"sku": "a", "qty": "2"},
        "model": {"mode": "json"},
        "tags": ["x", [1]],
        "secret": "<ApiClient>",
    }
    assert "sk_live" not in json.dumps(check)


def test_json_safe_keeps_plain_payloads_byte_identical():
    import enum

    from traccia.governance.pep import _json_safe

    class Level(int, enum.Enum):
        HIGH = 3

    class Name(str, enum.Enum):
        A = "a"

    plain = {"a": 1, "b": [1.5, "x", None, True, {"c": (1, 2)}], 2: "int key", "lvl": Level.HIGH, "n": Name.A}
    assert json.dumps(_json_safe(plain)) == json.dumps(plain)
    nested = current = []
    for _ in range(40):
        current.append([])
        current = current[0]
    json.dumps(_json_safe(nested))  # deep or circular input is cut off, not a crash
    loop = {}
    loop["self"] = loop
    json.dumps(_json_safe(loop))


# --- denials ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("denial, opts, repeat", [
    # the server's idempotency_denied obligation says an idempotency policy itself denied the call
    pytest.param(["idempotency_repeat"], {}, ("idempotency_repeat", True), id="a repeat: already done"),
    pytest.param(["idempotency_unconfirmed"], {}, ("idempotency_unconfirmed", False), id="unconfirmed"),
    pytest.param(["idempotency_in_flight"], {"in_flight_wait_s": 0}, ("idempotency_in_flight", False),
                 id="in flight, not waited"),
    pytest.param(["High-Risk Tool blocked charge"], {}, None, id="another policy"),
    # an Observe idempotency policy's reason beside a Block policy's deny: no obligation, so no
    # compound step is skipped as "already done", and no in-flight call is waited out
    pytest.param((["Spend cap exceeded", "idempotency_repeat"], {}), {}, None, id="an observed repeat beside a block"),
    pytest.param((["Spend cap exceeded", "idempotency_in_flight"], {}), {}, None,
                 id="an observed in-flight beside a block"),
])
def test_denials(idem_server, denial, opts, repeat):
    from traccia import AgentBlockedError

    idem_server.deny = {"charge": denial}

    @idem_tool("charge", **opts)
    def charge(amount):
        raise AssertionError("a denied tool must not run")

    with _identity():
        with pytest.raises(AgentBlockedError) as info:  # existing handlers still catch it
            charge(10)
    if repeat is None:
        assert not isinstance(info.value, IdempotencyRepeat)
    else:
        assert isinstance(info.value, IdempotencyRepeat)
        assert (info.value.reason, info.value.already_done) == repeat
        assert info.value.reasons == [repeat[0]]
    assert len(idem_server.checks) == 1 and idem_server.settles == []


def test_a_compound_tool_skips_a_step_an_earlier_attempt_finished(idem_server):
    idem_server.deny = {"charge": ["idempotency_repeat"]}  # charged on the first attempt
    ran = []

    @idem_tool("charge")
    def charge(order_id):
        ran.append("charge")

    @idem_tool("send_receipt")
    def send_receipt(order_id):
        ran.append("receipt")

    @idem_tool("checkout")
    def checkout(order_id):
        try:
            charge(order_id)
        except IdempotencyRepeat as exc:
            if not exc.already_done:
                raise
        send_receipt(order_id)
        return "done"

    with _identity():
        assert checkout("o1") == "done"
    assert ran == ["receipt"]
    assert _outcomes(idem_server) == ["commit", "commit"]  # receipt and checkout committed


# --- keying ------------------------------------------------------------------------------------------


def k_refund(order_id, currency="usd", note=None, retries=0):
    return idempotency_key()


_MISSING = object()


def k_notify(to, channel=_MISSING):
    return idempotency_key()


def k_kwargs(**kwargs):
    return idempotency_key()


def k_digest(user_id=None, plan="free", request_id=None):
    return idempotency_key()


def k_tag(label, cls):
    return idempotency_key()


def k_transfer(account, /, **opts):
    return idempotency_key()


def k_hinted(customer: str, page: int = 1):
    return idempotency_key()


def k_plain(customer, page=1):
    return idempotency_key()


def k_issue_refund(order_id, amount, refund_all=None):
    return idempotency_key()


def k_publish(doc, version=1):
    return idempotency_key()


def k_send_email(to, template="welcome", **kwargs):
    return idempotency_key()


_REFUND_TYPES = {"order_id": "string", "amount": "number", "refund_all": "boolean"}
_LOOKUP_TYPES = {"customer": "string", "page": "number"}


@pytest.mark.parametrize("pep", [True, False], ids=["checks on", "checks off"])
@pytest.mark.parametrize("fn, name, opts, args, kwargs, material", [
    # Node has no parameter defaults: an argument left at its default keys as if left out
    pytest.param(k_refund, "refund", {}, ("o1",), {}, {"order_id": "o1"}, id="defaults left out"),
    pytest.param(k_refund, "refund", {}, ("o1",), {"currency": "usd", "retries": 0}, {"order_id": "o1"},
                 id="defaults passed"),
    pytest.param(k_refund, "refund", {}, ("o1",), {"retries": 0.0}, {"order_id": "o1"},
                 id="equal to the default by canonical form"),
    pytest.param(k_refund, "refund", {}, ("o1",), {"currency": "eur"}, {"order_id": "o1", "currency": "eur"},
                 id="a value other than the default"),
    pytest.param(k_refund, "refund", {}, ("o1",), {"retries": False}, {"order_id": "o1", "retries": False},
                 id="False is not 0 in a key"),
    pytest.param(k_notify, "notify", {}, ("x@y.z",), {}, {"to": "x@y.z"}, id="a sentinel default"),
    pytest.param(k_kwargs, "send_email", {"key_fields": ["to"]}, (), {"to": "x@y.z", "body": "hi"}, {"to": "x@y.z"},
                 id="key_fields reach kwargs"),
    pytest.param(k_kwargs, "send_email", {"key_fields": ["to", "too"]}, (), {"to": "x@y.z"}, {"to": "x@y.z"},
                 id="a key field the call lacks"),
    # with both key fields at their default, a changing request id must not key the call
    pytest.param(k_digest, "send_digest", {"key_fields": ["user_id", "plan"]}, (), {"request_id": "r1"}, {},
                 id="key fields at their defaults ignore noise"),
    pytest.param(k_digest, "send_digest", {"key_fields": ["user_id", "plan"]}, (), {"user_id": "u2", "request_id": "r"},
                 {"user_id": "u2"}, id="a key field given"),
    pytest.param(k_tag, "tag", {}, ("x", "gold"), {}, {"label": "x", "cls": "gold"}, id="cls not first is data"),
    pytest.param(k_transfer, "transfer", {}, ("acct_B",), {"memo": "x"}, {"account": "acct_B", "memo": "x"},
                 id="positional-only and kwargs"),
    # Node cannot read types: hints never coerce, only explicit key_types do
    pytest.param(k_hinted, "lookup", {}, (42,), {}, {"customer": 42}, id="type hints do not coerce"),
    pytest.param(k_hinted, "lookup", {}, ("42",), {}, {"customer": "42"}, id="type hints do not coerce a string"),
    pytest.param(k_plain, "lookup", {}, (42,), {}, {"customer": 42}, id="no type hints"),
    pytest.param(k_issue_refund, "issue_refund", {"key_types": _REFUND_TYPES}, ("123", Decimal("10.5"), True), {},
                 {"order_id": "123", "amount": Decimal("10.5"), "refund_all": True}, id="key_types on typed values"),
    pytest.param(k_issue_refund, "issue_refund", {"key_types": _REFUND_TYPES}, (123, "10.50", "true"), {},
                 {"order_id": "123", "amount": Decimal("10.5"), "refund_all": True}, id="key_types on drifting types"),
    pytest.param(k_plain, "lookup", {"key_types": _LOOKUP_TYPES}, (42,), {}, {"customer": "42"},
                 id="key_types to string"),
    pytest.param(k_plain, "lookup", {"key_types": _LOOKUP_TYPES}, ("42",), {"page": "1"}, {"customer": "42"},
                 id="key_types coerce before defaults are left out"),
    pytest.param(k_publish, "publish", {"key_types": {"version": "string"}}, ("d",), {}, {"doc": "d"},
                 id="key_types with a default left out"),
    pytest.param(k_publish, "publish", {"key_types": {"version": "string"}}, ("d", 1), {}, {"doc": "d"},
                 id="key_types leave out a default they convert"),
    pytest.param(k_publish, "publish", {"key_types": {"version": "string"}}, ("d", 2), {}, {"doc": "d", "version": "2"},
                 id="key_types convert a value"),
    pytest.param(k_send_email, "send_email", {"key_fields": ["to", "cc?", "template"]}, ("x@y.z",), {}, {"to": "x@y.z"},
                 id="an optional key field left out"),
    pytest.param(k_send_email, "send_email", {"key_fields": ["to", "cc?", "template"]}, ("x@y.z",), {"cc": "c@y.z"},
                 {"to": "x@y.z", "cc": "c@y.z"}, id="an optional key field given"),
])
def test_keying_through_the_decorator(idem_server, pep, fn, name, opts, args, kwargs, material):
    tool = idem_tool(name, **opts)(fn)
    with _identity(pep=pep):
        key = tool(*args, **kwargs)
    call_key = derive_key("a1", name, material)
    assert key == _keyed(call_key)
    assert [c["context"]["idempotency"]["key"] for c in idem_server.checks] == ([call_key] if pep else [])


def file_ticket_for_test(cls, title):  # a plain function whose first parameter is data
    return idempotency_key()


def _method_calls():
    """Tool calls through methods, keyed by case: self or cls is left out only for a method."""
    class Mailer:
        @idem_tool("send_email")
        def send(self, to):
            return idempotency_key()

    class Repo:
        @classmethod
        @idem_tool("make")
        def make(cls, size):
            return idempotency_key()

    class Desk:
        def __init__(self, owner):
            self.owner = owner

        @idem_tool("assign")
        def assign(self, ticket_id):
            return idempotency_key()

    class Payments:
        pass

    def _refund(self, order_id):
        return idempotency_key()

    def tool_method(fn):
        @functools.wraps(fn)
        def inner(self, *a, **k):
            return fn(self, *a, **k)
        return idem_tool("cancel")(inner)

    def logged(fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            return fn(*a, **k)
        return inner

    # attached after the class: a mixin, Payments.refund = observe(...)(_refund)
    Payments.refund = idem_tool("refund")(_refund)
    Payments.cancel = tool_method(lambda self, order_id: idempotency_key())
    Payments.create = classmethod(idem_tool("create")(lambda cls, name: idempotency_key()))
    Payments.wrapped = logged(idem_tool("refund_wrapped")(_refund))
    bound_tool = idem_tool("refund_bound")(_refund)
    ticket = idem_tool("file_ticket")(file_ticket_for_test)

    @idem_tool("tag")
    def tag(self, label):  # a plain function: "self" is data here
        return idempotency_key()

    payments = Payments()
    payments.bound = types.MethodType(bound_tool, payments)
    return {
        "method": lambda: Mailer().send("x@y.z"),
        "classmethod": lambda: Repo.make(3),
        "plain function with cls first": lambda: ticket("bug", "login fails"),
        "method on one instance": lambda: Desk("ann").assign("t1"),
        "method on another instance": lambda: Desk("bob").assign("t1"),
        "method attached later": lambda: Payments().refund("o1"),
        "wrapped method attached later": lambda: Payments().cancel("o1"),  # signature follows __wrapped__
        "classmethod attached later": lambda: Payments.create("n"),
        "wrapped by another decorator": lambda: Payments().wrapped("o1"),
        "bound with MethodType": lambda: payments.bound("o1"),
        "plain function with self first": lambda: tag("gold", "x"),
    }


@pytest.mark.parametrize("pep", [True, False], ids=["checks on", "checks off"])
@pytest.mark.parametrize("case, name, material", [
    ("method", "send_email", {"to": "x@y.z"}),
    ("classmethod", "make", {"size": 3}),
    ("plain function with cls first", "file_ticket", {"cls": "bug", "title": "login fails"}),
    ("method on one instance", "assign", {"ticket_id": "t1"}),
    ("method on another instance", "assign", {"ticket_id": "t1"}),
    ("method attached later", "refund", {"order_id": "o1"}),
    ("wrapped method attached later", "cancel", {"order_id": "o1"}),
    ("classmethod attached later", "create", {"name": "n"}),
    ("wrapped by another decorator", "refund_wrapped", {"order_id": "o1"}),
    ("bound with MethodType", "refund_bound", {"order_id": "o1"}),
    ("plain function with self first", "tag", {"self": "gold", "label": "x"}),
])
def test_keying_of_methods(idem_server, pep, case, name, material):
    call = _method_calls()[case]
    with _identity(pep=pep):
        key = call()
    call_key = derive_key("a1", name, material)
    assert key == _keyed(call_key)
    if pep:
        [check] = idem_server.checks
        assert check["context"]["idempotency"]["key"] == call_key
        assert check["context"]["input"] == material  # the check sees the same arguments
    else:
        assert idem_server.checks == []


def test_key_extra(idem_server):
    class Payments:
        def __init__(self, account_id):
            self.account_id = account_id

        @idem_tool("refund", key_extra=lambda self, order_id: {"account": self.account_id})
        def refund(self, order_id):
            return idempotency_key()

    tenant = contextvars.ContextVar("tenant")

    @idem_tool("send_reminder", key_extra=lambda *args, **kwargs: {"tenant": tenant.get()})
    def send_reminder(template="welcome"):
        return idempotency_key()

    @idem_tool("refund_plain", key_extra=lambda order_id: None)
    def refund_plain(order_id):
        return idempotency_key()

    with _identity(pep=False):
        # inputs that are not arguments: the account on self
        a, b = Payments("acct_1").refund("o1"), Payments("acct_2").refund("o1")
        # ... or the current tenant
        tenant.set("t_1")
        first = send_reminder()
        tenant.set("t_2")
        second = send_reminder()
        # None leaves the key unchanged
        plain = refund_plain("o1")
    assert a == _keyed(derive_key("a1", "refund", {"order_id": "o1"}, extra={"account": "acct_1"})) != b
    assert first == _keyed(derive_key("a1", "send_reminder", {}, extra={"tenant": "t_1"})) != second
    assert plain == _keyed(derive_key("a1", "refund_plain", {"order_id": "o1"}))


def _reason_inside():
    """What idempotency_key() says inside an unkeyable call."""
    try:
        idempotency_key()
    except Unkeyable as exc:
        return str(exc)
    raise AssertionError("expected the call to be unkeyable")


def u_client(client, to):
    return _reason_inside()


def u_card(card):
    return _reason_inside()


def u_refund(order_id):
    return _reason_inside()


def u_transfer(account, /, **opts):
    return _reason_inside()


def u_notify(*args, **kw):
    return _reason_inside()


def u_amount(order_id, amount):
    return _reason_inside()


class _Card:
    def model_dump(self, mode="python"):
        raise RuntimeError("pan 4111111111111111")


def _raising_extra(order_id):
    raise KeyError("secret-tenant-id")


_KEY_EXTRA_CALLS = []


def _recording_extra(*args, **kwargs):
    _KEY_EXTRA_CALLS.append(1)


@pytest.mark.parametrize("pep", [True, False], ids=["checks on", "checks off"])
@pytest.mark.parametrize("fn, opts, args, kwargs, reason, secret", [
    pytest.param(u_client, {}, (object(), "x@y.z"), {}, "client is object", None, id="an object argument"),
    # a raising model_dump() or key_extra may hold raw values: only the type is sent
    pytest.param(u_card, {}, (_Card(),), {}, "RuntimeError while keying arguments", "4111", id="a raising model_dump"),
    pytest.param(u_refund, {"key_extra": _raising_extra}, ("o1",), {}, "KeyError while computing key_extra", "secret",
                 id="a raising key_extra"),
    # a **kwargs entry would overwrite the positional-only or *args value: two calls could share a key;
    # key_extra is not called for a call that cannot be keyed
    pytest.param(u_transfer, {"key_extra": _recording_extra}, ("acct_A",), {"account": "memo"},
                 "**opts entry 'account' has the same name as a parameter", None, id="kwargs entry named like a parameter"),
    pytest.param(u_notify, {}, ("carol",), {"args": ["alice", "bob"]}, "**kw entry 'args' has the same name as a parameter",
                 None, id="kwargs entry named like args"),
    # key_types never fails the call, even on a value it cannot convert
    pytest.param(u_amount, {"key_types": {"amount": "string"}}, ("o1", Decimal("NaN")), {},
                 "amount is a non-finite Decimal", None, id="key_types on a value they cannot convert"),
])
def test_unkeyable_calls_still_run(idem_server, pep, fn, opts, args, kwargs, reason, secret):
    _KEY_EXTRA_CALLS.clear()
    tool = idem_tool(fn.__name__, **opts)(fn)
    with _identity(pep=pep):
        assert tool(*args, **kwargs) == reason  # it ran; the key accessor says why it has no key
    sent = [c["context"]["idempotency"] for c in idem_server.checks]
    assert sent == ([{"unkeyable": reason}] if pep else [])
    assert all(c["action"]["idempotent"] is True for c in idem_server.checks)
    assert idem_server.settles == []  # nothing was reserved
    assert _KEY_EXTRA_CALLS == []
    if secret:
        assert secret not in json.dumps(sent)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("pep, downstream, passed", [
    pytest.param(True, DOWNSTREAM, "per attempt", id="the server's per-attempt key"),
    pytest.param(True, None, "keyed", id="no per-attempt key"),
    pytest.param(True, "", "keyed", id="an empty per-attempt key"),
    pytest.param(True, "idk1_short", "keyed", id="a short per-attempt key"),
    pytest.param(True, "IDK1_" + "b" * 64, "keyed", id="an upper-case per-attempt key"),
    pytest.param(True, 123, "keyed", id="a per-attempt key that is not text"),
    # checks off: the call key keyed by the API key. A plain hash of guessable arguments (an order
    # id, a phone number) could be reversed by whoever receives it.
    pytest.param(False, DOWNSTREAM, "keyed", id="checks off"),
])
def test_downstream_key(idem_server, is_async, pep, downstream, passed):
    idem_server.downstream_key = downstream

    def body():
        return idempotency_key(), _current_span()

    if is_async:
        @idem_tool("charge")
        async def charge(amount):
            return body()
    else:
        @idem_tool("charge")
        def charge(amount):
            return body()

    async def call_async():
        with _identity(pep=pep):
            return await charge(10)

    def call():
        with _identity(pep=pep):
            return charge(10)

    key, span = _run(is_async, call_async if is_async else call)
    call_key = derive_key("a1", "charge", {"amount": 10})
    expected = DOWNSTREAM if passed == "per attempt" else _keyed(call_key)
    assert key == expected != call_key
    assert span.attributes["traccia.idempotency.downstream_key"] == expected
    assert "traccia.idempotency.key" not in span.attributes  # the call key never goes on a span
    if pep:
        [settle] = idem_server.settles
        assert settle["idempotency"]["key"] == call_key  # settles still name the call key
    else:
        assert idem_server.checks == [] and idem_server.settles == []


def test_without_an_api_key_the_call_key_is_passed_downstream(monkeypatch):
    from traccia.governance import tool_call as tool_call_mod
    from traccia.governance.idempotency import IdempotencySpec

    monkeypatch.setattr("traccia.client_config.resolve_credentials", lambda: (None, None))

    @idem_tool("refund")
    def refund(order_id):
        return idempotency_key()

    with _identity(pep=False):
        assert refund("o1") == derive_key("a1", "refund", {"order_id": "o1"})
        # No offline key to sign, so none is sent. Low-level: with no API key a check cannot be sent
        # at all, so the field it would carry is read from _tool_key().
        _call_key, wire = tool_call_mod._tool_key(IdempotencySpec(name="refund"), {"order_id": "o1"})
    assert set(wire) == {"key"}


def test_idempotency_scope(idem_server):
    @idem_tool("issue_refund", idempotency_scope="workspace")
    def issue_refund(order_id):
        return idempotency_key()

    @idem_tool("cancel")
    def cancel(order_id):
        return None

    keys = []
    for agent in ("billing-agent", "support-agent"):
        with runtime_config.run_identity(pep_enabled=True, agent_id=agent):
            keys.append(issue_refund("o1"))
    # a workspace-scoped tool has one key for every agent, and says so
    assert keys[0] == keys[1] == _keyed(derive_key(None, "issue_refund", {"order_id": "o1"}))
    assert [c["action"]["idempotency_scope"] for c in idem_server.checks] == ["workspace", "workspace"]
    assert [c["agent_id"] for c in idem_server.checks] == ["billing-agent", "support-agent"]
    # agent scope is the default, and not sent
    with _identity():
        cancel("o1")
    assert "idempotency_scope" not in idem_server.checks[-1]["action"]


# --- definition validation ---------------------------------------------------------------------------


def d_charge(amount):
    return None


def d_send(to, body):
    return None


def d_send_kwargs(to, **kwargs):
    return None


def d_send_cc(to, cc=None):
    return None


def d_gen(x):
    yield x


async def d_agen(x):
    yield x


async def _async_extra(order_id):
    return {"tenant": "t"}


class _AsyncExtra:
    async def __call__(self, order_id):
        return {"tenant": "t"}


_NAME_ERROR = r"idempotent tools need a name: pass @observe\(name='...'\)"


@pytest.mark.parametrize("fn, opts, error, match", [
    pytest.param(d_charge, {"as_type": "span", "idempotent": True}, ValueError, r"idempotent=True requires as_type='tool'",
                 id="not a tool"),
    pytest.param(d_gen, {"idempotent": True}, TypeError, "idempotent tools cannot be generators in v1", id="generator"),
    pytest.param(d_agen, {"idempotent": True}, TypeError, "idempotent tools cannot be generators in v1",
                 id="async generator"),
    # an idempotent tool needs an explicit name: stripe_tools.refund and paypal_tools.refund would share
    # a key, and renaming or moving the function would change every key
    pytest.param(d_charge, {"idempotent": True}, ValueError, _NAME_ERROR, id="no name"),
    pytest.param(d_charge, {"name": "stripe_refund", "idempotent": True}, None, None, id="a name"),
    pytest.param(d_charge, {}, None, None, id="not idempotent: the name may come from the function"),
    pytest.param(d_send, {"key_fields": ["to"]}, ValueError, "key_fields requires idempotent=True",
                 id="key_fields without idempotent"),
    pytest.param(d_send, {"idempotent": True, "key_fields": []}, ValueError, "key_fields must name at least one parameter",
                 id="empty key_fields list"),
    pytest.param(d_send, {"idempotent": True, "key_fields": ()}, ValueError, "key_fields must name at least one parameter",
                 id="empty key_fields tuple"),
    pytest.param(d_send, {"name": "send", "idempotent": True, "key_fields": ["to", 3]}, TypeError,
                 "key_fields must be parameter names", id="a key field that is not a string"),
    pytest.param(d_send_kwargs, {"name": "send", "idempotent": True, "key_fields": ["to", 3]}, TypeError,
                 "key_fields must be parameter names", id="a key field that is not a string, with kwargs"),
    pytest.param(d_send, {"idempotent": True, "key_fields": ["to", "cc"]}, ValueError, "key_fields: unknown parameter 'cc'",
                 id="an unknown key field"),
    pytest.param(d_send_kwargs, {"name": "send", "idempotent": True, "key_fields": ["to", "cc"]}, None, None,
                 id="a key field kwargs may carry"),
    pytest.param(d_send_cc, {"name": "send", "idempotent": True, "key_fields": ["to", "cc?"]}, None, None,
                 id="an optional key field"),
    pytest.param(d_send_cc, {"name": "send", "idempotent": True, "key_fields": ["to", "bcc?"]}, ValueError,
                 "key_fields: unknown parameter 'bcc'", id="an unknown optional key field"),
    pytest.param(d_charge, {"unconfirmed_on": [TimeoutError]}, ValueError, "require idempotent=True",
                 id="unconfirmed_on without idempotent"),
    pytest.param(d_charge, {"release_on": [ValueError]}, ValueError, "require idempotent=True",
                 id="release_on without idempotent"),
    pytest.param(d_charge, {"idempotent": True, "release_on": ["ValueError"]}, TypeError, "release_on must list",
                 id="release_on names"),
    pytest.param(d_charge, {"release_unconfirmed": True}, ValueError, "require idempotent=True",
                 id="release_unconfirmed without idempotent"),
    pytest.param(d_charge, {"idempotent": True, "unconfirmed_on": ["TimeoutError"]}, TypeError, "exception classes",
                 id="unconfirmed_on names"),
    pytest.param(d_charge, {"idempotent": True, "unconfirmed_on": [OSError], "release_unconfirmed": True}, ValueError,
                 "not both", id="unconfirmed_on and release_unconfirmed"),
    pytest.param(d_charge, {"idempotency_scope": "workspace"}, ValueError, "idempotency_scope requires idempotent=True",
                 id="scope without idempotent"),
    pytest.param(d_charge, {"idempotent": True, "idempotency_scope": "org"}, ValueError,
                 "idempotency_scope must be 'agent' or 'workspace'", id="an unknown scope"),
    pytest.param(d_charge, {"name": "refund", "idempotent": True, "idempotency_scope": "agent"}, None, None,
                 id="agent scope"),
    pytest.param(d_charge, {"name": "refund", "key_extra": lambda *a: {}}, ValueError, "key_extra requires idempotent=True",
                 id="key_extra without idempotent"),
    pytest.param(d_charge, {"name": "refund", "idempotent": True, "key_extra": {"tenant": "t"}}, TypeError,
                 "key_extra must be callable", id="key_extra not callable"),
    # an async key_extra gives a coroutine, never its values: rejected up front
    pytest.param(d_charge, {"name": "refund", "idempotent": True, "key_extra": _async_extra}, TypeError,
                 "key_extra must return its values directly", id="async key_extra"),
    pytest.param(d_charge, {"name": "refund", "idempotent": True, "key_extra": functools.partial(_async_extra)},
                 TypeError, "key_extra must return its values directly", id="partial of an async key_extra"),
    pytest.param(d_charge, {"name": "refund", "idempotent": True, "key_extra": _AsyncExtra()}, TypeError,
                 "key_extra must return its values directly", id="async __call__ key_extra"),
    pytest.param(d_charge, {"name": "charge", "in_flight_wait_s": 5}, ValueError, "in_flight_wait_s requires idempotent=True",
                 id="in_flight_wait_s without idempotent"),
    pytest.param(d_charge, {"name": "charge", "idempotent": True, "in_flight_wait_s": -1}, ValueError,
                 "in_flight_wait_s must be 0 or more seconds", id="negative in_flight_wait_s"),
    pytest.param(d_charge, {"name": "lookup", "key_types": {"amount": "string"}}, ValueError,
                 "key_types requires idempotent=True", id="key_types without idempotent"),
    pytest.param(d_charge, {"name": "lookup", "idempotent": True, "key_types": {"amount": "text"}}, ValueError,
                 "key_types: 'amount' must be 'string', 'number' or 'boolean'", id="an unknown key type"),
    pytest.param(d_charge, {"name": "lookup", "idempotent": True, "key_types": {"amt": "string"}}, ValueError,
                 "key_types: unknown parameter 'amt'", id="key_types for an unknown parameter"),
])
def test_definition_validation(fn, opts, error, match):
    """The idempotency options are checked when the tool is defined, not when it is called."""
    opts = {"as_type": "tool", **opts}
    if error is None:
        observe(**opts)(fn)
        return
    with pytest.raises(error, match=match):
        observe(**opts)(fn)


# --- a sync tool that hands back an awaitable --------------------------------------------------------


class _CompiledCoroutine(collections.abc.Coroutine):
    """A coroutine that is not a native one (as Cython or mypyc compile them): only send, throw,
    close and __await__, no cr_running or cr_frame."""

    def __init__(self, inner):
        self._inner = inner

    def send(self, value):
        return self._inner.send(value)

    def throw(self, *args):
        return self._inner.throw(*args)

    def close(self):
        return self._inner.close()

    def __await__(self):
        return self


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited")
@pytest.mark.parametrize("kind, fate, runs, outcome", [
    pytest.param("plain value", "returned", True, "commit", id="a plain value commits at once"),
    pytest.param("coroutine", "completes", True, "commit", id="coroutine completes"),
    pytest.param("coroutine", "fails", True, "release", id="coroutine fails"),
    pytest.param("coroutine", "times out", True, "unconfirmed", id="coroutine times out"),
    # the work never ran: abandon, so a retry does not wait out the lease
    pytest.param("coroutine", "cancelled before it starts", False, "abandon", id="coroutine cancelled before it starts"),
    pytest.param("coroutine", "dropped", False, "abandon", id="coroutine never awaited"),
    # a returned task runs by itself: dropping it never abandons it
    pytest.param("task", "completes", True, "commit", id="task completes, dropped by the caller"),
    pytest.param("task", "times out", True, "unconfirmed", id="task times out"),
    # whether a task's coroutine ran is read from its frame's RESUME (3.11+); before that it is
    # not known, so the cancel is held like any other
    pytest.param("task", "cancelled before it starts", False, "abandon" if sys.version_info >= (3, 11) else "unconfirmed",
                 id="task cancelled before it starts"),
    pytest.param("task", "cancelled while running", True, "unconfirmed", id="task cancelled while running"),
    # whether a non-native coroutine ran is not known: a cancel is held, never abandoned
    pytest.param("compiled task", "completes", True, "commit", id="task of a non-native coroutine completes"),
    pytest.param("compiled task", "cancelled before it starts", False, "unconfirmed",
                 id="task of a non-native coroutine cancelled"),
    pytest.param("future", "completes", False, "commit", id="future completes"),
    pytest.param("future", "dropped", False, "abandon", id="future dropped unfinished"),
    pytest.param("concurrent future", "completes", True, "commit", id="concurrent future completes"),
    pytest.param("concurrent future", "fails", True, "release", id="concurrent future fails"),
    # cancel() can succeed while a run_coroutine_threadsafe coroutine is running: held
    pytest.param("concurrent future", "cancelled", False, "unconfirmed", id="concurrent future cancelled"),
])
def test_deferred_awaitables(idem_server, kind, fate, runs, outcome):
    """A sync tool that returns an awaitable settles when the work ends, not when it returns."""
    from traccia.governance.pep import flush_pending_settles

    ran, seen = [], {}
    errors = {"fails": DefiniteFailure("declined"), "times out": TimeoutError("read timed out")}
    gate = threading.Event()
    pool = ThreadPoolExecutor(1)

    async def work(order_id):
        ran.append(order_id)
        try:
            seen["key"] = idempotency_key()  # bound while the work runs
        except RuntimeError:
            seen["key"] = None
        seen["settles while running"] = len(idem_server.settles)
        if fate == "cancelled while running":
            await asyncio.sleep(10)
        await asyncio.sleep(0)
        if fate in errors:
            raise errors[fate]
        return "ok"

    def work_in_thread(order_id):
        gate.wait(5)
        ran.append(order_id)
        if fate in errors:
            raise errors[fate]
        return "ok"

    @idem_tool("charge")
    def charge(order_id):
        if kind == "plain value":
            ran.append(order_id)
            return "ok"
        if kind == "coroutine":
            return work(order_id)
        if kind == "task":
            return asyncio.get_running_loop().create_task(work(order_id))
        if kind == "compiled task":
            return asyncio.get_running_loop().create_task(_CompiledCoroutine(work(order_id)))
        if kind == "future":
            return asyncio.get_running_loop().create_future()
        if fate == "cancelled":
            return Future()
        return pool.submit(work_in_thread, order_id)

    async def main():
        with _identity():
            returned = charge("o1")
            seen["settles at return"] = len(idem_server.settles)
            if kind == "plain value":
                assert returned == "ok"
            elif fate == "dropped":
                del returned
                gc.collect()
            elif kind == "future":
                returned.set_result("ok")
                await asyncio.sleep(0)
            elif kind == "concurrent future":
                if fate == "cancelled":
                    returned.cancel()
                else:
                    gate.set()
                    if fate == "fails":
                        assert isinstance(returned.exception(5), DefiniteFailure)
                    else:
                        assert returned.result(5) == "ok"
            elif fate == "completes" and kind == "task":
                del returned  # dropping a task never abandons it
                gc.collect()
                for _ in range(5000):  # until the task has finished
                    if len(asyncio.all_tasks()) == 1:
                        break
                    await asyncio.sleep(0.001)
            elif fate == "completes":
                assert await returned == "ok"
            elif fate.startswith("cancelled"):
                waiter = asyncio.ensure_future(returned) if kind == "coroutine" else returned
                if fate == "cancelled while running":
                    while not ran:
                        await asyncio.sleep(0)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            else:
                with pytest.raises(type(errors[fate])):
                    await returned

    try:
        asyncio.run(main())
    finally:
        gate.set()
        pool.shutdown(wait=True)
    gc.collect()
    flush_pending_settles()
    assert seen["settles at return"] == (1 if kind == "plain value" else 0)
    assert ran == (["o1"] if runs else [])
    if runs and kind in ("coroutine", "task", "compiled task"):
        assert seen["key"] == _keyed(derive_key("a1", "charge", {"order_id": "o1"}))
        assert seen["settles while running"] == 0
    assert _outcomes(idem_server) == [outcome]


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited")
def test_a_never_started_deferred_call_only_hands_its_settle_off(idem_server):
    # The finalizer runs inside the garbage collector, maybe while this thread holds
    # ThreadPoolExecutor.submit()'s lock: it must only queue the work. Low-level: the hand-off is
    # intercepted because no outcome tells "settled inside the collector" from "handed off".
    from traccia.governance import pep

    handed = []

    async def work():
        return "ok"

    @idem_tool("book")
    def book(room):
        return work()

    with patch.object(pep, "hand_off", side_effect=handed.append), \
         patch.object(pep, "start_handoff_worker"):
        with _identity():
            pending = book("101")
        del pending
        gc.collect()
    assert len(handed) == 1 and idem_server.settles == []  # nothing settled inside the collector
    handed[0]()
    pep.flush_pending_settles()
    assert _outcomes(idem_server) == ["abandon"]


def test_an_error_tracking_a_returned_awaitable_holds_the_key(idem_server, monkeypatch):
    # Whatever goes wrong in the SDK's own bookkeeping after the tool returned must never release the
    # key (the work may be running). Low-level: a bookkeeping bug can only be injected by patching.
    from traccia.governance import pep
    from traccia.governance.pep import flush_pending_settles

    def broken(self, awaitable):
        raise RuntimeError("bookkeeping bug")

    monkeypatch.setattr(pep.ToolCallSettle, "_defer", broken)
    charges = []

    async def work(order_id):
        charges.append(order_id)
        return "ok"

    @idem_tool("charge")
    def charge(order_id):
        return asyncio.get_running_loop().create_task(work(order_id))

    async def main():
        with _identity():
            return await charge("o1")

    assert asyncio.run(main()) == "ok"  # the caller still gets its awaitable
    flush_pending_settles()
    assert charges == ["o1"]
    assert _outcomes(idem_server) == ["unconfirmed"]


# --- a call denied as in flight waits and checks again ----------------------------------------------


def _run_as(runner, sync_tool, async_tool, *args):
    if runner == "sync":
        with _identity():
            return sync_tool(*args)

    async def main():
        with _identity():
            if runner == "async":
                return await async_tool(*args)
            if runner == "sync on the event loop thread":
                return sync_tool(*args)
            return await asyncio.to_thread(sync_tool, *args)  # off the loop: waiting is harmless
    return asyncio.run(main())


@pytest.mark.parametrize("script, wait, runner, result, slept, rechecks", [
    pytest.param([IN_FLIGHT, IN_FLIGHT, None], None, "sync", "charged", [0.5, 1.0], [None, True, True],
                 id="waits and runs once the first call failed"),
    pytest.param([IN_FLIGHT, IN_FLIGHT, None], None, "async", "charged", [0.5, 1.0], [None, True, True],
                 id="async waits too"),
    pytest.param([IN_FLIGHT, REPEAT], None, "sync", ("idempotency_repeat", True), [0.5], [None, True],
                 id="ends as a repeat: already done"),
    pytest.param([IN_FLIGHT] * 50, 5, "sync", ("idempotency_in_flight", False), [0.5, 1.0, 2.0],
                 [None, True, True, True], id="gives up after its limit"),
    # sleeping on the loop would freeze it: raise at once
    pytest.param([IN_FLIGHT, None], None, "sync on the event loop thread", ("idempotency_in_flight", False), [], [None],
                 id="a sync tool on the event loop thread does not wait"),
    pytest.param([IN_FLIGHT, None], None, "sync on a worker thread", "charged", [0.5], [None, True],
                 id="a sync tool on a worker thread of an async app waits"),
])
def test_in_flight(idem_server, script, wait, runner, result, slept, rechecks):
    idem_server.script = list(script)
    opts = {} if wait is None else {"in_flight_wait_s": wait}
    ran = []

    @idem_tool("charge", **opts)
    def charge(amount):
        ran.append(amount)
        return "charged"

    @idem_tool("charge", **opts)
    async def charge_async(amount):
        ran.append(amount)
        return "charged"

    if isinstance(result, tuple):
        with pytest.raises(IdempotencyRepeat) as info:
            _run_as(runner, charge, charge_async, 10)
        assert (info.value.reason, info.value.already_done) == result
        assert ran == [] and idem_server.settles == []
    else:
        assert _run_as(runner, charge, charge_async, 10) == result
        assert ran == [10] and _outcomes(idem_server) == ["commit"]
    assert idem_server.slept == slept
    # rechecks are flagged, so the server neither counts nor stores them
    assert [c["context"]["idempotency"].get("recheck") for c in idem_server.checks] == rechecks


@pytest.mark.parametrize("script, pep, is_async, same_args, result, checks, outcomes", [
    # under Block the inner call is still checked, but never waits on its own reservation; the
    # outer call is held, as its body may have done work before the inner call was denied
    pytest.param([None, IN_FLIGHT], True, False, True, ("idempotency_in_flight", False), 2, ["unconfirmed"],
                 id="under Block"),
    pytest.param([None, IN_FLIGHT], True, True, True, ("idempotency_in_flight", False), 2, ["unconfirmed"],
                 id="under Block, async"),
    # with Observe, Warn or no idempotency policy the server allows it: no local block
    pytest.param([], True, False, True, "inner", 2, ["commit", "commit"], id="without a Block policy"),
    pytest.param([], True, False, False, "inner", 2, ["commit", "commit"], id="with other arguments"),
    pytest.param([], False, False, True, "inner", 0, [], id="with checks off"),
])
def test_a_tool_calling_itself(idem_server, script, pep, is_async, same_args, result, checks, outcomes):
    idem_server.script = list(script)

    def inner_order(order_id):
        return order_id if same_args else order_id + "-2"

    @idem_tool("refund", key_fields=["order_id"], in_flight_wait_s=30)
    def refund(order_id, depth=0):
        return refund(inner_order(order_id), depth=1) if depth == 0 else "inner"

    @idem_tool("refund", key_fields=["order_id"], in_flight_wait_s=30)
    async def refund_async(order_id, depth=0):
        return await refund_async(inner_order(order_id), depth=1) if depth == 0 else "inner"

    async def call_async():
        with _identity(pep=pep):
            return await refund_async("o1")

    def call():
        with _identity(pep=pep):
            return refund("o1")

    if isinstance(result, tuple):
        with pytest.raises(IdempotencyRepeat) as info:
            _run(is_async, call_async if is_async else call)
        assert (info.value.reason, info.value.already_done) == result
    else:
        assert _run(is_async, call_async if is_async else call) == result
    assert idem_server.slept == []
    assert len(idem_server.checks) == checks
    assert _outcomes(idem_server) == outcomes


def test_a_call_a_finished_tool_left_behind_is_not_a_re_entry(idem_server):
    # A task or thread the tool spawned copies its context; once the tool has returned, a call from
    # there is a new call that may wait, not a re-entry that may not.
    idem_server.script = [None, IN_FLIGHT]  # the first call reserves; then someone else holds it
    left_behind = []

    @idem_tool("refund", key_fields=["order_id"])
    def refund(order_id, again=False):
        if not again:
            left_behind.append(contextvars.copy_context())
        return "ran"

    with _identity():
        assert refund("o1") == "ran"
        assert left_behind[0].run(refund, "o1", True) == "ran"  # waited once, then ran
    assert idem_server.slept == [0.5]


def test_in_flight_rechecks_are_jittered():
    from traccia.governance import tool_call as tool_call_mod

    state = random.getstate()
    random.seed(308)
    try:
        pauses = [tool_call_mod._jittered(2.0) for _ in range(200)]
    finally:
        random.setstate(state)
    assert all(1.0 <= p <= 3.0 for p in pauses)
    assert max(pauses) - min(pauses) > 1.0  # spread out, not in lockstep
