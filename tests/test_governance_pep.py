from unittest.mock import MagicMock, patch

import pytest

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


def test_check_http_error_allows():
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=503, text="down")
                out = check_policy(action={"type": "llm_call"})
                assert out["effect"] == "allow"
                assert out["reasons"] == ["check_http_error"]


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
    pep.remember_tool_result("search", {"q": "x"})
    pep.remember_tool_result("get_refund_policy", {"as_of": "2026-09-24T11:40:00Z", "text": "policy"})
    decision = {"id": "dec-fresh", "effect": "allow", "would_have": False, "reasons": []}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
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
    decision = {"id": "dec-v", "effect": "allow", "obligations": {}}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                enforce_llm_call({"model": "gemini-2.0-flash-001"}, vendor="google_gemini")
                body = session.post.call_args.kwargs["json"]
                assert body["context"]["vendor"] == "google_gemini"

