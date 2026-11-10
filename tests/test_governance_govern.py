"""@govern inherits agent_id from init; decorator arg is an override only."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traccia import observe, runtime_config
from traccia.governance.govern import govern
from traccia.governance.policy import AgentBlockedError, GovernanceConfigError


def test_govern_inherits_agent_id_from_init():
    runtime_config.set_agent_id("from-init")
    try:
        with patch("traccia.governance.govern.check_agent_status") as check:
            @govern(policy_fail_mode="closed", name="run")
            def run() -> str:
                return "ok"

            assert run() == "ok"
            check.assert_called_once_with("from-init", policy_fail_mode="closed")
    finally:
        runtime_config.set_agent_id(None)


def test_govern_explicit_agent_id_overrides_init():
    runtime_config.set_agent_id("from-init")
    try:
        with patch("traccia.governance.govern.check_agent_status") as check:
            @govern(agent_id="override", name="run")
            def run() -> str:
                return "ok"

            assert run() == "ok"
            check.assert_called_once_with("override", policy_fail_mode=None)
    finally:
        runtime_config.set_agent_id(None)


def test_govern_falls_back_to_env(monkeypatch):
    runtime_config.set_agent_id(None)
    monkeypatch.setenv("TRACCIA_AGENT_ID", "from-env")
    with patch("traccia.governance.govern.check_agent_status") as check:
        @govern(name="run")
        def run() -> str:
            return "ok"

        assert run() == "ok"
        check.assert_called_once_with("from-env", policy_fail_mode=None)


def test_govern_without_an_agent_id_is_a_configuration_error(monkeypatch):
    # TRC-342: a governed run without identity is never silently allowed.
    runtime_config.set_agent_id(None)
    monkeypatch.delenv("TRACCIA_AGENT_ID", raising=False)
    ran = []
    with patch("traccia.governance.govern.check_agent_status") as check:
        @govern(policy_fail_mode="open", name="run")
        def run() -> str:
            ran.append(1)
            return "ok"

        with pytest.raises(GovernanceConfigError):
            run()
        check.assert_not_called()
    assert ran == []


def _outage_tool_run(**govern_kwargs):
    ran = []

    @observe(as_type="tool")
    def refund(amount: int):
        ran.append(amount)
        return {"refunded": amount}

    @govern(agent_id="support", **govern_kwargs)
    def run():
        return refund(500)

    session = MagicMock()
    session.post.return_value = MagicMock(status_code=503, text="down")
    return run, ran, session


def test_govern_blocks_tool_call_during_policy_outage_by_default():
    run, ran, session = _outage_tool_run(name="run")
    with patch("traccia.governance.govern.check_agent_status"), \
            patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")), \
            patch("traccia.governance.pep._http_session", session):
        with pytest.raises(AgentBlockedError):
            run()
    assert ran == []
    assert runtime_config.pep_fail_mode() is None


def test_govern_open_override_allows_tool_call_during_policy_outage():
    run, ran, session = _outage_tool_run(policy_fail_mode="open", name="run")
    with patch("traccia.governance.govern.check_agent_status"), \
            patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")), \
            patch("traccia.governance.pep._http_session", session):
        assert run() == {"refunded": 500}
    assert ran == [500]


def test_bare_govern_follows_the_traccia_setting(monkeypatch):
    monkeypatch.setenv("TRACCIA_AGENT_ID", "from-env")
    with patch("traccia.governance.govern.check_agent_status") as check:
        @govern
        def run():
            return runtime_config.pep_fail_mode()

        assert run() is None                            # no override for the run
        assert check.call_args.kwargs == {"policy_fail_mode": None}
