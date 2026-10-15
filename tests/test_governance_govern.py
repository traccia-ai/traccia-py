"""@govern inherits agent_id from init; decorator arg is an override only."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traccia import observe, runtime_config
from traccia.governance.govern import govern
from traccia.governance.policy import AgentBlockedError


def test_govern_inherits_agent_id_from_init():
    runtime_config.set_agent_id("from-init")
    try:
        with patch("traccia.governance.govern.check_agent_status") as check:
            @govern(fail_open=False, name="run")
            def run() -> str:
                return "ok"

            assert run() == "ok"
            check.assert_called_once_with("from-init", fail_open=False)
    finally:
        runtime_config.set_agent_id(None)


def test_govern_explicit_agent_id_overrides_init():
    runtime_config.set_agent_id("from-init")
    try:
        with patch("traccia.governance.govern.check_agent_status") as check:
            @govern(agent_id="override", fail_open=True, name="run")
            def run() -> str:
                return "ok"

            assert run() == "ok"
            check.assert_called_once_with("override", fail_open=True)
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
        check.assert_called_once_with("from-env", fail_open=True)


def test_govern_warns_when_identity_missing(monkeypatch, caplog):
    runtime_config.set_agent_id(None)
    monkeypatch.delenv("TRACCIA_AGENT_ID", raising=False)
    with patch("traccia.governance.govern.check_agent_status") as check:
        @govern(name="run")
        def run() -> str:
            return "ok"

        assert run() == "ok"
        check.assert_not_called()
    assert "Skipping policy check" in caplog.text


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


def test_govern_fail_closed_blocks_tool_call_during_policy_outage():
    run, ran, session = _outage_tool_run(fail_open=False, name="run")
    with patch("traccia.governance.govern.check_agent_status"), \
            patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")), \
            patch("traccia.governance.pep._http_session", session):
        with pytest.raises(AgentBlockedError):
            run()
    assert ran == []
    assert runtime_config.pep_fail_open() is True


def test_govern_fail_open_allows_tool_call_during_policy_outage():
    run, ran, session = _outage_tool_run(name="run")
    with patch("traccia.governance.govern.check_agent_status"), \
            patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")), \
            patch("traccia.governance.pep._http_session", session):
        assert run() == {"refunded": 500}
    assert ran == [500]


def test_bare_govern_defaults_to_fail_open(monkeypatch):
    monkeypatch.setenv("TRACCIA_AGENT_ID", "from-env")
    with patch("traccia.governance.govern.check_agent_status") as check:
        @govern
        def run() -> bool:
            return runtime_config.pep_fail_open()

        assert run() is True
        assert check.call_args.kwargs == {"fail_open": True}
