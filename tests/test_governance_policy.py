"""Tests for runtime governance policy checks."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from traccia.governance.config import gov_config
from traccia.governance import policy as policy_module
from traccia.governance.policy import (
    AgentBlockedError,
    AgentStatusCache,
    _InFlightCoalescer,
    _derive_base_url,
    _status_url,
    check_agent_status,
)


@pytest.fixture(autouse=True)
def reset_policy_state():
    policy_module._status_cache = AgentStatusCache()
    policy_module._inflight = _InFlightCoalescer()
    gov_config.status_check_endpoint = None
    gov_config.post_block_endpoint = None
    gov_config.status_cache_ttl_seconds = 60
    yield


class _FakeTracing:
    def __init__(self, api_key="key", endpoint="https://api.traccia.ai/v1/traces"):
        self.api_key = api_key
        self.endpoint = endpoint


class _FakeConfig:
    def __init__(self, api_key="key", endpoint="https://api.traccia.ai/v1/traces"):
        self.tracing = _FakeTracing(api_key=api_key, endpoint=endpoint)


def test_derive_base_url():
    assert _derive_base_url("https://api.traccia.ai/v1/traces") == "https://api.traccia.ai"


def test_default_status_url():
    gov_config.status_check_endpoint = None
    assert (
        _status_url("https://api.traccia.ai", "agent-1")
        == "https://api.traccia.ai/api/v1/agents/agent-1/status"
    )


def test_status_url_override():
    gov_config.status_check_endpoint = "https://custom.example/agents/{agent_id}/status"
    assert _status_url("https://api.traccia.ai", "agent-1") == "https://custom.example/agents/agent-1/status"
    gov_config.status_check_endpoint = None


def test_cache_hit():
    cache = AgentStatusCache(ttl_seconds=60)
    cache.set("a1", "allowed")
    assert cache.get("a1") == ("allowed", None)


def test_cache_expiry():
    cache = AgentStatusCache(ttl_seconds=0)
    cache.set("a1", "allowed")
    time.sleep(0.01)
    assert cache.get("a1") is None


@patch("traccia.governance.policy._http_session")
@patch("traccia.governance.policy.load_config")
def test_check_agent_status_allowed(mock_load_config, mock_session):
    gov_config.status_check_endpoint = None
    mock_load_config.return_value = _FakeConfig()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "allowed"}
    mock_session.get.return_value = mock_response

    check_agent_status("agent-1", fail_open=True)
    mock_session.get.assert_called_once()
    args, kwargs = mock_session.get.call_args
    assert args[0] == "https://api.traccia.ai/api/v1/agents/agent-1/status"


@patch("traccia.governance.policy._http_session")
@patch("traccia.governance.policy.load_config")
def test_check_agent_status_hard_block(mock_load_config, mock_session):
    gov_config.status_check_endpoint = None
    mock_load_config.return_value = _FakeConfig()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "hard_block", "policy_id": "pol-1"}
    mock_session.get.return_value = mock_response

    with pytest.raises(AgentBlockedError):
        check_agent_status("agent-1", fail_open=False)


@patch("traccia.governance.policy._http_session")
@patch("traccia.governance.policy.load_config")
def test_check_agent_status_fail_open_on_http_error(mock_load_config, mock_session):
    gov_config.status_check_endpoint = None
    mock_load_config.return_value = _FakeConfig()
    mock_session.get.side_effect = TimeoutError("timeout")

    check_agent_status("agent-1", fail_open=True)


@patch("traccia.governance.policy._http_session")
@patch("traccia.governance.policy.load_config")
def test_check_agent_status_fail_closed_on_http_error(mock_load_config, mock_session):
    gov_config.status_check_endpoint = None
    mock_load_config.return_value = _FakeConfig()
    mock_session.get.side_effect = TimeoutError("timeout")

    with pytest.raises(AgentBlockedError):
        check_agent_status("agent-1", fail_open=False)


@patch("traccia.governance.policy._fetch_and_apply_status")
def test_inflight_coalescing(mock_fetch):
    barrier = threading.Barrier(2)
    call_count = {"n": 0}

    def slow_fetch(agent_id, *, fail_open):
        call_count["n"] += 1
        barrier.wait(timeout=2)
        policy_module._status_cache.set(agent_id, "allowed", None)

    mock_fetch.side_effect = slow_fetch

    errors = []

    def worker():
        try:
            check_agent_status("agent-coalesce", fail_open=True)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors
    assert call_count["n"] == 1
