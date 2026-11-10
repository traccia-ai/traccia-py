"""Credentials passed to init() must reach policy, prompts and evals — not only tracing."""

from unittest.mock import MagicMock, patch

import pytest

import traccia
from traccia.config import load_config
from traccia.governance.pep import check_policy
from traccia.prompts.client import resolve_credentials

API_KEY = "tr_test_init_only_key"
ENDPOINT = "http://127.0.0.1:9/v1/traces"
CRED_ENV = ("TRACCIA_API_KEY", "TRACCIA_ENDPOINT", "AGENT_DASHBOARD_API_KEY", "AGENT_DASHBOARD_ENDPOINT")


@pytest.fixture
def init_only_credentials(monkeypatch, tmp_path):
    traccia.stop_tracing()  # an earlier test may have left the SDK initialized
    for name in CRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    # Precondition: no env/file credentials anywhere, so init() is the only source.
    assert not load_config().tracing.api_key
    assert not load_config().tracing.endpoint

    traccia.init(api_key=API_KEY, endpoint=ENDPOINT, agent_id="agent-1", auto_start_trace=False)
    yield
    traccia.stop_tracing()


def test_policy_check_uses_init_credentials(init_only_credentials):
    with patch("traccia.governance.pep._http_session") as session:
        session.post.return_value = MagicMock(status_code=200, json=lambda: {"effect": "allow", "would_have": False})
        decision = check_policy(action={"type": "llm_call", "name": "llm_call", "model": "gpt-4o-mini"})

    assert "missing_credentials" not in decision.get("reasons", [])
    session.post.assert_called_once()
    assert session.post.call_args.args[0] == "http://127.0.0.1:9/api/v1/policy/check"
    assert session.post.call_args.kwargs["headers"]["Authorization"] == f"Bearer {API_KEY}"


def test_prompt_and_eval_credentials_use_init(init_only_credentials):
    assert resolve_credentials() == (API_KEY, "http://127.0.0.1:9")


def test_explicit_argument_overrides_init(init_only_credentials):
    assert resolve_credentials(api_key="per-call")[0] == "per-call"


FILE_KEY = "tr_test_file_only_key"
FILE_ENDPOINT = "http://127.0.0.2:9/v1/traces"


@pytest.fixture
def file_only_credentials(monkeypatch, tmp_path):
    traccia.stop_tracing()
    for name in CRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "traccia.toml").write_text(
        f'[tracing]\napi_key = "{FILE_KEY}"\nendpoint = "{FILE_ENDPOINT}"\n'
    )
    yield
    traccia.stop_tracing()


@pytest.mark.parametrize("initialize", ["init", "start_tracing"])
def test_file_key_is_never_paired_with_the_default_endpoint(file_only_credentials, initialize):
    if initialize == "init":
        traccia.init(agent_id="agent-1", auto_start_trace=False, enable_patching=False)
    else:
        # start_tracing() reads env only, so it has no key; the file key must keep the file endpoint.
        traccia.start_tracing(agent_id="agent-1", enable_patching=False)

    with patch("traccia.governance.pep._http_session") as session:
        session.post.return_value = MagicMock(status_code=200, json=lambda: {"effect": "allow", "would_have": False})
        check_policy(action={"type": "llm_call", "name": "llm_call", "model": "gpt-4o-mini"})

    session.post.assert_called_once()
    assert session.post.call_args.args[0] == "http://127.0.0.2:9/api/v1/policy/check"
    assert session.post.call_args.kwargs["headers"]["Authorization"] == f"Bearer {FILE_KEY}"
    assert resolve_credentials() == (FILE_KEY, "http://127.0.0.2:9")
