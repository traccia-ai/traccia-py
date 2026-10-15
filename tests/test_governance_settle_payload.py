"""
Spend Cap settlement payloads and prompt estimates in traccia.governance.pep, used by the
Groq integration: what the SDK sends so the server charges a call instead of releasing its
reservation, and how settles run in the background.
"""

from unittest.mock import MagicMock, patch

import pytest

from traccia import runtime_config
from traccia.governance.pep import enforce_llm_call, finish_llm_call, reserved_usd, settle_policy

CREDENTIALS = ("key", "https://app.traccia.ai/v1/traces")


class PolicyServer:
    def __init__(self, decision=None):
        self.decision = decision or {"effect": "allow", "would_have": False, "obligations": {"reserved_usd": 0.25}}
        self.checks = []
        self._settles = []
        self.session = MagicMock()
        self.session.post.side_effect = self._post

    @property
    def settles(self):
        """Settles received, after waiting for those still running in the background."""
        from traccia.governance.pep import flush_pending_settles

        flush_pending_settles()
        return self._settles

    def _post(self, url, **kwargs):
        response = MagicMock(status_code=200)
        if url.endswith("/check"):
            self.checks.append(kwargs["json"])
            response.json.return_value = self.decision
        else:
            self._settles.append(kwargs["json"])
            response.json.return_value = {}
        return response


@pytest.fixture
def server():
    server = PolicyServer()
    with patch("traccia.governance.pep._credentials", return_value=CREDENTIALS):
        with patch("traccia.governance.pep._http_session", server.session):
            with runtime_config.run_identity(agent_id="support", pep_enabled=True):
                yield server


# -- pep ---------------------------------------------------------------------------


def test_settle_sends_model_vendor_and_token_counts(server):
    settle_policy(
        {"obligations": {"reserved_usd": 0.25}},
        model="openai/gpt-oss-120b",
        vendor="groq",
        input_tokens=11,
        output_tokens=5,
    )
    [body] = server.settles
    assert body["reserved_usd"] == 0.25
    assert body["release"] is False
    assert body["actual_usd"] is None
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["vendor"] == "groq"
    assert body["input_tokens"] == 11
    assert body["output_tokens"] == 5


def test_finish_llm_call_passes_everything_through(server):
    finish_llm_call(
        {"obligations": {"reserved_usd": 0.25}},
        actual_usd=0.01,
        model="m",
        vendor="groq",
        input_tokens=1,
        output_tokens=2,
    )
    body = server.settles[0]
    assert (body["actual_usd"], body["model"], body["vendor"], body["input_tokens"], body["output_tokens"]) == (
        0.01,
        "m",
        "groq",
        1,
        2,
    )


def test_settle_without_usage_keeps_the_old_payload(server):
    finish_llm_call({"obligations": {"reserved_usd": 0.25}}, release=True)
    body = server.settles[0]
    assert body["release"] is True
    assert body["input_tokens"] is None and body["output_tokens"] is None


def test_enforce_sends_the_vendor_only_when_given(server):
    enforce_llm_call({"model": "llama-3.3-70b-versatile"}, vendor="groq")
    enforce_llm_call({"model": "gpt-4o-mini"})
    assert server.checks[0]["context"]["vendor"] == "groq"
    assert "vendor" not in server.checks[1]["context"]


@pytest.mark.parametrize(
    "decision,expected",
    [
        ({"obligations": {"reserved_usd": 0.25}}, 0.25),
        ({"obligations": {"reserved_usd": 1}}, 1),
        ({"obligations": {}}, None),
        ({"obligations": {"reserved_usd": True}}, None),
        ({}, None),
        (None, None),
    ],
)
def test_reserved_usd(decision, expected):
    assert reserved_usd(decision) == expected


# -- prompt estimate sent with the policy check -------------------------------------


from traccia.governance import pep as pep_mod  # noqa: E402
from traccia.governance.pep import (  # noqa: E402
    IMAGE_PROMPT_TOKENS,
    estimate_prompt_tokens,
    finish_llm_call_in_background,
    flush_pending_settles,
)


def test_estimate_counts_text_parts_not_their_repr():
    text = "word " * 2000
    as_string = estimate_prompt_tokens({"messages": [{"role": "user", "content": text}]})
    as_parts = estimate_prompt_tokens({"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]})
    assert as_parts == as_string


def test_estimate_counts_an_image_at_a_fixed_allowance_not_by_its_url():
    huge_data_url = "data:image/png;base64," + "A" * 500_000
    content = [{"type": "image_url", "image_url": {"url": huge_data_url}}]
    assert estimate_prompt_tokens({"messages": [{"role": "user", "content": content}]}) == IMAGE_PROMPT_TOKENS


def test_estimate_without_tiktoken_never_counts_unspaced_text_as_a_few_tokens(monkeypatch):
    import traccia.processors.token_counter as tc

    monkeypatch.setattr(tc, "_count_with_tiktoken", lambda text, model=None: None)
    blob = '{"a":' + "1" * 3995 + "}"  # 4,001 characters, no whitespace
    assert estimate_prompt_tokens({"messages": [{"role": "user", "content": blob}]}) == 1000


def test_estimate_counts_tool_call_arguments():
    message = {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "x" * 400}}]}
    assert estimate_prompt_tokens({"messages": [message]}) >= 100


def test_estimate_is_none_for_an_empty_prompt():
    assert estimate_prompt_tokens({"messages": [{"role": "user", "content": None}]}) is None


# -- background settles ---------------------------------------------------------------


def test_background_settle_runs_as_the_calling_agent(server):
    finish_llm_call_in_background({"obligations": {"reserved_usd": 0.25}}, actual_usd=0.01)
    flush_pending_settles()
    [body] = server.settles
    assert body["agent_id"] == "support"
    assert body["actual_usd"] == 0.01


def test_background_settle_skips_calls_with_nothing_reserved(server, monkeypatch):
    submitted = []
    monkeypatch.setattr(pep_mod, "_settle_pool", lambda: submitted.append(1))
    finish_llm_call_in_background(None)
    finish_llm_call_in_background({"obligations": {}})
    assert submitted == []


def test_flush_waits_for_a_slow_settle(server, monkeypatch):
    import threading

    gate = threading.Event()
    original = pep_mod.finish_llm_call

    def slow(*args, **kwargs):
        gate.wait(5)
        original(*args, **kwargs)

    monkeypatch.setattr(pep_mod, "finish_llm_call", slow)
    finish_llm_call_in_background({"obligations": {"reserved_usd": 0.25}}, release=True)
    assert server._settles == []  # still running
    threading.Timer(0.05, gate.set).start()
    flush_pending_settles()
    assert len(server.settles) == 1
