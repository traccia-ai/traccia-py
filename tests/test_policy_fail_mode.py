"""TRC-342: what @govern policy checks do when Traccia gives no answer.

The customer's setting (Settings -> Policy Enforcement) arrives on answers the SDK
already gets: the agent-status answer at the start of a run and each check answer.
The mode, or a local override, applies only when Traccia gives no answer.
"""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from traccia import observe, runtime_config
from traccia.governance import fail_mode
from traccia.governance import policy as policy_module
from traccia.governance.config import configure_governance, gov_config
from traccia.governance.govern import govern
from traccia.governance.pep import check_policy
from traccia.governance.policy import (
    AgentBlockedError,
    AgentStatusCache,
    ApprovalPending,
    GovernanceConfigError,
    _InFlightCoalescer,
    check_agent_status,
)

KEY = "tr_live_key"
ENDPOINT = "https://api.traccia.ai/v1/traces"
BASE = "https://api.traccia.ai"


def _response(status: int, body: Any = None) -> MagicMock:
    return MagicMock(status_code=status, json=lambda: body, text="")


def _decision(effect="allow", **extra) -> Dict[str, Any]:
    return {"id": "dec", "effect": effect, "would_have": False, "reasons": [], **extra}


def _status(status="allowed", **extra) -> Dict[str, Any]:
    return {"status": status, "agent_id": "support", **extra}


class Traccia:
    """A fake Traccia: status answers on GET, check answers on POST.

    Each answer is a response, an exception, or a callable taking the request.
    Requests carry the API key, so tests can tell callers apart.
    """

    def __init__(self):
        self.status: Any = _response(200, _status())
        self.check: Any = _response(200, _decision())
        self.gets: List[Dict[str, Any]] = []
        self.posts: List[Dict[str, Any]] = []

    def _answer(self, answer, request):
        if callable(answer) and not isinstance(answer, MagicMock):
            answer = answer(request)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def get(self, url, headers=None, timeout=None):
        request = {"url": url, "key": (headers or {}).get("Authorization", "")[7:]}
        self.gets.append(request)
        return self._answer(self.status, request)

    def post(self, url, headers=None, json=None, timeout=None):
        request = {"url": url, "key": (headers or {}).get("Authorization", "")[7:], "json": json}
        if url.endswith("/policy/settle") or url.endswith("/blocks"):
            return _response(200, {})
        self.posts.append(request)
        return self._answer(self.check, request)


@pytest.fixture
def traccia():
    fake = Traccia()
    policy_module._status_cache = AgentStatusCache()
    policy_module._inflight = _InFlightCoalescer()
    gov_config.status_cache_ttl_seconds = 60
    credentials = {"value": (KEY, ENDPOINT)}
    with patch("traccia.governance.policy._http_session", fake), \
            patch("traccia.governance.pep._http_session", fake), \
            patch("traccia.governance.policy.client_config.resolve_credentials", lambda: credentials["value"]), \
            patch("traccia.governance.pep._credentials", lambda: credentials["value"]):
        fake.credentials = credentials
        yield fake


def _checked(**run):
    """A policy check inside a governed run (pep enabled), as the instrumentation makes it."""
    with runtime_config.run_identity(agent_id="support", pep_enabled=True, **run):
        return check_policy(action={"type": "llm_call", "name": "llm_call", "model": "gpt-4o"})


def _outage(traccia, exc: BaseException = ConnectionError("refused")):
    traccia.check = exc
    traccia.status = exc


IDENT = fail_mode.identity(KEY, BASE)


# --- defaults and where the setting comes from ---------------------------------

def test_no_setting_and_no_answer_blocks(traccia):
    _outage(traccia)
    with pytest.raises(AgentBlockedError, match="policy fail mode is closed"):
        _checked()
    with pytest.raises(AgentBlockedError, match="policy fail mode is closed"):
        check_agent_status("support")


def test_the_setting_from_the_status_answer_applies_to_the_runs_checks(traccia):
    traccia.status = _response(200, _status(fail_mode="open", fail_mode_version=5))
    check_agent_status("support")
    _outage(traccia)
    assert _checked()["reasons"] == ["check_error"]          # allowed: the customer chose open


def test_the_setting_from_a_check_answer_applies_to_later_checks(traccia):
    traccia.check = _response(200, _decision(fail_mode="open", fail_mode_version=5))
    assert _checked()["id"] == "dec"
    _outage(traccia)
    assert _checked()["effect"] == "allow"


def test_a_delayed_status_answer_cannot_replace_a_newer_check_answer(traccia):
    traccia.check = _response(200, _decision(fail_mode="closed", fail_mode_version=20))
    _checked()
    traccia.status = _response(200, _status(fail_mode="open", fail_mode_version=10))
    check_agent_status("support")
    _outage(traccia)
    with pytest.raises(AgentBlockedError):
        _checked()


def test_a_delayed_check_answer_cannot_replace_a_newer_status_answer(traccia):
    traccia.status = _response(200, _status(fail_mode="open", fail_mode_version=20))
    check_agent_status("support")
    traccia.check = _response(200, _decision(fail_mode="closed", fail_mode_version=10))
    _checked()
    _outage(traccia)
    assert _checked()["effect"] == "allow"


def test_a_slower_answer_with_a_newer_version_still_wins():
    # Request order is not setting order: the version decides.
    fail_mode.note_answer(IDENT, {"fail_mode": "closed", "fail_mode_version": 20}, started_at=2.0)
    fail_mode.note_answer(IDENT, {"fail_mode": "open", "fail_mode_version": 30}, started_at=1.0)
    assert fail_mode.customer_mode(IDENT) == "open"


# --- null clears; omission preserves --------------------------------------------

def test_null_with_a_newer_version_clears_the_setting(traccia):
    traccia.status = _response(200, _status(fail_mode="open", fail_mode_version=5))
    check_agent_status("support")
    traccia.check = _response(200, _decision(fail_mode=None, fail_mode_version=6))
    _checked()
    _outage(traccia)
    with pytest.raises(AgentBlockedError):                    # back to the default, closed
        _checked()


def test_null_with_an_older_version_does_not_clear():
    fail_mode.note_answer(IDENT, {"fail_mode": "open", "fail_mode_version": 6}, started_at=1.0)
    fail_mode.note_answer(IDENT, {"fail_mode": None, "fail_mode_version": 5}, started_at=2.0)
    assert fail_mode.customer_mode(IDENT) == "open"


@pytest.mark.parametrize("answer", [
    {},                                                       # ingestion could not read the setting
    {"fail_mode": None},                                      # half a snapshot
    {"fail_mode_version": 99},
], ids=["both_missing", "version_missing", "mode_missing"])
def test_missing_fields_leave_the_setting_unchanged(traccia, answer):
    traccia.status = _response(200, _status(fail_mode="open", fail_mode_version=5))
    check_agent_status("support")
    policy_module._status_cache = AgentStatusCache()           # make the next status a real answer
    traccia.status = _response(200, _status(**answer))
    check_agent_status("support")
    traccia.check = _response(200, _decision(**answer))
    _checked()
    assert fail_mode.customer_mode(IDENT) == "open"


INVALID_DECISIONS = [
    {"effect": "allow", "fail_mode": "open"},                 # no would_have
    {"effect": "allow", "would_have": "false"},
    {"effect": None, "would_have": False},
    {"effect": "unexpected", "would_have": False},
    {"effect": "allow", "would_have": False, "fail_mode": "OPEN", "fail_mode_version": 9},
    {"effect": "allow", "would_have": False, "fail_mode": "open", "fail_mode_version": "9"},
    {"effect": "allow", "would_have": False, "fail_mode": "open", "fail_mode_version": True},
    ["allow"],
]


@pytest.mark.parametrize("body", INVALID_DECISIONS)
def test_an_invalid_decision_is_no_answer_and_does_not_change_the_setting(traccia, body):
    traccia.check = _response(200, body)
    with pytest.raises(AgentBlockedError, match="invalid decision"):
        _checked()
    assert fail_mode.customer_mode(IDENT) is None


def test_an_invalid_status_answer_is_no_answer(traccia):
    traccia.status = _response(200, {"status": "allowed", "fail_mode": "sometimes", "fail_mode_version": 1})
    with pytest.raises(AgentBlockedError, match="invalid status answer"):
        check_agent_status("support")


# --- lifetime -------------------------------------------------------------------

def test_the_setting_expires_back_to_closed(traccia, monkeypatch):
    monkeypatch.setattr(fail_mode, "FAIL_MODE_TTL_S", 0.05)
    traccia.check = _response(200, _decision(fail_mode="open", fail_mode_version=5))
    _checked()
    time.sleep(0.1)
    _outage(traccia)
    with pytest.raises(AgentBlockedError):
        _checked()


def test_a_cached_status_does_not_renew_the_setting(traccia, monkeypatch):
    # The 60 s status cache is not a new answer: reusing it must not keep "open" alive.
    # A cache hit before expiry must not extend it: the setting still expires TTL
    # after the real answer.
    monkeypatch.setattr(fail_mode, "FAIL_MODE_TTL_S", 0.3)
    traccia.status = _response(200, _status(fail_mode="open", fail_mode_version=5))
    check_agent_status("support")
    time.sleep(0.15)
    check_agent_status("support")                             # a cache hit while still fresh
    assert len(traccia.gets) == 1
    time.sleep(0.25)                                          # 0.4 s after the answer
    _outage(traccia)
    with pytest.raises(AgentBlockedError):
        _checked()


def test_only_a_real_answer_renews_the_setting(traccia, monkeypatch):
    monkeypatch.setattr(fail_mode, "FAIL_MODE_TTL_S", 0.15)
    traccia.check = _response(200, _decision(fail_mode="open", fail_mode_version=5))
    _checked()
    time.sleep(0.1)
    _checked()                                                # a real answer, same setting
    time.sleep(0.1)
    _outage(traccia)
    assert _checked()["effect"] == "allow"                    # 0.2 s after the first, 0.1 s after the second


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(fail_mode, "FAIL_MODE_CACHE_MAX", 3)
    for i in range(5):
        fail_mode.note_answer(fail_mode.identity(f"key-{i}", BASE), {"fail_mode": "open", "fail_mode_version": 1}, 0.0)
    assert len(fail_mode._state) == 3
    assert fail_mode.customer_mode(fail_mode.identity("key-0", BASE)) is None     # evicted
    assert fail_mode.customer_mode(fail_mode.identity("key-4", BASE)) == "open"


def test_a_new_init_starts_without_a_remembered_setting():
    fail_mode.note_answer(IDENT, {"fail_mode": "open", "fail_mode_version": 1}, 0.0)
    configure_governance(config_file="/nonexistent/traccia.toml")
    assert fail_mode.customer_mode(IDENT) is None


# --- rejection ------------------------------------------------------------------

@pytest.mark.parametrize("status, reason", [(401, "invalid_api_key"), (403, "permission_denied")])
def test_a_rejected_key_is_refused_even_with_open(traccia, status, reason):
    gov_config.policy_fail_mode = "open"
    traccia.check = _response(200, _decision(fail_mode="open", fail_mode_version=5))
    _checked()
    traccia.check = _response(status, {"detail": "revoked"})
    with pytest.raises(AgentBlockedError) as exc:
        _checked(pep_fail_mode="open")
    assert exc.value.reasons == [reason]
    traccia.status = _response(status, {"detail": "revoked"})
    with pytest.raises(AgentBlockedError, match="rejected this API key"):
        check_agent_status("support", policy_fail_mode="open")


def _in_flight(traccia, then: Callable[[], None], late_answer: Any):
    """Send check A and hold its answer; run *then*; release A with *late_answer*.

    *then* may replace traccia.check: A is already on its way.
    """
    reached, release = threading.Event(), threading.Event()

    def held(request):
        reached.set()
        release.wait(5)
        if isinstance(late_answer, BaseException):
            raise late_answer
        return late_answer

    traccia.check = held
    outcome: Dict[str, Any] = {}

    def run_a():
        try:
            with runtime_config.run_identity(agent_id="support", pep_enabled=True, pep_fail_mode="open"):
                outcome["a"] = check_policy(action={"type": "llm_call", "name": "A"})
        except Exception as exc:
            outcome["a"] = exc

    thread = threading.Thread(target=run_a)
    thread.start()
    assert reached.wait(5)
    then()
    release.set()
    thread.join(5)
    return outcome["a"]


@pytest.mark.parametrize("late", [
    _response(200, _decision(fail_mode="open", fail_mode_version=9)),
    _response(500, {"detail": "down"}),
    TimeoutError("read timed out"),
    _response(200, {"effect": "allow"}),
], ids=["allow", "http_500", "timeout", "malformed"])
def test_a_check_sent_before_a_rejection_is_refused(traccia, late):
    # A is sent, B gets 401, a fresh C caches open, then A answers or fails. A must be
    # refused, not allowed through C's open (the sequence behind the gateway bug).
    def rejection_then_fresh_open():
        traccia.check = _response(401, {"detail": "revoked"})
        with pytest.raises(AgentBlockedError):
            _checked()
        traccia.check = _response(200, _decision(fail_mode="open", fail_mode_version=10))
        _checked()
        assert fail_mode.customer_mode(IDENT) == "open"

    a = _in_flight(traccia, rejection_then_fresh_open, late)
    assert isinstance(a, AgentBlockedError) and a.reasons == ["invalid_api_key"]
    assert fail_mode.customer_mode(IDENT) == "open"          # C's answer still stands


def test_a_status_answer_sent_before_a_rejection_is_refused(traccia):
    reached, release = threading.Event(), threading.Event()

    def status(request):
        reached.set()
        release.wait(5)
        return _response(200, _status(fail_mode="open", fail_mode_version=9))

    traccia.status = status
    outcome = {}

    def run():
        try:
            check_agent_status("support")
            outcome["status"] = "allowed"
        except Exception as exc:
            outcome["status"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    assert reached.wait(5)
    traccia.check = _response(401, {"detail": "revoked"})
    with pytest.raises(AgentBlockedError):
        _checked()
    release.set()
    thread.join(5)
    assert isinstance(outcome["status"], AgentBlockedError)
    assert fail_mode.customer_mode(IDENT) is None


# --- precedence: the override applies only when there is no answer ---------------

def test_the_open_override_never_beats_a_hard_block(traccia):
    traccia.status = _response(200, _status("hard_block", policy_id="pol-1"))
    with patch.object(policy_module, "_record_block_async"):
        with pytest.raises(AgentBlockedError, match="hard blocked"):
            check_agent_status("support", policy_fail_mode="open")


def test_the_open_override_never_beats_a_deny(traccia):
    traccia.check = _response(200, _decision("deny", reasons=["spend exceeded cap"]))
    with pytest.raises(AgentBlockedError, match="spend exceeded cap"):
        _checked(pep_fail_mode="open")


def test_the_open_override_never_beats_a_configuration_error(traccia, monkeypatch):
    traccia.credentials["value"] = (None, ENDPOINT)
    with pytest.raises(GovernanceConfigError):
        _checked(pep_fail_mode="open")
    traccia.credentials["value"] = (KEY, ENDPOINT)
    monkeypatch.delenv("TRACCIA_AGENT_ID", raising=False)
    runtime_config.set_agent_id(None)
    with runtime_config.run_identity(pep_enabled=True, pep_fail_mode="open"):
        with pytest.raises(GovernanceConfigError, match="agent_id"):
            check_policy(action={"type": "llm_call"})


def test_observe_warn_and_approvals_are_unchanged_under_closed(traccia):
    traccia.check = _response(200, {**_decision("deny"), "would_have": True})
    assert _checked(pep_fail_mode="closed")["would_have"] is True      # Observe/Warn: proceeds
    traccia.check = _response(200, _decision("queue", approval_id="apr-1"))
    with pytest.raises(ApprovalPending):
        _checked(pep_fail_mode="closed")


def test_the_closed_override_beats_an_open_setting(traccia):
    traccia.check = _response(200, _decision(fail_mode="open", fail_mode_version=5))
    _checked()
    _outage(traccia)
    with pytest.raises(AgentBlockedError):
        _checked(pep_fail_mode="closed")


def test_the_init_override_applies_to_direct_checks(traccia):
    # check_policy() called by the customer's own code follows the same rules.
    _outage(traccia)
    runtime_config.set_agent_id("support")
    try:
        with pytest.raises(AgentBlockedError):
            check_policy(action={"type": "tool_call", "name": "refund"})
        gov_config.policy_fail_mode = "open"
        assert check_policy(action={"type": "tool_call", "name": "refund"})["effect"] == "allow"
    finally:
        runtime_config.set_agent_id(None)


def test_an_override_that_disagrees_warns_once(traccia, caplog):
    traccia.check = _response(200, _decision(fail_mode="closed", fail_mode_version=5))
    _checked()
    _outage(traccia)
    with caplog.at_level("WARNING", logger="traccia.governance"):
        _checked(pep_fail_mode="open")
        _checked(pep_fail_mode="open")
    assert sum("overrides your Traccia setting" in r.getMessage() for r in caplog.records) == 1


def test_init_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="policy_fail_mode"):
        configure_governance(policy_fail_mode="sometimes", config_file="/nonexistent/traccia.toml")


# --- identity -------------------------------------------------------------------

def test_two_keys_with_the_same_agent_never_share_status_or_setting(traccia):
    def status(request):
        if request["key"] == "key-a":
            return _response(200, _status("hard_block", policy_id="pol", fail_mode="open", fail_mode_version=3))
        return _response(200, _status())

    traccia.status = status
    traccia.credentials["value"] = ("key-a", ENDPOINT)
    with patch.object(policy_module, "_record_block_async"):
        with pytest.raises(AgentBlockedError):
            check_agent_status("support")
    traccia.credentials["value"] = ("key-b", ENDPOINT)
    check_agent_status("support")                              # not key A's cached hard block
    assert [g["key"] for g in traccia.gets] == ["key-a", "key-b"]
    assert fail_mode.customer_mode(fail_mode.identity("key-a", BASE)) == "open"
    assert fail_mode.customer_mode(fail_mode.identity("key-b", BASE)) is None


def test_two_endpoints_never_share_status(traccia):
    check_agent_status("support")
    traccia.credentials["value"] = (KEY, "https://eu.traccia.ai/v1/traces")
    check_agent_status("support")
    assert [g["url"] for g in traccia.gets] == [
        "https://api.traccia.ai/api/v1/agents/support/status",
        "https://eu.traccia.ai/api/v1/agents/support/status",
    ]


def test_callers_sharing_one_status_fetch_apply_their_own_override(traccia):
    reached, release = threading.Event(), threading.Event()

    def status(request):
        reached.set()
        release.wait(5)
        raise ConnectionError("refused")

    traccia.status = status
    outcome = {}

    def caller(name, mode):
        try:
            check_agent_status("support", policy_fail_mode=mode)
            outcome[name] = "allowed"
        except AgentBlockedError:
            outcome[name] = "blocked"

    leader = threading.Thread(target=caller, args=("open", "open"))
    leader.start()
    assert reached.wait(5)
    follower = threading.Thread(target=caller, args=("default", None))
    follower.start()
    time.sleep(0.05)
    release.set()
    leader.join(5)
    follower.join(5)
    assert len(traccia.gets) == 1
    assert outcome == {"open": "allowed", "default": "blocked"}


# --- govern() --------------------------------------------------------------------

def _tool_run(**govern_kwargs):
    ran = []

    @observe(as_type="tool")
    def refund(amount: int):
        ran.append(amount)
        return {"refunded": amount}

    @govern(agent_id="support", **govern_kwargs)
    def run():
        return refund(5)

    return run, ran


def test_fail_open_is_a_deprecated_alias_for_the_override(traccia):
    _outage(traccia)
    with pytest.warns(DeprecationWarning, match="fail_open"):
        run, ran = _tool_run(fail_open=True)
    assert run() == {"refunded": 5}
    with pytest.warns(DeprecationWarning):
        run, ran = _tool_run(fail_open=False)
    with pytest.raises(AgentBlockedError):
        run()
    assert ran == []


def test_omitting_fail_open_follows_the_traccia_setting(traccia):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        run, ran = _tool_run()
    traccia.status = _response(200, _status(fail_mode="open", fail_mode_version=5))
    traccia.check = ConnectionError("refused")
    assert run() == {"refunded": 5}                           # the customer chose open
    fail_mode.clear_for_tests()
    policy_module._status_cache = AgentStatusCache()
    traccia.status = ConnectionError("refused")
    with pytest.raises(AgentBlockedError):                    # nothing known: closed
        run()


@pytest.mark.parametrize("fail_open, mode", [(True, "closed"), (False, "open")])
def test_conflicting_fail_open_and_policy_fail_mode_is_an_error(fail_open, mode):
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="conflicts"):
        govern(agent_id="support", fail_open=fail_open, policy_fail_mode=mode)


def test_matching_fail_open_and_policy_fail_mode_is_accepted():
    with pytest.warns(DeprecationWarning):
        govern(agent_id="support", fail_open=True, policy_fail_mode="open")


def test_concurrent_runs_never_share_their_override(traccia):
    # Status answers; checks fail. The runs interleave at each await, so a leaked
    # override would let the default run through (or block the open one).
    traccia.check = ConnectionError("refused")
    order = []

    @govern(agent_id="support", policy_fail_mode="open")
    async def open_run():
        order.append("open started")
        await asyncio.sleep(0)
        return check_policy(action={"type": "llm_call"})["effect"]

    @govern(agent_id="support")
    async def default_run():
        order.append("default started")
        await asyncio.sleep(0)
        try:
            check_policy(action={"type": "llm_call"})
            return "allowed"
        except AgentBlockedError:
            return "blocked"

    async def both():
        return await asyncio.gather(open_run(), default_run())

    assert asyncio.run(both()) == ["allow", "blocked"]
    assert order == ["open started", "default started"]       # both were in flight together
    assert runtime_config.pep_fail_mode() is None


def test_an_override_ends_with_its_run(traccia):
    # A run's override must not leak into the process: later checks follow the
    # customer's setting (here none, so closed).
    traccia.check = ConnectionError("refused")

    @govern(agent_id="support", policy_fail_mode="open")
    def open_run():
        return check_policy(action={"type": "llm_call"})["effect"]

    assert open_run() == "allow"
    assert gov_config.policy_fail_mode is None
    with pytest.raises(AgentBlockedError):
        _checked()
    runtime_config.set_agent_id("support")
    try:
        with pytest.raises(AgentBlockedError):
            check_policy(action={"type": "tool_call", "name": "refund"})   # direct, outside any run
    finally:
        runtime_config.set_agent_id(None)


def test_an_override_applies_to_nested_checks(traccia):
    _outage(traccia)

    @govern(agent_id="support", policy_fail_mode="open")
    def run():
        def nested():
            return check_policy(action={"type": "tool_call", "name": "lookup"})
        return nested()["effect"]

    assert run() == "allow"


# --- instrumentation never swallows the new errors -------------------------------

def test_a_blocked_llm_call_never_reaches_the_provider(traccia, monkeypatch):
    # The real OpenAI wrapper around a stand-in client: the check fails closed, the
    # wrapper re-raises, and the provider is never called.
    import sys
    import types

    import traccia as traccia_pkg
    from traccia.instrumentation import openai as instrumentation
    from traccia.tracer.provider import TracerProvider

    provider_calls = []
    fake_openai = types.ModuleType("openai")
    fake_openai.ChatCompletion = types.SimpleNamespace(create=lambda **kwargs: provider_calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(traccia_pkg, "_global_tracer_provider", TracerProvider())
    monkeypatch.setattr(instrumentation, "_patched", False)
    assert instrumentation.patch_openai()
    traccia.check = ConnectionError("refused")

    @govern(agent_id="support")
    def run():
        return fake_openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(AgentBlockedError, match="policy fail mode is closed"):
        run()
    assert provider_calls == []
    traccia.check = _response(200, _decision())
    run()
    assert len(provider_calls) == 1                           # the same wrapper passes an allow


# --- review: malformed status, cached status after a rejection, repeated init() ---

@pytest.mark.parametrize("body", [
    {},
    {"status": "nonsense"},
    {"status": None},
    {"status": 5},
    {"agent_id": "support", "fail_mode": "open", "fail_mode_version": 3},     # a setting but no status
], ids=["empty", "unknown", "null", "number", "no_status"])
def test_a_status_answer_without_a_known_status_is_no_answer(traccia, body):
    traccia.status = _response(200, body)
    ran = []

    @govern(agent_id="support")
    def run():
        ran.append(1)

    with pytest.raises(AgentBlockedError, match="invalid status answer"):
        run()
    assert ran == []
    assert fail_mode.customer_mode(IDENT) is None              # nothing learned from it
    check_agent_status("support", policy_fail_mode="open")     # it is "no answer": the override applies


def test_a_rejection_disqualifies_cached_statuses_for_every_agent_of_the_key(traccia):
    check_agent_status("support")
    check_agent_status("billing")
    assert len(traccia.gets) == 2                              # both cached as allowed
    traccia.check = _response(401, {"detail": "revoked"})
    with pytest.raises(AgentBlockedError):
        _checked()                                              # the key is rejected on a check
    traccia.status = _response(401, {"detail": "revoked"})
    for agent in ("support", "billing"):
        with pytest.raises(AgentBlockedError, match="rejected this API key"):
            check_agent_status(agent, policy_fail_mode="open")
    assert len(traccia.gets) == 4                              # neither cached allow was reused
    traccia.status = _response(200, _status())                 # key restored
    check_agent_status("support")
    check_agent_status("support")                              # cached again
    assert len(traccia.gets) == 5


def test_a_rejection_of_another_key_keeps_this_keys_cached_status(traccia):
    check_agent_status("support")
    fail_mode.note_rejection(fail_mode.identity("other-key", BASE), 401)
    check_agent_status("support")
    assert len(traccia.gets) == 1


@pytest.fixture
def initialized(monkeypatch, tmp_path):
    import traccia as traccia_pkg

    traccia_pkg.stop_tracing()
    for name in ("TRACCIA_API_KEY", "TRACCIA_ENDPOINT", "AGENT_DASHBOARD_API_KEY", "AGENT_DASHBOARD_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    kwargs = dict(api_key=KEY, endpoint=ENDPOINT, agent_id="support", auto_start_trace=False)
    traccia_pkg.init(**kwargs)
    yield lambda **extra: traccia_pkg.init(**{**kwargs, **extra})
    traccia_pkg.stop_tracing()


def test_calling_init_again_forgets_the_remembered_setting(initialized):
    fail_mode.note_answer(IDENT, {"fail_mode": "open", "fail_mode_version": 5}, 0.0)
    assert fail_mode.customer_mode(IDENT) == "open"
    initialized()
    assert fail_mode.customer_mode(IDENT) is None


def test_calling_init_again_applies_its_override(initialized):
    initialized(policy_fail_mode="closed")
    assert gov_config.policy_fail_mode == "closed"
    initialized(policy_fail_mode="open")
    assert gov_config.policy_fail_mode == "open"
    with pytest.raises(ValueError, match="policy_fail_mode"):
        initialized(policy_fail_mode="sometimes")


@pytest.mark.parametrize("late", [
    _response(200, _decision(fail_mode="open", fail_mode_version=9)),
    TimeoutError("read timed out"),
], ids=["allow", "timeout"])
def test_calling_init_again_keeps_the_rejection_barrier(traccia, initialized, late):
    # A is sent, B gets 401, init() runs again, then A answers. A must be refused:
    # the re-init forgets settings, not rejections.
    def rejection_then_init():
        traccia.check = _response(401, {"detail": "revoked"})
        with pytest.raises(AgentBlockedError):
            _checked()
        initialized(policy_fail_mode="open")

    a = _in_flight(traccia, rejection_then_init, late)
    assert isinstance(a, AgentBlockedError) and a.reasons == ["invalid_api_key"]
    traccia.check = _response(200, _decision(fail_mode="open", fail_mode_version=10))
    assert _checked()["id"] == "dec"                           # a check sent after it is answered normally
    assert fail_mode.customer_mode(IDENT) == "open"
