"""Internal runtime policy checks against the Traccia platform."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

from traccia import client_config
from traccia.governance import fail_mode
from traccia.governance.config import gov_config

logger = logging.getLogger("traccia.governance")

DEFAULT_STATUS_PATH = "/api/v1/agents/{agent_id}/status"
AGENT_STATUSES = ("allowed", "soft_block", "hard_block")
DEFAULT_BLOCK_PATH = "/api/v1/agents/{agent_id}/blocks"


class AgentBlockedError(Exception):
    """Raised when an agent execution is hard blocked by governance policy."""

    def __init__(self, message: str, *, decision_id=None, remaining_budget_usd=None, reasons=None):
        super().__init__(message)
        self.decision_id = decision_id
        self.remaining_budget_usd = remaining_budget_usd
        self.reasons = reasons or []


class GovernanceConfigError(AgentBlockedError):
    """@govern or a policy check is missing its API key, endpoint or agent ID.

    The call is blocked, whatever the fail mode: a misconfigured check never allows.
    """


REJECT_REASONS = {401: "invalid_api_key", 403: "permission_denied"}


class ApprovalPending(Exception):
    """Raised when a tool call is waiting for a person. Do not run the tool or retry the check.

    Catch this at the agent boundary. If the framework retries raised errors, return
    pending_tool_result() so the model ends the turn. Traccia does not resume the agent.
    """

    def __init__(self, message: str, *, approval_id=None, expires_at=None, decision_id=None):
        super().__init__(message)
        self.approval_id = approval_id
        self.expires_at = expires_at
        self.decision_id = decision_id


def pending_tool_result(error: ApprovalPending) -> dict:
    """A normal tool result the model can read. Use this when a framework retries exceptions."""
    return {
        "status": "pending_approval",
        "approval_id": error.approval_id,
        "expires_at": error.expires_at,
    }


class AgentStatusCache:
    """Thread-safe TTL cache for agent status responses."""

    def __init__(self, ttl_seconds: int = 60) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[Any, Tuple[float, str, Optional[str]]] = {}
        # When the request behind each entry was sent (fail_mode clock): a status
        # fetched before the key was rejected no longer holds (TRC-342).
        self._sent_at: Dict[Any, float] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Tuple[str, Optional[str]]]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            timestamp, status, policy_id = entry
            if time.time() - timestamp < self.ttl_seconds:
                return status, policy_id
            del self._cache[key]
            return None

    def set(self, key: Any, status: str, policy_id: Optional[str] = None, sent_at: Optional[float] = None) -> None:
        with self._lock:
            self._cache[key] = (time.time(), status, policy_id)
            self._sent_at[key] = fail_mode._now() if sent_at is None else sent_at

    def sent_at(self, key: Any) -> float:
        with self._lock:
            return self._sent_at.get(key, 0.0)

    def drop(self, key: Any) -> None:
        with self._lock:
            self._cache.pop(key, None)
            self._sent_at.pop(key, None)


class _InFlightCoalescer:
    """Share one HTTP status fetch when multiple callers miss the cache."""

    def __init__(self) -> None:
        self._futures: Dict[Any, Future] = {}
        self._lock = threading.Lock()

    def run(self, key: Any, fn):
        with self._lock:
            future = self._futures.get(key)
            if future is None:
                future = Future()
                self._futures[key] = future
                leader = True
            else:
                leader = False

        if leader:
            try:
                result = fn()
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            finally:
                with self._lock:
                    self._futures.pop(key, None)
            return future.result()

        return future.result()


_status_cache = AgentStatusCache()
_inflight = _InFlightCoalescer()
_http_session = requests.Session()


def _derive_base_url(traces_endpoint: str) -> str:
    parsed = urlparse(traces_endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


def _status_url(base_url: str, agent_id: str) -> str:
    if gov_config.status_check_endpoint:
        return gov_config.status_check_endpoint.format(agent_id=agent_id)
    return f"{base_url}{DEFAULT_STATUS_PATH.format(agent_id=agent_id)}"


def _block_url(base_url: str, agent_id: str) -> str:
    if gov_config.post_block_endpoint:
        return gov_config.post_block_endpoint.format(agent_id=agent_id)
    return f"{base_url}{DEFAULT_BLOCK_PATH.format(agent_id=agent_id)}"


def _handle_status(status: str, agent_id: str) -> None:
    if status == "hard_block":
        logger.error(
            "Agent %s is HARD BLOCKED by governance policy. Terminating execution.",
            agent_id,
        )
        raise AgentBlockedError(
            f"Agent {agent_id} execution is hard blocked by governance policy."
        )
    if status == "soft_block":
        logger.warning(
            "Agent %s is SOFT BLOCKED by governance policy. Execution continuing with warning.",
            agent_id,
        )


def _record_block_async(
    block_url: str,
    api_key: str,
    agent_id: str,
    status: str,
    policy_id: str,
) -> None:
    def _post() -> None:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {"policy_id": policy_id, "block_type": status}
            _http_session.post(block_url, headers=headers, json=payload, timeout=5)
        except Exception as exc:
            logger.warning("Failed to record agent block to Traccia API: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


def rejected_error(status: int) -> AgentBlockedError:
    """Traccia refused the API key. No fail mode or override applies."""
    return AgentBlockedError(
        f"Traccia rejected this API key (HTTP {status}). Check TRACCIA_API_KEY and its workspace.",
        reasons=[REJECT_REASONS.get(status, "rejected")],
    )


def _status_key(api_key: str, status_url: str) -> Tuple[str, str]:
    """Status belongs to one API key and URL (endpoint and agent), never to the agent ID alone."""
    return hashlib.sha256(api_key.encode()).hexdigest(), status_url


def _unanswered(ident, started_at: float, detail: str) -> Tuple[str, Any]:
    rejected = fail_mode.rejected_since(ident, started_at)
    if rejected:
        return "rejected", rejected
    return "unavailable", detail


def _fetch_status(api_key: str, base_url: str, status_url: str, agent_id: str) -> Tuple[str, Any]:
    """One status request, shared by the callers that missed the cache. Never raises.

    Returns ("ok", answer), ("rejected", 401/403) or ("unavailable", detail). Records
    the fail mode, a rejection, the cache entry and a block report once, here; each
    caller then applies its own override to the result.
    """
    ident = fail_mode.identity(api_key, base_url)
    started_at = fail_mode._now()
    try:
        response = _http_session.get(status_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
    except Exception as exc:
        return _unanswered(ident, started_at, f"error: {exc}")
    if response.status_code in fail_mode.REJECT_STATUSES:
        fail_mode.note_rejection(ident, response.status_code)
        return "rejected", response.status_code
    if response.status_code != 200:
        return _unanswered(ident, started_at, f"HTTP {response.status_code}")
    try:
        answer = response.json()
    except Exception:
        answer = None
    if not fail_mode.valid_fields(answer) or answer.get("status") not in AGENT_STATUSES:
        return _unanswered(ident, started_at, "invalid status answer")
    rejected = fail_mode.rejected_since(ident, started_at)
    if rejected:
        return "rejected", rejected
    fail_mode.note_answer(ident, answer, started_at)

    status = answer["status"]
    policy_id = answer.get("policy_id")
    _status_cache.set(_status_key(api_key, status_url), status, policy_id, sent_at=started_at)
    if status in ("soft_block", "hard_block") and policy_id:
        _record_block_async(_block_url(base_url, agent_id), api_key, agent_id, status, policy_id)
    return "ok", answer


def check_agent_status(agent_id: str, *, policy_fail_mode: Optional[str] = None) -> None:
    """
    Verify agent status with the Traccia platform before execution.

    A hard block raises AgentBlockedError, whatever the fail mode. When Traccia gives
    no answer, the fail mode applies: *policy_fail_mode* (this run's override), else
    init's, else the customer's setting, else closed (block). A missing API key or
    endpoint raises GovernanceConfigError; use @observe for tracing only.
    """
    api_key, traces_endpoint = client_config.resolve_credentials()
    if not api_key:
        raise GovernanceConfigError(
            "Traccia API key not found. @govern requires a Traccia platform account. "
            "Set TRACCIA_API_KEY, or use @observe for tracing only."
        )
    if not traces_endpoint:
        raise GovernanceConfigError(
            "Traccia endpoint not found. @govern requires a Traccia platform endpoint. "
            "Set TRACCIA_ENDPOINT, or use @observe for tracing only."
        )
    base_url = _derive_base_url(traces_endpoint)
    status_url = _status_url(base_url, agent_id)
    key = _status_key(api_key, status_url)
    ident = fail_mode.identity(api_key, base_url)

    _status_cache.ttl_seconds = gov_config.status_cache_ttl_seconds
    cached = _status_cache.get(key)
    if cached and fail_mode.rejected_since(ident, _status_cache.sent_at(key)):
        # The key was rejected after this status was fetched, by any check for any
        # agent: it no longer holds. Ask again; a still-rejected key gets 401/403.
        _status_cache.drop(key)
        cached = None
    if cached:
        # A cached status is not a new answer: it does not renew the fail mode.
        _handle_status(cached[0], agent_id)
        return

    try:
        outcome, value = _inflight.run(key, lambda: _fetch_status(api_key, base_url, status_url, agent_id))
    except Exception as exc:  # pragma: no cover - _fetch_status does not raise
        outcome, value = "unavailable", f"error: {exc}"
    if outcome == "rejected":
        raise rejected_error(value)
    if outcome == "unavailable":
        local = policy_fail_mode if policy_fail_mode is not None else gov_config.policy_fail_mode
        mode = fail_mode.resolve(ident, local)
        if mode == "open":
            logger.warning("Could not verify agent status with Traccia (%s). Allowing execution (policy fail mode: open).", value)
            return
        logger.error("Could not verify agent status with Traccia (%s). Blocking execution (policy fail mode: closed).", value)
        raise AgentBlockedError(
            f"Could not verify agent status with Traccia ({value}). Blocking because the policy fail mode is closed.",
            reasons=["policy_unavailable"],
        )
    _handle_status(value["status"], agent_id)
