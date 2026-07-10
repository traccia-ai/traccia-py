"""Internal runtime policy checks against the Traccia platform."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

from traccia.config import load_config
from traccia.governance.config import gov_config

logger = logging.getLogger("traccia.governance")

DEFAULT_STATUS_PATH = "/api/v1/agents/{agent_id}/status"
DEFAULT_BLOCK_PATH = "/api/v1/agents/{agent_id}/blocks"


class AgentBlockedError(Exception):
    """Raised when an agent execution is hard blocked by governance policy."""


class AgentStatusCache:
    """Thread-safe TTL cache for agent status responses."""

    def __init__(self, ttl_seconds: int = 60) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[float, str, Optional[str]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Tuple[str, Optional[str]]]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            timestamp, status, policy_id = entry
            if time.time() - timestamp < self.ttl_seconds:
                return status, policy_id
            del self._cache[key]
            return None

    def set(self, key: str, status: str, policy_id: Optional[str] = None) -> None:
        with self._lock:
            self._cache[key] = (time.time(), status, policy_id)


class _InFlightCoalescer:
    """Share one HTTP status fetch when multiple callers miss the cache."""

    def __init__(self) -> None:
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def run(self, key: str, fn):
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
    elif status != "allowed":
        logger.warning("Unknown agent status received: %s. Treating as allowed.", status)


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


def _fetch_and_apply_status(agent_id: str, *, fail_open: bool) -> None:
    config = load_config()
    api_key = config.tracing.api_key
    if not api_key:
        logger.warning(
            "Traccia API key not found. @govern requires a Traccia platform account; "
            "use @observe for tracing-only setups."
        )
        return

    traces_endpoint = config.tracing.endpoint
    if not traces_endpoint:
        logger.warning(
            "Traccia endpoint not found. @govern requires a Traccia platform endpoint; "
            "use @observe for tracing-only setups."
        )
        return

    base_url = _derive_base_url(traces_endpoint)
    status_url = _status_url(base_url, agent_id)
    headers = {"Authorization": f"Bearer {api_key}"}

    response = _http_session.get(status_url, headers=headers, timeout=5)
    if response.status_code != 200:
        message = (
            f"Failed to fetch agent status from Traccia (HTTP {response.status_code})."
        )
        if fail_open:
            logger.warning("%s Allowing execution (fail_open).", message)
            return
        raise AgentBlockedError(
            f"{message} Blocking execution because fail_open is False."
        )

    status_data = response.json()
    status = status_data.get("status", "allowed")
    policy_id = status_data.get("policy_id")

    _status_cache.set(agent_id, status, policy_id)

    if status in ("soft_block", "hard_block") and policy_id:
        _record_block_async(
            _block_url(base_url, agent_id),
            api_key,
            agent_id,
            status,
            policy_id,
        )

    _handle_status(status, agent_id)


def check_agent_status(agent_id: str, *, fail_open: bool = True) -> None:
    """
    Verify agent status with the Traccia platform before execution.

    Requires a Traccia API key and endpoint. Open-source / self-hosted tracing-only
    users should use @observe instead of @govern.
    """
    _status_cache.ttl_seconds = gov_config.status_cache_ttl_seconds

    cached = _status_cache.get(agent_id)
    if cached:
        status, _policy_id = cached
        _handle_status(status, agent_id)
        return

    def _fetch() -> None:
        _fetch_and_apply_status(agent_id, fail_open=fail_open)

    try:
        _inflight.run(agent_id, _fetch)
    except AgentBlockedError:
        raise
    except Exception as exc:
        if fail_open:
            logger.warning(
                "Error fetching agent status from Traccia API: %s. Allowing execution (fail_open).",
                exc,
            )
            return
        logger.error(
            "Error fetching agent status from Traccia API: %s. Blocking execution (fail_closed).",
            exc,
        )
        raise AgentBlockedError(
            f"Failed to verify agent status and fail_open is False. Error: {exc}"
        ) from exc
