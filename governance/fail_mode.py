"""What policy checks do when Traccia gives no answer (TRC-342).

The customer chooses in Traccia (Settings -> Policy Enforcement): block the call
("closed", the default) or let it through ("open"). The setting arrives on answers
the SDK already gets, the agent-status answer at the start of a @govern run and
each policy-check answer, as ``fail_mode`` and ``fail_mode_version``. It is kept
per API key and endpoint for FAIL_MODE_TTL_S after the last answer that carried it.

A local override (``init(policy_fail_mode=...)``, or ``@govern(policy_fail_mode=...)``
for one run) replaces the customer's setting, but only when Traccia gives no
answer: it never overrides a 401/403, a block, a deny or a configuration error.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("traccia.governance")

FAIL_MODES = ("open", "closed")
DEFAULT_FAIL_MODE = "closed"
FAIL_MODE_TTL_S = 600.0  # the gateway's default (GATEWAY_POLICY_FAIL_MODE_CACHE_S)
FAIL_MODE_CACHE_MAX = 10_000
REJECT_STATUSES = (401, 403)

Identity = Tuple[str, str]

# identity -> (mode or None, expires at, version, rejected at, rejected status).
# A lower version than held is an older, slower answer and is ignored. "rejected
# at" is set on a 401/403: an answer to a request sent before it is refused.
_state: "OrderedDict[Identity, Tuple[Optional[str], float, int, float, int]]" = OrderedDict()
_warned: set = set()
_lock = threading.Lock()


def _now() -> float:
    return time.monotonic()


def identity(api_key: str, endpoint: str) -> Identity:
    """Whose setting this is: the API key (hashed) at that Traccia endpoint."""
    return hashlib.sha256(api_key.encode()).hexdigest(), endpoint


def validate_mode(value: Any, name: str = "policy_fail_mode") -> Optional[str]:
    if value is None or value in FAIL_MODES:
        return value
    raise ValueError(f"{name} must be 'open', 'closed' or None, not {value!r}")


def _is_version(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_fields(answer: Any) -> bool:
    """Fields that are present must be valid. Absent fields are fine (unknown)."""
    if not isinstance(answer, dict):
        return False
    if "fail_mode" in answer and answer["fail_mode"] not in (None, *FAIL_MODES):
        return False
    return "fail_mode_version" not in answer or _is_version(answer["fail_mode_version"])


def _store(key: Identity, entry: Tuple[Optional[str], float, int, float, int]) -> None:
    _state[key] = entry
    _state.move_to_end(key)
    while len(_state) > FAIL_MODE_CACHE_MAX:
        _state.popitem(last=False)


def note_answer(ident: Identity, answer: Dict[str, Any], started_at: float) -> None:
    """Keep the setting from a server answer to a request sent at started_at.

    Both fields must be present: a missing one means the server could not read the
    setting, and what is held stays. ``fail_mode: null`` with a version clears it.
    """
    if "fail_mode" not in answer or "fail_mode_version" not in answer or not valid_fields(answer):
        return
    mode, version = answer["fail_mode"], answer["fail_mode_version"]
    now = _now()
    with _lock:
        held = _state.get(ident)
        if held and (started_at < held[3] or version < held[2]):
            return
        _store(ident, (mode, now + FAIL_MODE_TTL_S, version, held[3] if held else 0.0, held[4] if held else 0))


def note_rejection(ident: Identity, status: int) -> None:
    """The key was refused: drop its setting and refuse requests already on their way."""
    now = _now()
    with _lock:
        held = _state.get(ident)
        _store(ident, (None, now + FAIL_MODE_TTL_S, held[2] if held else 0, now, status))


def rejected_since(ident: Identity, started_at: float) -> Optional[int]:
    """The 401/403 status if the key was refused after a request sent at started_at."""
    with _lock:
        held = _state.get(ident)
    if held and held[3] > started_at:
        return held[4]
    return None


def customer_mode(ident: Identity) -> Optional[str]:
    """The customer's setting from Traccia, or None (unset, unknown or expired)."""
    with _lock:
        held = _state.get(ident)
        if held and held[1] > _now():
            return held[0]
        _state.pop(ident, None)
    return None


def local_mode() -> Optional[str]:
    """The local override: this @govern run's, else init's, else None."""
    from traccia import runtime_config
    from traccia.governance.config import gov_config

    run = runtime_config.pep_fail_mode()
    return run if run is not None else gov_config.policy_fail_mode


def resolve(ident: Identity, local: Optional[str]) -> str:
    """The mode to apply when Traccia gives no answer."""
    traccia = customer_mode(ident) or DEFAULT_FAIL_MODE
    if local is None:
        return traccia
    if local != traccia:
        with _lock:
            first = (ident, local) not in _warned
            _warned.add((ident, local))
        if first:
            logger.warning(
                "policy_fail_mode=%r overrides your Traccia setting (%r) in this process. "
                "The Traccia Gateway follows the Traccia setting.", local, traccia,
            )
    return local


def reset() -> None:
    """Forget remembered settings (a new init()), but keep each key's rejection.

    A request sent before a 401/403 must stay refused even if init() runs while it
    is on its way; requests sent after the rejection are unaffected.
    """
    with _lock:
        rejected = {key: (None, held[1], 0, held[3], held[4]) for key, held in _state.items() if held[3]}
        _state.clear()
        _state.update(rejected)
        _warned.clear()


def clear_for_tests() -> None:
    with _lock:
        _state.clear()
        _warned.clear()
