"""Runtime policy enforcement combined with observability."""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Callable, Optional

from traccia import observe, runtime_config
from traccia.governance.evidence import governed_run
from traccia.governance.policy import check_agent_status

logger = logging.getLogger("traccia.governance")


def _resolve_govern_agent_id(explicit: Optional[str]) -> Optional[str]:
    """init is identity. govern agent_id is an override only."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    from_init = runtime_config.get_agent_id()
    if from_init and str(from_init).strip():
        return str(from_init).strip()
    env = (os.environ.get("TRACCIA_AGENT_ID") or "").strip()
    return env or None


def govern(agent_id: Optional[str] = None, fail_open: bool = True, **observe_kwargs):
    """
    Observability plus runtime policy enforcement.

    Unlike @observe, @govern:
    1. Polls agent status (lagged next-run breaker) before the function body.
    2. Turns on per-call policy checks for instrumented LLM clients and
       @observe(as_type="tool") functions (Spend Cap, Model Boundary, Loop Cap).

    Identity comes from init(agent_id=...) or TRACCIA_AGENT_ID. Pass agent_id
    here only to override for one function in a multi-agent process.

    Deny raises AgentBlockedError. Reshape may swap the model on the in-flight
    request. Observe/Warn in the dashboard still let the call proceed.

    fail_open=False blocks the run, and every checked call inside it, when Traccia
    cannot give a decision (missing credentials, HTTP error, timeout).

    Requires a Traccia account (API key + endpoint). Tracing-only or self-hosted
    setups should use @observe instead.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        observed_func = observe(**observe_kwargs)(func)
        attributes = observe_kwargs.get("attributes") or {}
        agent_name = attributes.get("agent.name")
        override_id = agent_id

        def _enforce_or_warn() -> None:
            aid = _resolve_govern_agent_id(override_id)
            if not aid:
                logger.warning(
                    "No agent_id on init, @govern, or TRACCIA_AGENT_ID. "
                    "Skipping policy check."
                )
                return
            check_agent_status(aid, fail_open=fail_open)

        def _run_kwargs() -> dict:
            kwargs: dict[str, Any] = {"pep_enabled": True, "pep_fail_open": fail_open}
            if override_id and str(override_id).strip():
                kwargs["agent_id"] = str(override_id).strip()
            if agent_name:
                kwargs["agent_name"] = agent_name
            return kwargs

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _enforce_or_warn()
            with runtime_config.run_identity(**_run_kwargs()), governed_run():
                return observed_func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _enforce_or_warn()
            with runtime_config.run_identity(**_run_kwargs()), governed_run():
                return await observed_func(*args, **kwargs)

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    if callable(agent_id):
        func = agent_id
        agent_id = None
        fail_open = True
        return decorator(func)

    return decorator
