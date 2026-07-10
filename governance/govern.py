"""Runtime policy enforcement combined with observability."""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Callable, Optional

from traccia import observe, runtime_config
from traccia.governance.policy import check_agent_status

logger = logging.getLogger("traccia.governance")


def govern(agent_id: Optional[str] = None, fail_open: bool = True, **observe_kwargs):
    """
    Observability plus runtime policy enforcement.

    Unlike @observe, @govern calls the Traccia platform agent-status API before each
    invocation. It requires a Traccia account (API key + endpoint). For tracing-only
    or self-hosted setups without the Traccia platform, use @observe instead.

    When agent_id is set, wraps execution in runtime_config.run_identity so child
    spans (LLM/tool calls) inherit the correct agent.id/agent.name for cost attribution.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        observed_func = observe(**observe_kwargs)(func)
        attributes = observe_kwargs.get("attributes") or {}
        agent_name = attributes.get("agent.name")
        resolved_agent_id = agent_id

        def _resolve_agent_id() -> Optional[str]:
            return resolved_agent_id or os.environ.get("TRACCIA_AGENT_ID")

        def _enforce_or_warn() -> None:
            aid = _resolve_agent_id()
            if not aid:
                logger.warning(
                    "No agent_id provided to @govern and TRACCIA_AGENT_ID is not set. "
                    "Skipping policy check."
                )
                return
            check_agent_status(aid, fail_open=fail_open)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _enforce_or_warn()
            with runtime_config.run_identity(
                agent_id=resolved_agent_id,
                agent_name=agent_name,
            ):
                return observed_func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _enforce_or_warn()
            with runtime_config.run_identity(
                agent_id=resolved_agent_id,
                agent_name=agent_name,
            ):
                return await observed_func(*args, **kwargs)

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    if callable(agent_id):
        func = agent_id
        agent_id = None
        fail_open = True
        return decorator(func)

    return decorator
