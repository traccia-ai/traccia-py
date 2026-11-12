"""Policy checks on outbound HTTP requests (TRC-307 Environment Boundary).

With ``check_outbound_http=True`` on init() (or ``TRACCIA_CHECK_OUTBOUND_HTTP=1``),
every HTTP request an agent sends inside ``@govern`` is checked before it goes out,
as an ``http_request`` action. An Environment Boundary policy can then stop an
agent outside production from calling a production URL, even when the call is
not a governed tool.

Every redirect hop is checked, not only the first URL:

- ``requests``: ``Session.send`` runs once per hop (``resolve_redirects`` calls it).
- ``httpx``: ``Client._send_single_request`` / ``AsyncClient._send_single_request``
  run once per hop.

Requests to Traccia itself, and requests made while a check is running, are never
checked. A deny raises ``AgentBlockedError`` and a held approval
``ApprovalPending`` before anything is sent. When Traccia gives no answer, the
policy fail mode applies, as for every other check.
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Optional
from urllib.parse import urlsplit

from traccia import runtime_config

logger = logging.getLogger("traccia.governance")

CONFIG_KEY = "check_outbound_http"
_TRUTHY = {"1", "true", "yes", "on"}
_checking: contextvars.ContextVar[bool] = contextvars.ContextVar("traccia_outbound_checking", default=False)
_patched = {"requests": False, "httpx": False}


def enabled() -> bool:
    configured = runtime_config.get_config_value(CONFIG_KEY)
    if configured is not None:
        return bool(configured)
    return os.environ.get("TRACCIA_CHECK_OUTBOUND_HTTP", "").strip().lower() in _TRUTHY


def set_enabled(value: bool) -> None:
    runtime_config.set_config_value(CONFIG_KEY, bool(value))
    if value:
        patch_outbound_http()


def _traccia_host() -> Optional[str]:
    try:
        from traccia import client_config

        _key, endpoint = client_config.resolve_credentials()
        return (urlsplit(endpoint).hostname or "").lower() if endpoint else None
    except Exception:
        return None


def should_check(url: str) -> bool:
    if _checking.get() or not enabled() or not runtime_config.pep_enabled():
        return False
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return False
    traccia = _traccia_host()
    return host != traccia


def check_request(method: str, url: str) -> None:
    """Check one outbound request (one hop). Raises when the call must not be sent."""
    if not should_check(url):
        return
    from traccia.governance.pep import check_policy

    token = _checking.set(True)
    try:
        check_policy(
            action={"type": "http_request", "name": "http_request", "method": str(method or "GET").upper()},
            context={"input": {"url": url, "method": str(method or "GET").upper()}},
        )
    finally:
        _checking.reset(token)


async def check_request_async(method: str, url: str) -> None:
    if not should_check(url):
        return
    import asyncio

    ctx = contextvars.copy_context()
    await asyncio.get_running_loop().run_in_executor(None, ctx.run, check_request, method, url)


def _patch_requests() -> bool:
    if _patched["requests"]:
        return True
    try:
        import requests
    except Exception:
        return False
    original = requests.Session.send
    if getattr(original, "_traccia_outbound", False):
        _patched["requests"] = True
        return True

    def send(self: Any, request: Any, **kwargs: Any):
        check_request(getattr(request, "method", "GET"), str(getattr(request, "url", "")))
        return original(self, request, **kwargs)

    send._traccia_outbound = True  # type: ignore[attr-defined]
    requests.Session.send = send
    _patched["requests"] = True
    return True


def _patch_httpx() -> bool:
    if _patched["httpx"]:
        return True
    try:
        import httpx
    except Exception:
        return False
    sync_original = getattr(httpx.Client, "_send_single_request", None)
    async_original = getattr(httpx.AsyncClient, "_send_single_request", None)
    if sync_original is None or async_original is None:
        logger.warning("httpx version not supported for outbound policy checks")
        return False
    if not getattr(sync_original, "_traccia_outbound", False):
        def send_single(self: Any, request: Any, *args: Any, **kwargs: Any):
            check_request(request.method, str(request.url))
            return sync_original(self, request, *args, **kwargs)

        send_single._traccia_outbound = True  # type: ignore[attr-defined]
        httpx.Client._send_single_request = send_single
    if not getattr(async_original, "_traccia_outbound", False):
        async def send_single_async(self: Any, request: Any, *args: Any, **kwargs: Any):
            await check_request_async(request.method, str(request.url))
            return await async_original(self, request, *args, **kwargs)

        send_single_async._traccia_outbound = True  # type: ignore[attr-defined]
        httpx.AsyncClient._send_single_request = send_single_async
    _patched["httpx"] = True
    return True


def patch_outbound_http() -> dict:
    """Install the checks on the HTTP clients that are installed. Safe to call twice."""
    return {"requests": _patch_requests(), "httpx": _patch_httpx()}
