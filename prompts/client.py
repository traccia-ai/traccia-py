"""HTTP client for prompt-runtime fetch."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests

from traccia.config import DEFAULT_OTLP_TRACE_ENDPOINT, load_config

logger = logging.getLogger("traccia.prompts")

DEFAULT_RUNTIME_PATH = "/api/v1/prompt-runtime/prompts/{name}"


class PromptFetchError(RuntimeError):
    """Raised when the prompt runtime API is unreachable or returns an error."""


def derive_base_url(traces_endpoint: str) -> str:
    parsed = urlparse(traces_endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


def resolve_credentials(
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    prompt_api_base: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (api_key, base_url) from args, env, or SDK config.

    Prefer prompt_api_base / TRACCIA_PROMPT_API_BASE when set; otherwise derive from the traces endpoint.
    """
    import os

    cfg = load_config()
    key = api_key or (cfg.tracing.api_key if cfg and cfg.tracing else None)
    traces = endpoint or (cfg.tracing.endpoint if cfg and cfg.tracing else None) or DEFAULT_OTLP_TRACE_ENDPOINT
    if not key:
        raise PromptFetchError(
            "Traccia API key not found. Call init(api_key=...) or set TRACCIA_API_KEY before load_prompt."
        )

    base = prompt_api_base
    if not base:
        try:
            from traccia.prompts import get_prompt_api_base

            base = get_prompt_api_base()
        except Exception:
            base = None
    if not base:
        base = os.environ.get("TRACCIA_PROMPT_API_BASE")
    if not base:
        base = derive_base_url(traces)
    return key, str(base).rstrip("/")


def fetch_prompt_runtime(
    name: str,
    *,
    label: Optional[str] = None,
    version: Optional[int] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    prompt_api_base: Optional[str] = None,
    timeout: float = 5.0,
    session: Optional[requests.Session] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """GET /api/v1/prompt-runtime/prompts/{name}. Returns (payload, etag)."""
    if label and version is not None:
        raise ValueError("Pass label or version, not both")

    key, base_url = resolve_credentials(
        api_key=api_key, endpoint=endpoint, prompt_api_base=prompt_api_base
    )
    params: Dict[str, Any] = {}
    if version is not None:
        params["version"] = version
    elif label:
        params["label"] = label
    else:
        params["label"] = "production"

    url = f"{base_url}{DEFAULT_RUNTIME_PATH.format(name=name)}"
    if params:
        url = f"{url}?{urlencode(params)}"

    http = session or requests.Session()
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    try:
        resp = http.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise PromptFetchError(f"Failed to fetch prompt '{name}': {exc}") from exc

    if resp.status_code == 404:
        raise PromptFetchError(f"Prompt '{name}' not found")
    if resp.status_code >= 400:
        detail = resp.text[:200] if resp.text else resp.reason
        raise PromptFetchError(f"Prompt fetch failed ({resp.status_code}): {detail}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise PromptFetchError(f"Invalid JSON from prompt runtime for '{name}'") from exc

    etag = resp.headers.get("ETag")
    return payload, etag
