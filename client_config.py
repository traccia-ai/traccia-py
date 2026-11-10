"""Effective client configuration shared by tracing, policy, prompts and evals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ClientConfig:
    """API key and traces endpoint resolved once by init(); what the OTLP exporter uses."""

    api_key: Optional[str]
    endpoint: Optional[str]


_client_config: Optional[ClientConfig] = None


def set_client_config(api_key: Optional[str], endpoint: Optional[str]) -> None:
    global _client_config
    _client_config = ClientConfig(api_key=api_key or None, endpoint=endpoint or None)


def get_client_config() -> Optional[ClientConfig]:
    return _client_config


def clear_client_config() -> None:
    global _client_config
    _client_config = None


def resolve_credentials(
    api_key: Optional[str] = None, endpoint: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Return (api_key, traces_endpoint).

    An explicit argument wins for its field. Otherwise the key and endpoint come
    as a pair from one source: the init() config when it has a key, else
    env/config file. The init() endpoint may be the default, so it is never
    paired with a key from another source.
    """
    effective = _client_config
    if effective and effective.api_key:
        return api_key or effective.api_key, endpoint or effective.endpoint

    from traccia.config import load_config

    cfg = load_config()
    key = api_key or cfg.tracing.api_key or os.environ.get("TRACCIA_API_KEY")
    traces = endpoint or cfg.tracing.endpoint or os.environ.get("TRACCIA_ENDPOINT")
    return key, traces
