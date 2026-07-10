"""Runtime governance configuration (advanced endpoint overrides)."""

from __future__ import annotations

from typing import Optional

from traccia.config import find_config_file, load_toml_config


class GovernanceConfig:
    """Optional overrides for policy API endpoints and cache behavior."""

    status_check_endpoint: Optional[str] = None
    post_block_endpoint: Optional[str] = None
    status_cache_ttl_seconds: int = 60


gov_config = GovernanceConfig()


def configure_governance(
    *,
    status_check_endpoint: Optional[str] = None,
    post_block_endpoint: Optional[str] = None,
    status_cache_ttl_seconds: Optional[int] = None,
    config_file: Optional[str] = None,
) -> None:
    """
    Apply governance settings from init kwargs and optional [governance] in traccia.toml.

    Endpoint overrides are advanced-only. By default, status and block URLs are derived
    from the tracing endpoint base URL.
    """
    toml_path = config_file or find_config_file()
    gov_section = {}
    if toml_path:
        try:
            toml_data = load_toml_config(toml_path)
            gov_section = toml_data.get("governance", {}) or {}
        except Exception:
            pass

    if status_check_endpoint is not None:
        gov_config.status_check_endpoint = status_check_endpoint
    elif gov_section.get("status_check_endpoint"):
        gov_config.status_check_endpoint = gov_section["status_check_endpoint"]

    if post_block_endpoint is not None:
        gov_config.post_block_endpoint = post_block_endpoint
    elif gov_section.get("post_block_endpoint"):
        gov_config.post_block_endpoint = gov_section["post_block_endpoint"]

    ttl = status_cache_ttl_seconds
    if ttl is None:
        ttl = gov_section.get("status_cache_ttl_seconds")
    if ttl is not None:
        gov_config.status_cache_ttl_seconds = int(ttl)
