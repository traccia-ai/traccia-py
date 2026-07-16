"""Prompt runtime helpers: load_prompt, prefetch_prompts."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from traccia.prompts.cache import PromptCache
from traccia.prompts.client import PromptFetchError, fetch_prompt_runtime
from traccia.prompts.compile import CompileError
from traccia.prompts.prompt import LoadedPrompt

logger = logging.getLogger("traccia.prompts")

_DEFAULT_TTL_S = 60.0
_cache = PromptCache(ttl_seconds=_DEFAULT_TTL_S)
_cache_ttl_s = _DEFAULT_TTL_S
_prompt_api_base: Optional[str] = None
_fetch_impl: Callable[..., Tuple[Dict[str, Any], Optional[str]]] = fetch_prompt_runtime


def configure_prompts(
    *,
    cache_ttl_s: Optional[float] = None,
    prompt_api_base: Optional[str] = None,
) -> None:
    """Update prompt client settings (called from init / start_tracing)."""
    global _cache_ttl_s, _prompt_api_base
    if cache_ttl_s is not None:
        _cache_ttl_s = float(cache_ttl_s)
        _cache.ttl_seconds = _cache_ttl_s
    if prompt_api_base is not None:
        _prompt_api_base = prompt_api_base.rstrip("/") or None


def get_prompt_api_base() -> Optional[str]:
    return _prompt_api_base


def reset_prompt_cache() -> None:
    """Clear the in-process prompt cache (tests / process recycle)."""
    _cache.clear()


def _set_fetch_impl_for_tests(fn: Callable[..., Tuple[Dict[str, Any], Optional[str]]]) -> None:
    global _fetch_impl
    _fetch_impl = fn


def _reset_fetch_impl_for_tests() -> None:
    global _fetch_impl
    _fetch_impl = fetch_prompt_runtime


def load_prompt(
    name: str,
    *,
    label: str = "production",
    version: Optional[int] = None,
    fallback: Optional[Mapping[str, Any]] = None,
    force_refresh: bool = False,
) -> LoadedPrompt:
    """
    Fetch a named prompt from Traccia with TTL cache, SWR, and optional fallback.

    Args:
        name: Prompt name in the workspace library.
        label: Deploy label (default production). Ignored when version is set.
        version: Exact version number (mutually exclusive with label).
        fallback: Body to use when fetch fails and no cached value exists.
        force_refresh: Bypass cache and fetch synchronously.

    Returns:
        LoadedPrompt with .compile(**vars), .config, .version, .is_fallback, .is_stale.
    """
    if version is not None:
        resolved_label: Optional[str] = None
    else:
        resolved_label = label

    key = _cache.make_key(name, label=resolved_label, version=version)

    def do_fetch() -> Tuple[Dict[str, Any], Optional[str]]:
        return _fetch_impl(name, label=resolved_label, version=version)

    if not force_refresh:
        cached = _cache.get(key)
        if cached is not None:
            entry, is_fresh = cached
            if is_fresh:
                return LoadedPrompt.from_payload(entry.payload, is_stale=False)
            # Stale-while-revalidate: serve last good, refresh in background
            _cache.stale_while_revalidate(key, do_fetch)
            logger.info("prompt_stale_served: serving cached prompt for %s", key)
            return LoadedPrompt.from_payload(entry.payload, is_stale=True)

    try:
        payload, etag = do_fetch()
        _cache.set(key, payload, etag=etag)
        return LoadedPrompt.from_payload(payload, is_stale=False)
    except Exception as exc:
        cached = _cache.get(key)
        if cached is not None:
            entry, _ = cached
            logger.warning("prompt_stale_served: fetch failed (%s); using last good cache", exc)
            return LoadedPrompt.from_payload(entry.payload, is_stale=True)

        if fallback is not None:
            logger.warning("load_prompt fallback used for '%s': %s", name, exc)
            return LoadedPrompt.from_fallback(name, fallback, label=resolved_label)

        if isinstance(exc, PromptFetchError):
            raise
        raise PromptFetchError(str(exc)) from exc


def prefetch_prompts(
    names: Sequence[str],
    *,
    label: str = "production",
    jitter_s: float = 1.0,
) -> List[LoadedPrompt]:
    """
    Warm the cache at startup. Optional jitter spreads multi-replica stampede.
    """
    results: List[LoadedPrompt] = []
    for i, name in enumerate(names):
        if i > 0 and jitter_s > 0:
            time.sleep(random.uniform(0, jitter_s))
        results.append(load_prompt(name, label=label, force_refresh=True))
    return results


__all__ = [
    "load_prompt",
    "prefetch_prompts",
    "configure_prompts",
    "reset_prompt_cache",
    "LoadedPrompt",
    "CompileError",
    "PromptFetchError",
]
