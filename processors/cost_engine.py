"""Cost calculation based on model pricing and token usage.

Pricing is loaded from the bundled snapshot shipped with the SDK
(data/pricing_snapshot.json). The snapshot is generated at release time from
LiteLLM's model_prices_and_context_window.json via scripts/build_pricing_snapshot.py.

Users who need fresher prices can run:
    traccia pricing refresh          # fetch from Traccia platform
    traccia pricing refresh --source litellm  # fetch directly from upstream
"""

from __future__ import annotations

import importlib.resources
import json
import logging
from typing import Any, Dict, Optional, Tuple

from traccia.pricing_matcher import match_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the bundled pricing snapshot once at import time
# ---------------------------------------------------------------------------

def _load_bundled_snapshot() -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Load pricing_snapshot.json shipped inside the traccia.data package.
    Returns (models_dict, generated_at_iso_string).
    Falls back to a minimal hardcoded table if the file is missing.
    """
    try:
        # Python 3.9+ supports importlib.resources.files()
        pkg = importlib.resources.files("traccia.data")  # type: ignore[attr-defined]
        data = (pkg / "pricing_snapshot.json").read_text(encoding="utf-8")
        snapshot = json.loads(data)
        models = snapshot.get("models", {})
        generated_at = snapshot.get("generated_at", "unknown")
        logger.debug(
            "Loaded bundled pricing snapshot (%d models, generated_at=%s)",
            len(models),
            generated_at,
        )
        return models, generated_at
    except Exception as exc:
        logger.warning(
            "Could not load bundled pricing snapshot (%s); falling back to minimal table.",
            exc,
        )
        # Minimal fallback so cost calculation never hard-fails
        fallback = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
        }
        return fallback, "unknown"


# Module-level singletons — loaded once, held for process lifetime.
BUNDLED_PRICING, BUNDLED_PRICING_GENERATED_AT = _load_bundled_snapshot()

# Backward-compat alias (deprecated name; kept for external code that imports it directly).
DEFAULT_PRICING = BUNDLED_PRICING


# ---------------------------------------------------------------------------
# Model-name matching
# ---------------------------------------------------------------------------

def _lookup_price(
    model: str, table: Dict[str, Dict[str, Any]], vendor: Optional[str] = None
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (matched_key, price_dict) for a given model name; see traccia.pricing_matcher."""
    matched = match_model(model, table, vendor)
    if matched is None:
        return None
    return matched.key, table[matched.key]


def match_pricing_model_key(
    model: str,
    pricing_table: Optional[Dict[str, Dict[str, Any]]] = None,
    vendor: Optional[str] = None,
) -> Optional[str]:
    """Return the pricing table key that would be used for *model*, if any."""
    table = pricing_table if pricing_table is not None else BUNDLED_PRICING
    matched = match_model(model, table, vendor)
    return matched.key if matched else None


# ---------------------------------------------------------------------------
# Core cost formula
# ---------------------------------------------------------------------------

def compute_cost_detail(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_table: Optional[Dict[str, Dict[str, Any]]] = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    vendor: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Price uncached input, cache read, cache write, and output.

    ``cached_prompt`` and ``cache_write`` come from the LiteLLM snapshot.
    When a cache rate is missing, that portion uses the input rate and
    ``cache_fallback`` is true. *vendor* (the span's ``llm.vendor``) prefers that
    provider's own price when several pricing keys fit the model name.

    The result also names the matched key (``model_key``), how it matched
    (``match_kind``: exact | alias | undated | prefix) and its ``provider``.
    """
    table = pricing_table if pricing_table is not None else BUNDLED_PRICING
    matched = match_model(model, table, vendor)
    if not matched:
        return None
    price = table[matched.key]
    cost = (prompt_tokens / 1_000.0) * price.get("prompt", 0.0)
    cost += (completion_tokens / 1_000.0) * price.get("completion", 0.0)
    cache_fallback = False
    if cache_read_tokens:
        rate = price.get("cached_prompt")
        if rate is None:
            rate = price.get("prompt", 0.0)
            cache_fallback = True
        cost += (cache_read_tokens / 1_000.0) * float(rate)
    if cache_write_tokens:
        rate = price.get("cache_write")
        if rate is None:
            rate = price.get("prompt", 0.0)
            cache_fallback = True
        cost += (cache_write_tokens / 1_000.0) * float(rate)
    return {
        "cost": round(cost, 6),
        "cache_fallback": cache_fallback,
        "model_key": matched.key,
        "match_kind": matched.kind,
        "provider": matched.provider,
    }


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_table: Optional[Dict[str, Dict[str, Any]]] = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    vendor: Optional[str] = None,
) -> Optional[float]:
    """
    Estimate cost in USD for one LLM call.

    Args:
        model: Model identifier (e.g. "gpt-4o", "claude-3-opus-20240229").
        prompt_tokens: Number of uncached input tokens.
        completion_tokens: Number of output tokens.
        pricing_table: Optional pricing table to use; defaults to BUNDLED_PRICING.
        cache_read_tokens: Tokens served from cache.
        cache_write_tokens: Tokens written to cache.
        vendor: The span's ``llm.vendor``; prefers that provider's own price.

    Returns:
        Estimated cost in USD rounded to 6 decimal places, or None if the
        model is not found in the pricing table.
    """
    detail = compute_cost_detail(
        model,
        prompt_tokens,
        completion_tokens,
        pricing_table=pricing_table,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        vendor=vendor,
    )
    if detail is None:
        return None
    return detail["cost"]
