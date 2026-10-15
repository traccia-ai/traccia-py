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
import re
from typing import Any, Dict, List, Optional, Tuple

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

# llm.vendor values set only by that provider's own SDK, which calls no other provider. Their
# calls are priced only from that provider's entries, never from another provider's entry for
# the same model name, which can be priced very differently. The platform's pricing matcher
# applies the same rule.
STRICT_VENDORS = frozenset({"groq"})

_REGION = re.compile(r"^(?:us|eu|apac|au|ca|jp|global|us-gov)\.")
_SUFFIX = re.compile(r"(?:-\d{4}-\d{2}-\d{2}|-\d{8}|-latest)$")


def _name(name: str) -> str:
    """Lower-cased, without the provider path or region; the date is kept."""
    return _REGION.sub("", name.strip().lower().rsplit("/", 1)[-1])


def _base_name(name: str) -> str:
    return _SUFFIX.sub("", _name(name))


def _strict_vendor(vendor: Optional[str]) -> Optional[str]:
    v = str(vendor or "").strip().lower()
    return v if v in STRICT_VENDORS else None


def _in_vendor_family(key: str, entry: Any, vendor: str) -> bool:
    """Whether the entry is the vendor's: its provider is the vendor, or a variant after "_"
    or "-" (as the platform's matcher counts provider families)."""
    provider = (entry.get("_provider") or entry.get("provider")) if isinstance(entry, dict) else None
    if not provider and "/" in key:
        provider = key.split("/", 1)[0]
    p = str(provider or "").lower()
    return p == vendor or p.startswith(vendor + "_") or p.startswith(vendor + "-")


# The last table indexed for a vendor: (table, its size, vendor, base name -> keys).
_vendor_index: Tuple[Any, int, Optional[str], Dict[str, List[str]]] = (None, 0, None, {})


def _vendor_keys(table: Dict[str, Dict[str, Any]], vendor: str) -> Dict[str, List[str]]:
    global _vendor_index
    cached, size, cached_vendor, index = _vendor_index
    if cached is table and size == len(table) and cached_vendor == vendor:
        return index
    index = {}
    for key, entry in table.items():
        if _in_vendor_family(key, entry, vendor):
            index.setdefault(_base_name(key), []).append(key)
    _vendor_index = (table, len(table), vendor, index)
    return index


def _lookup_vendor_price(
    model: str, table: Dict[str, Dict[str, Any]], vendor: str
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """The vendor's own entry for *model*: a vendor key with the same base name (lower-cased,
    without the provider path, a region, or a date or -latest suffix), the key for the exact
    dated name first, then the shortest. None when the vendor has no such key."""
    wanted = _name(str(model))
    keys = _vendor_keys(table, vendor).get(_SUFFIX.sub("", wanted)) if wanted else None
    if not keys:
        return None
    key = min(keys, key=lambda k: (_name(k) != wanted, len(k), k))
    return key, table[key]


def _lookup_price(
    model: str, table: Dict[str, Dict[str, Any]], vendor: Optional[str] = None
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Return (matched_key, price_dict) for a given model name.

    A call from a vendor in STRICT_VENDORS is priced only from that vendor's entries (see
    _lookup_vendor_price). Otherwise the vendor is not used, and matching supports:
    - Exact match (case-sensitive)
    - Lowercase match
    - Longest-prefix match for version-suffixed names:
        e.g. "claude-3-opus-20240229" -> "claude-3-opus"
             "gpt-4o-2024-08-06"      -> "gpt-4o"
    """
    if not model:
        return None
    m = str(model).strip()
    if not m:
        return None
    strict = _strict_vendor(vendor)
    if strict:
        return _lookup_vendor_price(m, table, strict)

    if m in table:
        return m, table[m]
    ml = m.lower()
    if ml in table:
        return ml, table[ml]

    # Provider-prefixed keys: grok-4.7 -> xai/grok-4.7, before prefix so grok-4 does not steal grok-4.7.
    suffix_hits = []
    for key, value in table.items():
        tail = key.replace("\\", "/").rsplit("/", 1)[-1]
        dot = tail.find(".")
        if dot > 0:
            head, rest = tail[:dot], tail[dot + 1 :]
            if head.replace("-", "").isalpha() and "-" in rest:
                tail = rest
        if tail.lower() == ml:
            suffix_hits.append((key, value))
    if suffix_hits:
        suffix_hits.sort(key=lambda item: len(item[0]))
        return suffix_hits[0]

    # Longest-key prefix wins to avoid "gpt-4" matching "gpt-4o".
    for key in sorted(table.keys(), key=len, reverse=True):
        if ml.startswith(key.lower()):
            return key, table[key]

    return None


def match_pricing_model_key(
    model: str,
    pricing_table: Optional[Dict[str, Dict[str, Any]]] = None,
    vendor: Optional[str] = None,
) -> Optional[str]:
    """Return the pricing table key that would be used for *model*, if any."""
    table = pricing_table if pricing_table is not None else BUNDLED_PRICING
    matched = _lookup_price(model, table, vendor)
    if not matched:
        return None
    key, _ = matched
    return key


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
    ``cache_fallback`` is true. *vendor* is the span's ``llm.vendor`` (see
    STRICT_VENDORS).
    """
    table = pricing_table if pricing_table is not None else BUNDLED_PRICING
    matched = _lookup_price(model, table, vendor)
    if not matched:
        return None
    _, price = matched
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
    return {"cost": round(cost, 6), "cache_fallback": cache_fallback}


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
        vendor: The span's ``llm.vendor``; a vendor in STRICT_VENDORS is priced only
            from its own entries.

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
