"""Pricing configuration loader with 4-level resolution.

Resolution order (highest precedence first):
  1. override   — programmatic dict passed to start_tracing(pricing_override=…)
  2. env        — TRACCIA_PRICING_OVERRIDE_JSON env var (AGENT_DASHBOARD_PRICING_JSON
                  is accepted as a deprecated alias)
  3. local_cache — ~/.cache/traccia/pricing.json written by `traccia pricing refresh`
  4. bundled    — data/pricing_snapshot.json shipped inside the SDK wheel

The `generated_at` timestamp travels with the table so staleness can be surfaced
in logs and span attributes without an extra DB or file lookup.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from traccia.processors.cost_engine import BUNDLED_PRICING, BUNDLED_PRICING_GENERATED_AT

logger = logging.getLogger(__name__)

PricingSource = Literal["bundled", "local_cache", "env", "override"]

# Default cache location — can be overridden by TRACCIA_PRICING_CACHE_PATH.
_DEFAULT_CACHE_PATH = Path.home() / ".cache" / "traccia" / "pricing.json"


def _cache_path() -> Path:
    env = os.getenv("TRACCIA_PRICING_CACHE_PATH")
    return Path(env) if env else _DEFAULT_CACHE_PATH


# ---------------------------------------------------------------------------
# Layer loaders
# ---------------------------------------------------------------------------

def _load_bundled() -> Tuple[Dict[str, Dict[str, Any]], PricingSource, str]:
    return BUNDLED_PRICING.copy(), "bundled", BUNDLED_PRICING_GENERATED_AT


def _load_local_cache() -> Optional[Tuple[Dict[str, Dict[str, Any]], PricingSource, str]]:
    """
    Load the user's locally-cached pricing file (written by `traccia pricing refresh`).
    Returns None if the file is absent or corrupt.
    """
    path = _cache_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = raw.get("models")
        if not isinstance(models, dict) or not models:
            return None
        generated_at = raw.get("generated_at", "unknown")
        return models, "local_cache", generated_at
    except Exception as exc:
        logger.debug("Could not load local pricing cache (%s): %s", path, exc)
        return None


def _load_env_override() -> Optional[Tuple[Dict[str, Dict[str, Any]], str]]:
    """
    Read TRACCIA_PRICING_OVERRIDE_JSON (or the legacy AGENT_DASHBOARD_PRICING_JSON alias).
    Returns (models_dict, env_var_name) or None.
    """
    # Preferred name first
    raw_json = os.getenv("TRACCIA_PRICING_OVERRIDE_JSON")
    var_name = "TRACCIA_PRICING_OVERRIDE_JSON"

    if raw_json is None:
        raw_json = os.getenv("AGENT_DASHBOARD_PRICING_JSON")
        var_name = "AGENT_DASHBOARD_PRICING_JSON"
        if raw_json is not None:
            logger.warning(
                "AGENT_DASHBOARD_PRICING_JSON is deprecated; "
                "rename it to TRACCIA_PRICING_OVERRIDE_JSON. "
                "The old name will be removed in a future minor version."
            )

    if not raw_json:
        return None

    try:
        data = json.loads(raw_json)
        if isinstance(data, dict) and data:
            return data, var_name
    except Exception as exc:
        logger.warning("Could not parse %s as JSON: %s", var_name, exc)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pricing_with_source(
    override: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], PricingSource, str]:
    """
    Return (pricing_table, source, generated_at) using the 4-level resolution chain.

    - pricing_table: model-name → {prompt, completion, …} (per-1K-token rates)
    - source:        one of "bundled" | "local_cache" | "env" | "override"
    - generated_at:  ISO timestamp string of the base table (before env/override merge)
    """
    # Start with bundled (lowest precedence)
    pricing, source, generated_at = _load_bundled()

    # Layer 2: local cache (user ran `traccia pricing refresh`)
    cached = _load_local_cache()
    if cached:
        pricing, source, generated_at = cached

    # Layer 3: env override (merges on top, does not replace entirely)
    env_result = _load_env_override()
    if env_result:
        env_pricing, _ = env_result
        pricing = {**pricing, **env_pricing}
        source = "env"

    # Layer 4: programmatic override (always wins; merges on top)
    if override:
        if not isinstance(override, dict):
            logger.warning("pricing_override must be a dict; ignoring.")
        else:
            pricing = {**pricing, **override}
            source = "override"

    return pricing, source, generated_at


def load_pricing(
    override: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible helper returning only the pricing table."""
    table, _, _ = load_pricing_with_source(override)
    return table


# ---------------------------------------------------------------------------
# Local cache helpers (used by the CLI refresh command)
# ---------------------------------------------------------------------------

def write_local_cache(snapshot: Dict[str, Any]) -> Path:
    """
    Persist a pricing snapshot dict to the local cache file.
    snapshot must contain at least {"models": {...}, "generated_at": "..."}.
    Returns the path written.
    """
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def clear_local_cache() -> bool:
    """Delete the local cache file. Returns True if it existed."""
    path = _cache_path()
    if path.exists():
        path.unlink()
        return True
    return False


def local_cache_info() -> Optional[Dict[str, Any]]:
    """
    Return metadata about the local cache without loading the full model dict.
    Returns None if no cache exists.
    """
    path = _cache_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = raw.get("models", {})
        return {
            "path": str(path),
            "generated_at": raw.get("generated_at", "unknown"),
            "source": raw.get("source", "unknown"),
            "source_url": raw.get("source_url"),
            "etag": raw.get("etag"),
            "model_count": len(models),
        }
    except Exception:
        return None


def snapshot_age_days(generated_at: str) -> Optional[float]:
    """
    Return the age in days of a snapshot given its generated_at ISO string.
    Returns None if the string cannot be parsed.
    """
    if not generated_at or generated_at == "unknown":
        return None
    try:
        # Handle both "Z" suffix and "+00:00"
        ts_str = generated_at.replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_str)
        now = datetime.now(timezone.utc)
        delta = now - ts
        return delta.total_seconds() / 86_400.0
    except Exception:
        return None
