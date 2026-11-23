"""Normalise LiteLLM's raw schema into Traccia's per-1K-token pricing schema.

LiteLLM stores cost *per single token*; Traccia stores cost *per 1,000 tokens*.
Shared by the SDK snapshot builder, tests, and platform pricing refresh.
check_snapshot() is the sanity check both the builder and `traccia pricing refresh`
run before saving a snapshot.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_PER_TOKEN_TO_PER_1K = 1_000.0

COST_FIELDS = frozenset(
    {
        "prompt",
        "completion",
        "cache_write",
        "cached_prompt",
        "input_audio",
        "output_audio",
        "input_image",
    }
)


def normalize(raw: dict) -> Dict[str, Dict[str, Any]]:
    """
    Convert a full LiteLLM pricing dict (model_id → raw_entry) into Traccia format.

    Returns:
        Dict mapping model_id → {prompt, completion, [cache_write, cached_prompt, ...]}
        Models without input/output cost are omitted.
    """
    models: Dict[str, Dict[str, Any]] = {}
    skipped = 0

    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            skipped += 1
            continue
        normalized = _normalize_entry(model_id, entry)
        if normalized is None:
            skipped += 1
            continue
        models[model_id] = normalized

    logger.debug("Normalized %d models (skipped %d)", len(models), skipped)
    return models


def _normalize_entry(model_id: str, entry: dict) -> Optional[Dict[str, Any]]:
    input_cpt = entry.get("input_cost_per_token")
    output_cpt = entry.get("output_cost_per_token")

    if input_cpt is None and output_cpt is None:
        return None

    model_entry: Dict[str, Any] = {}

    if input_cpt is not None:
        model_entry["prompt"] = round(float(input_cpt) * _PER_TOKEN_TO_PER_1K, 9)
    if output_cpt is not None:
        model_entry["completion"] = round(float(output_cpt) * _PER_TOKEN_TO_PER_1K, 9)

    _opt(entry, "cache_creation_input_token_cost", "cache_write", model_entry)
    _opt(entry, "cache_read_input_token_cost", "cached_prompt", model_entry)
    _opt(entry, "input_cost_per_audio_token", "input_audio", model_entry)
    _opt(entry, "output_cost_per_audio_token", "output_audio", model_entry)
    _opt(entry, "input_cost_per_image_token", "input_image", model_entry)

    if entry.get("litellm_provider"):
        model_entry["_provider"] = entry["litellm_provider"]
    if entry.get("mode"):
        model_entry["_mode"] = entry["mode"]
    if entry.get("max_tokens"):
        model_entry["_max_tokens"] = entry["max_tokens"]
    if entry.get("max_input_tokens"):
        model_entry["_max_input_tokens"] = entry["max_input_tokens"]
    if entry.get("max_output_tokens"):
        model_entry["_max_output_tokens"] = entry["max_output_tokens"]

    return model_entry


def _opt(entry: dict, src_key: str, dst_key: str, target: dict) -> None:
    val = entry.get(src_key)
    if val is not None:
        target[dst_key] = round(float(val) * _PER_TOKEN_TO_PER_1K, 9)


def diff_models(
    prev: Dict[str, Dict[str, Any]],
    curr: Dict[str, Dict[str, Any]],
    epsilon: float = 1e-9,
) -> Set[str]:
    """
    Return model keys whose pricing rates changed between two normalized dicts.

    Only compares cost fields; metadata prefixed with _ is ignored.
    """
    changed: Set[str] = set()

    all_keys = set(prev) | set(curr)
    for key in all_keys:
        previous = prev.get(key, {})
        current = curr.get(key, {})
        for field in COST_FIELDS:
            previous_value = previous.get(field, 0.0)
            current_value = current.get(field, 0.0)
            if abs(float(previous_value) - float(current_value)) > epsilon:
                changed.add(key)
                break

    return changed


# Sanity checks for a pricing snapshot (see check_snapshot).
MIN_MODELS = 1000
MAX_MODEL_DROP = 0.10  # more than 10% of the previous models disappearing is suspicious
# Long-lived names from each major provider; update when LiteLLM retires one.
REQUIRED_MODELS = ("gpt-4o", "gpt-4o-mini", "claude-sonnet-4-5", "claude-haiku-4-5", "gemini/gemini-2.5-flash")


def cost_fields(rates: Dict[str, Any]) -> Dict[str, Any]:
    """The price fields of one model (fields starting with "_" are metadata)."""
    return {k: v for k, v in rates.items() if not k.startswith("_")}


def _valid_price(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def check_snapshot(
    models: Any,
    previous: Optional[Dict[str, Dict[str, Any]]] = None,
    strict: bool = True,
) -> List[str]:
    """
    Return the problems that make a pricing snapshot unsafe to use (empty list if fine).

    Every price must be a finite number >= 0. With strict=True (traccia's own snapshot)
    it must also have at least MIN_MODELS models, keep the REQUIRED_MODELS priced, and
    not lose more than MAX_MODEL_DROP of the models in previous.
    """
    if not isinstance(models, dict) or not models:
        return ["no models"]
    problems = []
    bad = [f"{name}.{field}={value!r}" for name, rates in models.items() if isinstance(rates, dict)
           for field, value in cost_fields(rates).items() if not _valid_price(value)]
    bad += [f"{name} is not an object" for name, rates in models.items() if not isinstance(rates, dict)]
    if bad:
        problems.append(f"{len(bad)} prices are not a finite number >= 0: "
                        f"{', '.join(bad[:10])}{' …' if len(bad) > 10 else ''}")
    if strict:
        if len(models) < MIN_MODELS:
            problems.append(f"only {len(models)} models (expected at least {MIN_MODELS})")
        if previous and len(models) < len(previous) * (1 - MAX_MODEL_DROP):
            problems.append(f"model count fell from {len(previous)} to {len(models)}")
        for name in REQUIRED_MODELS:
            rates = models.get(name)
            if not isinstance(rates, dict) or not rates.get("prompt") or not rates.get("completion"):
                problems.append(f"{name} is missing or has no prompt/completion price")
    return problems
