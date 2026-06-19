"""Normalise LiteLLM's raw schema into Traccia's per-1K-token pricing schema.

LiteLLM stores cost *per single token*; Traccia stores cost *per 1,000 tokens*.
Shared by the SDK snapshot builder, tests, and platform pricing refresh.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

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
