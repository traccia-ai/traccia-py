#!/usr/bin/env python3
"""
Build-time script: fetch LiteLLM model_prices_and_context_window.json,
normalize it into Traccia's per-1K-token schema, and write
traccia-py/data/pricing_snapshot.json.

Run before each release:
    python scripts/build_pricing_snapshot.py

The generated file is packaged into the wheel so the SDK ships with
an offline snapshot. Users who need fresher prices can run:
    traccia pricing refresh
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# Where the snapshot lives relative to this script's repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "data" / "pricing_snapshot.json"

# LiteLLM prices are per-token; we store per-1K-token (multiply by 1000).
_PER_TOKEN_TO_PER_1K = 1_000.0


def _normalize_model(model_id: str, entry: dict) -> dict | None:
    """
    Convert one LiteLLM model entry into Traccia's pricing schema.
    Returns None for entries that lack input or output cost (e.g. embeddings
    that only report cost-per-image, or models without listed pricing).
    """
    input_cpt = entry.get("input_cost_per_token")
    output_cpt = entry.get("output_cost_per_token")

    # Must have at least one meaningful rate to be useful for cost calculation.
    if input_cpt is None and output_cpt is None:
        return None

    model_entry: dict = {}

    if input_cpt is not None:
        model_entry["prompt"] = round(float(input_cpt) * _PER_TOKEN_TO_PER_1K, 9)
    if output_cpt is not None:
        model_entry["completion"] = round(float(output_cpt) * _PER_TOKEN_TO_PER_1K, 9)

    # Optional fields
    if entry.get("cache_creation_input_token_cost") is not None:
        model_entry["cache_write"] = round(
            float(entry["cache_creation_input_token_cost"]) * _PER_TOKEN_TO_PER_1K, 9
        )
    if entry.get("cache_read_input_token_cost") is not None:
        model_entry["cached_prompt"] = round(
            float(entry["cache_read_input_token_cost"]) * _PER_TOKEN_TO_PER_1K, 9
        )
    if entry.get("input_cost_per_audio_token") is not None:
        model_entry["input_audio"] = round(
            float(entry["input_cost_per_audio_token"]) * _PER_TOKEN_TO_PER_1K, 9
        )
    if entry.get("output_cost_per_audio_token") is not None:
        model_entry["output_audio"] = round(
            float(entry["output_cost_per_audio_token"]) * _PER_TOKEN_TO_PER_1K, 9
        )
    if entry.get("input_cost_per_image_token") is not None:
        model_entry["input_image"] = round(
            float(entry["input_cost_per_image_token"]) * _PER_TOKEN_TO_PER_1K, 9
        )

    # Metadata we store for informational purposes
    if entry.get("litellm_provider"):
        model_entry["_provider"] = entry["litellm_provider"]
    if entry.get("mode"):
        model_entry["_mode"] = entry["mode"]
    if entry.get("max_tokens"):
        model_entry["_max_tokens"] = entry["max_tokens"]
    if entry.get("max_input_tokens"):
        model_entry["_max_input_tokens"] = entry["max_input_tokens"]

    return model_entry


def build(url: str = LITELLM_URL, output: Path = SNAPSHOT_PATH) -> None:
    print(f"Fetching {url} …")
    req = urllib.request.Request(url, headers={"User-Agent": "traccia-build/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_body = resp.read()

    raw = json.loads(raw_body)
    print(f"  Fetched {len(raw_body):,} bytes — {len(raw)} raw model entries")

    models: dict[str, dict] = {}
    skipped = 0
    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            skipped += 1
            continue
        normalized = _normalize_model(model_id, entry)
        if normalized is None:
            skipped += 1
            continue
        models[model_id] = normalized

    print(f"  Normalized {len(models)} models (skipped {skipped} without pricing)")

    snapshot = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "litellm",
        "source_url": url,
        "models": models,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Written to {output}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else LITELLM_URL
    build(url=url)
