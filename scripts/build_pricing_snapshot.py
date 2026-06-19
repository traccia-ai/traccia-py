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

from traccia.pricing_normalizer import normalize

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# Where the snapshot lives relative to this script's repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "data" / "pricing_snapshot.json"


def build(url: str = LITELLM_URL, output: Path = SNAPSHOT_PATH) -> None:
    print(f"Fetching {url} …")
    req = urllib.request.Request(url, headers={"User-Agent": "traccia-build/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_body = resp.read()

    raw = json.loads(raw_body)
    print(f"  Fetched {len(raw_body):,} bytes — {len(raw)} raw model entries")

    models = normalize(raw)
    skipped = len(raw) - len(models)

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
