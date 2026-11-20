#!/usr/bin/env python3
"""
Refresh data/pricing_snapshot.json from LiteLLM's model_prices_and_context_window.json,
normalized into Traccia's per-1K-token schema.

The snapshot is committed and packaged into the wheel as-is; releases never fetch it.
The "Refresh pricing snapshot" workflow runs this weekly and opens a pull request,
so price changes are reviewed and tested before they ship. To run it by hand:
    python scripts/build_pricing_snapshot.py                 # latest LiteLLM main commit
    python scripts/build_pricing_snapshot.py --ref <sha>     # a specific LiteLLM commit

The source is always pinned to one LiteLLM commit (recorded as source_ref). The run fails
if the result looks broken (too few models, key models missing or unpriced). Models LiteLLM
no longer lists keep their last known price (people still call retired models), listed
in the snapshot's "retained" field; pass --drop-removed to prune them. traccia-node has the same script
(scripts/build-pricing-snapshot.mjs) with the same checks.

Users who need fresher prices can run:
    traccia pricing refresh
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Repo root is the traccia package source tree (see pyproject package-dir).
# When run as `python scripts/build_pricing_snapshot.py` the package is not
# installed yet, so import the module directly from source.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pricing_normalizer import normalize

LITELLM_REPO = "BerriAI/litellm"
LITELLM_FILE = "model_prices_and_context_window.json"

# Where the snapshot lives relative to this script's repo root.
SNAPSHOT_PATH = REPO_ROOT / "data" / "pricing_snapshot.json"

# Sanity checks: a snapshot failing these is not written.
MIN_MODELS = 1000
MAX_MODEL_DROP = 0.10  # fail if more than 10% of the current models disappear
# Long-lived names from each major provider; update when LiteLLM retires one.
REQUIRED_MODELS = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-5", "claude-haiku-4-5", "gemini/gemini-2.5-flash"]
BIG_CHANGE = 0.5  # list price changes above 50% in the summary

_SHA = re.compile(r"^[0-9a-f]{40}$")


def _get(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "traccia-build/1.0"}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:
        return resp.read()


def resolve_ref(ref: str) -> str:
    """Turn a branch or tag of the LiteLLM repo into the full commit SHA it points at."""
    if _SHA.match(ref):
        return ref
    sha = _get(f"https://api.github.com/repos/{LITELLM_REPO}/commits/{ref}", accept="application/vnd.github.sha")
    sha = sha.decode().strip()
    if not _SHA.match(sha):
        raise SystemExit(f"Could not resolve LiteLLM ref {ref!r} to a commit SHA")
    return sha


def cost_fields(rates: dict) -> dict:
    """The price fields of one model (fields starting with "_" are metadata)."""
    return {k: v for k, v in rates.items() if not k.startswith("_")}


def _valid_price(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def check(models: dict, previous: dict | None) -> None:
    """Raise SystemExit if the new snapshot looks broken.

    previous is the last LiteLLM model set, without the models we retained ourselves,
    so retained models don't count towards the allowed drop.
    """
    problems = []
    if len(models) < MIN_MODELS:
        problems.append(f"only {len(models)} models (expected at least {MIN_MODELS})")
    if previous and len(models) < len(previous) * (1 - MAX_MODEL_DROP):
        problems.append(f"model count fell from {len(previous)} to {len(models)}")
    for name in REQUIRED_MODELS:
        rates = models.get(name)
        if not rates or not rates.get("prompt") or not rates.get("completion"):
            problems.append(f"{name} is missing or has no prompt/completion price")
    bad = [f"{name}.{field}={value!r}" for name, rates in models.items()
           for field, value in cost_fields(rates).items() if not _valid_price(value)]
    if bad:
        problems.append(f"{len(bad)} prices are not a finite number >= 0: "
                        f"{', '.join(bad[:10])}{' …' if len(bad) > 10 else ''}")
    if problems:
        raise SystemExit("Pricing snapshot sanity check failed:\n  - " + "\n  - ".join(problems))


def summarize(models: dict, previous: dict, source_url: str, kept: list) -> str:
    """Markdown summary of what changed (used as the pull request body)."""
    added = sorted(set(models) - set(previous))
    removed = sorted(set(previous) - set(models))
    changed, big = [], []
    for name in sorted(set(models) & set(previous)):
        old_rates, new_rates = cost_fields(previous[name]), cost_fields(models[name])
        for field in sorted(set(old_rates) | set(new_rates)):
            old, new = old_rates.get(field), new_rates.get(field)
            if old == new:
                continue
            changed.append(name)
            # A price going to or from zero is always worth a look.
            if old is not None and new is not None and (not old or not new or abs(new - old) / old > BIG_CHANGE):
                big.append(f"| `{name}` | {field} | {old} | {new} |")
    lines = [
        f"Source: {source_url}",
        "",
        f"- Models: {len(previous)} → {len(models)} ({len(added)} added, {len(removed)} removed)",
        f"- Models with a changed price (any field): {len(set(changed))}",
    ]
    if kept:
        lines.append(f"- Kept at their last known price (LiteLLM no longer lists them): {len(kept)}")
    if removed:
        lines.append("- Removed (traces of these models will no longer get a cost): "
                     f"{', '.join(f'`{m}`' for m in removed[:50])}{' …' if len(removed) > 50 else ''}")
    if big:
        lines += ["", f"Price changes over {int(BIG_CHANGE * 100)}% or to/from zero (check these):", "",
                  "| Model | Field | Old (per 1K) | New (per 1K) |", "|---|---|---|---|", *big[:100]]
    return "\n".join(lines) + "\n"


def build(ref: str = "main", url: str | None = None, output: Path = SNAPSHOT_PATH,
          summary: Path | None = None, drop_removed: bool = False) -> bool:
    """Write a new snapshot if prices changed. Returns True if the file was written."""
    source_ref = None
    if url is None:
        source_ref = resolve_ref(ref)
        url = f"https://raw.githubusercontent.com/{LITELLM_REPO}/{source_ref}/{LITELLM_FILE}"
    print(f"Fetching {url} …")
    raw_body = _get(url)
    raw = json.loads(raw_body)
    print(f"  Fetched {len(raw_body):,} bytes — {len(raw)} raw model entries")

    models = normalize(raw)
    print(f"  Normalized {len(models)} models (skipped {len(raw) - len(models)} without pricing)")

    previous, retained = None, []
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        previous, retained = old.get("models"), old.get("retained", [])
    upstream_before = {k: v for k, v in previous.items() if k not in set(retained)} if previous else None
    check(models, upstream_before)

    kept = [] if drop_removed or not previous else sorted(set(previous) - set(models))
    models.update({name: previous[name] for name in kept})

    if summary is not None:
        summary.write_text(summarize(models, previous or {}, url, kept), encoding="utf-8")
    if models == previous and kept == sorted(retained):
        print("  Prices unchanged; snapshot left as is")
        return False

    snapshot = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "litellm",
        "source_url": url,
        "source_ref": source_ref,
        "models": models,
        # Models LiteLLM no longer lists, kept at their last known price.
        "retained": kept,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"  Written to {output}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="LiteLLM branch, tag or commit SHA (default: main)")
    parser.add_argument("--url", help="read this URL instead of LiteLLM on GitHub (for testing)")
    parser.add_argument("--summary", type=Path, help="write a Markdown summary of the changes here")
    parser.add_argument("--drop-removed", action="store_true",
                        help="remove models LiteLLM no longer lists instead of keeping their last price")
    args = parser.parse_args()
    build(ref=args.ref, url=args.url, summary=args.summary, drop_removed=args.drop_removed)
