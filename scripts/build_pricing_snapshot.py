#!/usr/bin/env python3
"""
Refresh data/pricing_snapshot.json from LiteLLM's model_prices_and_context_window.json,
normalized into Traccia's per-1K-token schema.

The snapshot is committed and packaged into the wheel as-is; releases never fetch it.
Run this as part of every release, in the version bump PR, so each release ships fresh,
reviewed prices (the publish workflow refuses a snapshot older than 14 days):
    python scripts/build_pricing_snapshot.py                 # latest LiteLLM main commit
    python scripts/build_pricing_snapshot.py --ref <sha>     # a specific LiteLLM commit

The source is always pinned to one LiteLLM commit (recorded as source_ref). The run fails
if the result looks broken (pricing_normalizer.check_snapshot: too few models, key models
missing or unpriced, invalid prices). Models LiteLLM no longer lists keep their last known
price (people still call retired models), listed in the snapshot's "retained" field; pass
--drop-removed to prune them. traccia-node copies this snapshot
(scripts/build-pricing-snapshot.mjs) and `traccia pricing refresh` downloads it from main.

Users who need fresher prices can run:
    traccia pricing refresh
"""

from __future__ import annotations

import argparse
import json
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

from pricing_normalizer import check_snapshot, cost_fields, normalize

LITELLM_REPO = "BerriAI/litellm"
LITELLM_FILE = "model_prices_and_context_window.json"

# Where the snapshot lives relative to this script's repo root.
SNAPSHOT_PATH = REPO_ROOT / "data" / "pricing_snapshot.json"

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


def check(models: dict, previous: dict | None) -> None:
    """Raise SystemExit if the new snapshot looks broken (see pricing_normalizer.check_snapshot).

    previous is the last LiteLLM model set, without the models we retained ourselves,
    so retained models don't count towards the allowed drop.
    """
    problems = check_snapshot(models, previous)
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
    """Write a new snapshot. Returns True if any price changed."""
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
    # Written even when prices are unchanged: generated_at records when the prices were
    # last checked against LiteLLM, which is what the release age check looks at.
    changed = models != previous or kept != sorted(retained)
    if not changed:
        print("  Prices unchanged; updating only generated_at and source_ref")

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
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="LiteLLM branch, tag or commit SHA (default: main)")
    parser.add_argument("--url", help="read this URL instead of LiteLLM on GitHub (for testing)")
    parser.add_argument("--summary", type=Path, help="write a Markdown summary of the changes here (for the release PR description)")
    parser.add_argument("--drop-removed", action="store_true",
                        help="remove models LiteLLM no longer lists instead of keeping their last price")
    args = parser.parse_args()
    build(ref=args.ref, url=args.url, summary=args.summary, drop_removed=args.drop_removed)
