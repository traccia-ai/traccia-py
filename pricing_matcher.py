"""Match the model name a provider SDK reports to a key in the pricing table.

Names are compared by their base name: lower-cased, without the provider path
(groq/openai/gpt-oss-120b -> gpt-oss-120b), a Bedrock region (us.), or a date or
-latest suffix. Among keys with the same base name, the span's vendor's own key wins
(its key for the exact dated version first), then the key for the exact name, then the
shortest key. A name with no matching base name gets no price, and neither does a call from
a strict vendor (Groq) whose own keys do not have it.

Kept identical in three repos, which all run tests/fixtures/pricing_match_cases.json:

    traccia-py                  pricing_matcher.py
    traccia-dashboard-service   app/services/pricing/matcher.py
    traccia-node                src/processor/pricing-matcher.ts (port)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

_REGION = re.compile(r"^(?:us|eu|apac|au|ca|jp|global|us-gov)\.")
_SUFFIX = re.compile(r"(?:-\d{4}-\d{2}-\d{2}|-\d{8}|-latest)$")

# llm.vendor values whose pricing providers have another name. Any other vendor is
# its own provider family. A family is the provider of that name plus its variants
# after "_" or "-": bedrock -> bedrock_converse, vertex_ai -> vertex_ai-language-models.
_VENDOR_PROVIDERS = {
    "openai": ("openai", "text-completion-openai"),
    "azure_openai": ("azure",),
    "google_gemini": ("gemini",),
    "google": ("vertex_ai",),  # LangChain Vertex AI models
    "aws": ("bedrock",),  # LangChain Bedrock models
}

# llm.vendor values set only by a provider's own SDK, which calls no other provider (the openai
# SDK also calls OpenAI-compatible providers, so "openai" is not one). Their calls are priced
# only from that provider's keys, never from another provider's key for the same model name.
# The SDKs' cost engines apply the same rule (STRICT_VENDORS).
_STRICT_VENDORS = frozenset({"groq"})

# First-party names the pricing table lists only under a dated key (from traccia-node's
# MODEL_ALIASES). Looked up by _name; a target missing from the table changes nothing.
_MODEL_ALIASES = {
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-opus-latest": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest": "claude-3-7-sonnet-20250219",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-sonnet-4-0": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-opus-4-0": "claude-opus-4-20250514",
}


class PriceMatch(NamedTuple):
    key: str
    kind: str  # exact | base
    provider: str  # pricing provider of the key ("" when unknown)
    # A base match the vendor did not settle, among keys of several providers.
    ambiguous: bool = False


def _name(name: str) -> str:
    """Lower-cased, without the provider path or region; the date is kept. An OpenAI
    fine-tune (ft:<base>:<org>:<suffix>:<id>) is named ft:<base>."""
    name = _REGION.sub("", name.strip().lower().rsplit("/", 1)[-1])
    if name.startswith("ft:"):
        name = ":".join(name.split(":", 2)[:2])
    return name


def base_name(name: str) -> str:
    return _SUFFIX.sub("", _name(name))


def _provider(key: str, entry: Any) -> str:
    provider = entry.get("_provider") if isinstance(entry, Mapping) else None
    if not provider and "/" in key:
        provider = key.split("/", 1)[0]
    return (provider or "").lower()


def _vendor_providers(vendor: Optional[str]) -> tuple:
    v = str(vendor or "").strip().lower()
    return _VENDOR_PROVIDERS.get(v, (v,)) if v else ()


def _in_family(provider: str, families: tuple) -> bool:
    return any(
        provider == f or provider.startswith(f + "_") or provider.startswith(f + "-")
        for f in families
    )


# The last table indexed, its size, and base name -> keys.
_last: tuple = (None, 0, {})


def _index(table: Mapping[str, Any]) -> Dict[str, List[str]]:
    global _last
    cached, size, index = _last
    if cached is table and size == len(table):
        return index
    index = {}
    for key in table:
        index.setdefault(base_name(key), []).append(key)
    _last = (table, len(table), index)
    return index


def candidate_keys(model: Optional[str], table: Mapping[str, Any]) -> List[str]:
    """Every key *model* could be priced from (all keys with its base name)."""
    if not model or not table or not str(model).strip():
        return []
    base = base_name(str(model))
    # The index can be stale for a table edited in place without changing size.
    return [k for k in _index(table).get(base, []) if k in table] if base else []


def match_model(
    model: Optional[str],
    table: Mapping[str, Any],
    vendor: Optional[str] = None,
) -> Optional[PriceMatch]:
    """Return the pricing key for *model*, or None. *vendor* is the span's llm.vendor."""
    keys = candidate_keys(model, table)
    alias = _MODEL_ALIASES.get(_name(str(model))) if model and str(model).strip() else None
    if alias in table and alias not in keys:
        keys.append(alias)
    if not keys:
        return None
    model = str(model)
    wanted = _name(model)
    providers = _vendor_providers(vendor)

    def own(key: str) -> bool:
        return _in_family(_provider(key, table[key]), providers)

    # The vendor's own keys first; then the exact key, the alias target, a key for the exact
    # name (with its date or -latest), and the shortest key, also when the vendor has no key
    # of its own.
    key = min(
        keys,
        key=lambda k: (not own(k), k != model, k != alias, _name(k) != wanted, len(k), k),
    )
    if not own(key) and str(vendor or "").strip().lower() in _STRICT_VENDORS:
        return None
    if key == model:
        return PriceMatch(key, "exact", _provider(key, table[key]))
    ambiguous = not own(key) and len({_provider(k, table[k]) for k in keys}) > 1
    return PriceMatch(key, "base", _provider(key, table[key]), ambiguous)
