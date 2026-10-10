"""Match the model name a provider SDK reports to a key in the pricing table.

Names are compared by their base name: lower-cased, without the provider path
(groq/openai/gpt-oss-120b -> gpt-oss-120b), a Bedrock region (us.), or a date or
-latest suffix. Among keys with the same base name, the span's vendor's own key wins,
then the shortest key. A name with no matching base name gets no price.

Kept byte-identical in two repos, which both run tests/fixtures/pricing_match_cases.json:

    traccia-py                  pricing_matcher.py
    traccia-dashboard-service   app/services/pricing/matcher.py
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

_REGION = re.compile(r"^(?:us|eu|apac|au|ca|jp|global|us-gov)\.")
_SUFFIX = re.compile(r"(?:-\d{4}-\d{2}-\d{2}|-\d{8}|-latest)$")


class PriceMatch(NamedTuple):
    key: str
    kind: str  # exact | base
    provider: str  # pricing provider of the key ("" when unknown)


def base_name(name: str) -> str:
    name = name.strip().lower().rsplit("/", 1)[-1]
    return _SUFFIX.sub("", _REGION.sub("", name))


def _provider(key: str, entry: Any) -> str:
    provider = entry.get("_provider") if isinstance(entry, Mapping) else None
    if not provider and "/" in key:
        provider = key.split("/", 1)[0]
    return (provider or "").lower()


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


def match_model(
    model: Optional[str],
    table: Mapping[str, Any],
    vendor: Optional[str] = None,
) -> Optional[PriceMatch]:
    """Return the pricing key for *model*, or None. *vendor* is the span's llm.vendor."""
    if not model or not table or not str(model).strip():
        return None
    model = str(model)
    if model in table:
        return PriceMatch(model, "exact", _provider(model, table[model]))
    keys = _index(table).get(base_name(model))
    if not keys:
        return None
    v = re.sub(r"[^a-z0-9]", "", (vendor or "").lower())

    def own(key: str) -> bool:
        p = re.sub(r"[^a-z0-9]", "", _provider(key, table[key]))
        return bool(v and p) and (p.startswith(v) or v.startswith(p))

    key = min(keys, key=lambda k: (not own(k), len(k), k))
    return PriceMatch(key, "base", _provider(key, table[key]))
