"""Match the model name a provider SDK reports to a key in the pricing table.

Provider SDKs and the LiteLLM-derived pricing table name the same model differently:

    SDK reports                               pricing key
    openai/gpt-oss-120b            (Groq)     groq/openai/gpt-oss-120b
    models/gemini-2.5-flash        (Gemini)   gemini-2.5-flash
    moonshotai/kimi-k2-instruct    (Groq)     groq/moonshotai/kimi-k2-instruct-0905
    us.anthropic.claude-…-v2:0     (Bedrock)  anthropic.claude-…-v2:0

Every key and every incoming name is reduced to progressively looser forms:

    EXACT    lower-cased as given
    ALIAS    provider prefix and SDK noise removed  (groq/openai/x -> openai/x),
             or just the last path segment           (openai/x -> x)
    UNDATED  date / version suffix removed           (x-0905 -> x)

A candidate's score is the looser of the two forms that met. Candidates priced by
the calling vendor (``llm.vendor``) win outright, since that is who bills the call;
then the lowest score, then a first-party entry, then the shortest key. If no form
meets, the longest key that the name starts with at a ``-``/``@``/``:`` boundary is
used (PREFIX).

This module uses only the standard library and is kept byte-identical in two repos,
so the SDK and the platform price a span the same way:

    traccia-py                  pricing_matcher.py
    traccia-dashboard-service   app/services/pricing/matcher.py

Both repos run the cases in tests/fixtures/pricing_match_cases.json. Change both
copies together.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

EXACT, ALIAS, UNDATED, PREFIX = 0, 1, 2, 3
_KIND = {EXACT: "exact", ALIAS: "alias", UNDATED: "undated", PREFIX: "prefix"}

# Prefixes SDKs or hosting platforms put in front of the model name.
_NOISE_PREFIXES = (
    re.compile(r"^models/"),  # Gemini SDK
    re.compile(r"^accounts/[^/]+/models/"),  # Fireworks
    re.compile(
        r"^(?:us|eu|apac|au|ca|jp|global)\.(?=[a-z])"
    ),  # Bedrock cross-region profile
)
_BEDROCK_VERSION = re.compile(r"(?:-v\d+)?:\d+$")  # …-v2:0
_VERSION_SUFFIX = re.compile(
    r"(?:[-@](?:\d{4}-\d{2}-\d{2}|\d{8})"  # -2024-08-06, -20241022, @20250929
    r"|-\d{4}|-0\d{2}|@\d{3}"  # -0905, -002, @001
    r"|-latest)$"
)
_PREFIX_BOUNDARY = "-@:"

# Pricing providers (LiteLLM's ``litellm_provider``) that are the model maker's own API.
_FIRST_PARTY = frozenset(
    {
        "openai",
        "text-completion-openai",
        "anthropic",
        "gemini",
        "vertex_ai-language-models",
        "mistral",
        "cohere",
        "cohere_chat",
        "deepseek",
        "xai",
    }
)

# ``llm.vendor`` values whose pricing provider has a different name.
_VENDOR_PROVIDERS = {
    "googlegemini": ("gemini", "vertexai"),
    "google": ("gemini", "vertexai"),
    "googlegenai": ("gemini", "vertexai"),
    "vertexai": ("vertexai",),
    "azureopenai": ("azure",),
    "awsbedrock": ("bedrock",),
    "aws": ("bedrock",),
    "mistralai": ("mistral",),
}


@dataclass(frozen=True)
class PriceMatch:
    """A pricing key chosen for a model name."""

    key: str
    kind: str  # exact | alias | undated | prefix
    provider: str  # pricing provider of the matched key ("" when unknown)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _strip_vendor_dot(segment: str) -> str:
    # Bedrock: anthropic.claude-3-5-sonnet -> claude-3-5-sonnet, but keep gpt-3.5 / grok-4.7.
    dot = segment.find(".")
    if dot > 0:
        head, rest = segment[:dot], segment[dot + 1 :]
        if head.replace("-", "").isalpha() and "-" in rest:
            return rest
    return segment


def _clean(name: str) -> str:
    for pattern in _NOISE_PREFIXES:
        name = pattern.sub("", name)
    head, sep, last = name.rpartition("/")
    last = _strip_vendor_dot(last)
    last = _BEDROCK_VERSION.sub("", last)
    return f"{head}{sep}{last}"


def _forms(name: str, strip_first_segment: bool) -> List[Tuple[int, str]]:
    exact = name.strip().replace("\\", "/").lower()
    alias = exact
    if strip_first_segment and "/" in alias:
        alias = alias.split("/", 1)[1]
    alias = _clean(alias)
    bare = alias.rsplit("/", 1)[-1]
    undated = _VERSION_SUFFIX.sub("", bare)
    # alias and bare share a level so an org prefix never outranks the vendor hint.
    return [(EXACT, exact), (ALIAS, alias), (ALIAS, bare), (UNDATED, undated)]


def _vendor_matches(vendor: str, provider: str) -> bool:
    if not vendor or not provider:
        return False
    p = _norm(provider)
    return any(
        p == v or p.startswith(v) for v in _VENDOR_PROVIDERS.get(vendor, (vendor,))
    )


class _Index:
    """Every form of every pricing key, so a lookup is a few dict hits."""

    def __init__(self, table: Mapping[str, Any]) -> None:
        self.forms: Dict[str, Dict[str, int]] = {}
        self.provider: Dict[str, str] = {}
        for key, entry in table.items():
            provider = entry.get("_provider") if isinstance(entry, Mapping) else None
            if not provider and "/" in key:
                provider = key.split("/", 1)[0]
            self.provider[key] = (provider or "").lower()
            # Keys are "<provider>/<model>" whenever they contain a slash; models without
            # a provider segment ("gpt-4o", "anthropic.claude-…") keep their full name.
            for level, form in _forms(key, strip_first_segment=True):
                if not form:
                    continue
                hits = self.forms.setdefault(form, {})
                if level < hits.get(key, PREFIX):
                    hits[key] = level

    def rank(self, key: str, score: int, vendor: str) -> tuple:
        provider = self.provider[key]
        first_party = provider in _FIRST_PARTY if provider else "/" not in key
        return (
            not _vendor_matches(vendor, provider),
            score,
            not first_party,
            len(key),
            key,
        )

    def match(self, key: str, score: int) -> PriceMatch:
        return PriceMatch(key, _KIND[score], self.provider[key])


_cache: Dict[int, Tuple[Mapping[str, Any], int, _Index]] = {}
_cache_lock = threading.Lock()
_CACHE_SIZE = 8


def _index_for(table: Mapping[str, Any]) -> _Index:
    # Keyed by identity; holding the table keeps its id from being reused. A table whose
    # size changes is re-indexed (pricing tables are replaced, not edited, in practice).
    entry = _cache.get(id(table))
    if entry is not None and entry[0] is table and entry[1] == len(table):
        return entry[2]
    index = _Index(table)
    with _cache_lock:
        if len(_cache) >= _CACHE_SIZE:
            _cache.pop(next(iter(_cache)))
        _cache[id(table)] = (table, len(table), index)
    return index


def match_model(
    model: Optional[str],
    table: Mapping[str, Any],
    vendor: Optional[str] = None,
) -> Optional[PriceMatch]:
    """
    Return the pricing key for *model*, or None when nothing plausible matches.

    Args:
        model: Model name as the provider SDK reported it.
        table: Pricing table mapping key -> entry. An entry's ``_provider`` (from
            LiteLLM) splits off the provider prefix and identifies the vendor's own price.
        vendor: The span's ``llm.vendor``; that provider's entry wins when several fit.

    Returns:
        The matched key, how it matched, and its provider; or None.
    """
    if not model or not table or not str(model).strip():
        return None
    model = str(model)
    if model in table and not vendor:
        return _index_for(table).match(model, EXACT)

    index = _index_for(table)
    v = _norm(vendor or "")

    scores: Dict[str, int] = {}
    for model_level, form in _forms(model, strip_first_segment=False):
        for key, key_level in index.forms.get(form, {}).items():
            score = max(model_level, key_level)
            if score < scores.get(key, PREFIX):
                scores[key] = score
    if scores:
        best = min(scores, key=lambda k: index.rank(k, scores[k], v))
        return index.match(best, scores[best])

    # Longest key the name starts with, cut at a boundary so gpt-4 never claims gpt-4.1.
    for form in dict.fromkeys(f for level, f in _forms(model, False) if level == ALIAS):
        for cut in range(len(form) - 1, 0, -1):
            if form[cut] not in _PREFIX_BOUNDARY:
                continue
            hits = index.forms.get(form[:cut])
            if hits:
                return index.match(
                    min(hits, key=lambda k: index.rank(k, hits[k], v)), PREFIX
                )
    return None
