"""Local ports of platform deterministic scorers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, Optional

BUILTIN_SCORERS = frozenset({"exact_match", "contains", "json_valid"})


def builtin_definition_hash(name: str) -> str:
    payload = '{"implementation_version":"1","type":"' + name + '"}'
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip()


def score_exact_match(*, output: Any, expected: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if expected is None:
        return {"name": "exact_match", "passed": False, "reason": "missing_expected_output", "score": 0.0}
    a = _normalize_text(output)
    e = _normalize_text(expected)
    case_sensitive = bool((config or {}).get("case_sensitive", False))
    if not case_sensitive:
        a, e = a.lower(), e.lower()
    passed = a == e
    return {
        "name": "exact_match",
        "passed": passed,
        "reason": None if passed else "mismatch",
        "score": 1.0 if passed else 0.0,
    }


def score_contains(*, output: Any, expected: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if expected is None:
        return {"name": "contains", "passed": False, "reason": "missing_expected_output", "score": 0.0}
    a = _normalize_text(output)
    e = _normalize_text(expected)
    case_sensitive = bool((config or {}).get("case_sensitive", False))
    if not case_sensitive:
        a, e = a.lower(), e.lower()
    if not e:
        return {"name": "contains", "passed": False, "reason": "empty_expected", "score": 0.0}
    passed = e in a
    return {
        "name": "contains",
        "passed": passed,
        "reason": None if passed else "not_found",
        "score": 1.0 if passed else 0.0,
    }


def score_json_valid(*, output: Any, expected: Any = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _ = expected, config
    if isinstance(output, (dict, list)):
        return {"name": "json_valid", "passed": True, "reason": None, "score": 1.0}
    if not isinstance(output, str):
        return {"name": "json_valid", "passed": False, "reason": "not_json", "score": 0.0}
    text = output.strip()
    if not text:
        return {"name": "json_valid", "passed": False, "reason": "empty", "score": 0.0}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"name": "json_valid", "passed": False, "reason": "parse_error", "score": 0.0}
    if not isinstance(parsed, (dict, list)):
        return {"name": "json_valid", "passed": False, "reason": "not_object_or_array", "score": 0.0}
    return {"name": "json_valid", "passed": True, "reason": None, "score": 1.0}


_BUILTIN_IMPL: Dict[str, Callable[..., Dict[str, Any]]] = {
    "exact_match": score_exact_match,
    "contains": score_contains,
    "json_valid": score_json_valid,
}


def run_builtin_scorer(
    name: str,
    *,
    output: Any,
    expected: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    impl = _BUILTIN_IMPL.get(name)
    if not impl:
        raise ValueError(f"Unknown builtin scorer: {name}")
    out = impl(output=output, expected=expected, config=config)
    out.setdefault("name", name)
    out.setdefault("status", "judged")
    out["type"] = name
    out["scorer_name"] = name
    out.setdefault("definition_hash", builtin_definition_hash(name))
    out.setdefault("definition_identity", "builtin")
    return out
