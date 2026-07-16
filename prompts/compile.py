"""{{var}} compile contract (parity with dashboard + Node SDK)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class CompileError(ValueError):
    """Missing required template variable."""


def extract_variable_names(text: str) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for match in _VAR_RE.finditer(text or ""):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def extract_from_body(prompt_type: str, body: Mapping[str, Any]) -> List[str]:
    if prompt_type == "text":
        return extract_variable_names(str(body.get("text") or ""))
    names: List[str] = []
    seen: Set[str] = set()
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else ""
        for name in extract_variable_names(str(content or "")):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def compile_string(
    template: str,
    variables: Mapping[str, Any],
    *,
    required: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    """Substitute {{vars}}. Missing required → CompileError. Unknown extras → warn names."""
    req = list(required) if required is not None else extract_variable_names(template)
    missing = [n for n in req if n not in variables or variables[n] is None]
    if missing:
        raise CompileError(f"Missing required variables: {', '.join(missing)}")

    used = set(extract_variable_names(template))
    extras = [k for k in variables.keys() if k not in used]

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        return str(variables[name])

    return _VAR_RE.sub(repl, template), extras


def compile_body(
    prompt_type: str,
    body: Mapping[str, Any],
    variables: Mapping[str, Any],
    *,
    declared: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    if declared:
        required = [d["name"] for d in declared if isinstance(d, dict) and d.get("name")]
    else:
        required = extract_from_body(prompt_type, body)

    used = set(required or [])

    if prompt_type == "text":
        text, _ = compile_string(str(body.get("text") or ""), variables, required=required)
        extras = [k for k in variables.keys() if k not in used]
        return {"text": text}, extras

    messages_out: List[Dict[str, Any]] = []
    req_once: Optional[Sequence[str]] = required
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content, _ = compile_string(str(msg.get("content") or ""), variables, required=req_once)
        req_once = None
        messages_out.append({**msg, "content": content})
    extras = [k for k in variables.keys() if k not in used]
    return {"messages": messages_out}, extras
