"""SDK policy enforcement point: check() on LLM and tool calls under @govern."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from traccia.config import load_config
from traccia.governance import evidence
from traccia.governance.policy import AgentBlockedError, ApprovalPending, _derive_base_url, _http_session
from traccia import runtime_config

logger = logging.getLogger("traccia.governance")

CHECK_PATH = "/api/v1/policy/check"
SETTLE_PATH = "/api/v1/policy/settle"
_BUDGET_CACHE_TTL_S = 5.0
_budget_lock = threading.Lock()
_remaining_budget: Dict[str, tuple[float, float]] = {}
_last_reads = evidence.last_reads
_retrieval_by_trace = evidence.retrieval_by_trace
_prompt_by_trace = evidence.prompt_by_trace
_RESOURCE_ID_KEYS = ("resource_id", "customer_id", "id")
_PROMPT_KEYS = ("traccia.prompt.name", "traccia.prompt.label", "traccia.prompt.version_id")


def _enrich_blocked(exc: AgentBlockedError, decision: Dict[str, Any]) -> AgentBlockedError:
    exc.decision_id = decision.get("id")
    exc.remaining_budget_usd = decision.get("remaining_budget_usd")
    exc.reasons = decision.get("reasons") or []
    return exc


def _credentials() -> tuple[Optional[str], Optional[str]]:
    config = load_config()
    api_key = config.tracing.api_key or os.environ.get("TRACCIA_API_KEY")
    endpoint = config.tracing.endpoint or os.environ.get("TRACCIA_ENDPOINT")
    return api_key, endpoint


def _agent_id() -> Optional[str]:
    return runtime_config.get_agent_id() or os.environ.get("TRACCIA_AGENT_ID")


def _as_otel_hex(value: Any, width: int) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return format(value, f"0{width}x")
    text = str(value).strip()
    return text or None


def _trace_ids() -> tuple[Optional[str], Optional[str]]:
    try:
        from traccia.context import get_current_span

        span = get_current_span()
        if span and getattr(span, "context", None):
            return (
                _as_otel_hex(getattr(span.context, "trace_id", None), 32),
                _as_otel_hex(getattr(span.context, "span_id", None), 16),
            )
    except Exception:
        pass
    return None, None


def _stamp_span(decision: Dict[str, Any]) -> None:
    try:
        from traccia.context import get_current_span

        span = get_current_span()
        if not span:
            return
        if decision.get("id"):
            span.set_attribute("traccia.policy.decision_id", str(decision["id"]))
        if decision.get("effect"):
            span.set_attribute("traccia.policy.effect", str(decision["effect"]))
        span.set_attribute("traccia.policy.would_have", bool(decision.get("would_have")))
        activation = decision.get("activation")
        if activation in ("observe", "warn", "block"):
            span.set_attribute("traccia.policy.activation", str(activation))
        ids = decision.get("policy_ids") or []
        if ids:
            span.set_attribute("traccia.policy.ids", ",".join(str(i) for i in ids))
        reasons = decision.get("reasons") or []
        if reasons:
            span.set_attribute("traccia.policy.reason", str(reasons[0])[:500])
    except Exception:
        logger.debug("Could not stamp policy attributes on span", exc_info=True)


def _cache_budget(agent_id: str, remaining: Optional[float]) -> None:
    if remaining is None:
        return
    with _budget_lock:
        _remaining_budget[agent_id] = (time.time(), float(remaining))


def cached_remaining_budget(agent_id: str) -> Optional[float]:
    with _budget_lock:
        entry = _remaining_budget.get(agent_id)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > _BUDGET_CACHE_TTL_S:
            del _remaining_budget[agent_id]
            return None
        return value


def _blocked_message(decision: Dict[str, Any]) -> str:
    reasons = decision.get("reasons") or []
    reason = reasons[0] if reasons else "policy denied this call"
    remaining = decision.get("remaining_budget_usd")
    decision_id = decision.get("id") or ""
    parts = [reason]
    if remaining is not None:
        parts.append(f"remaining budget ${remaining:.2f}")
    if decision_id:
        parts.append(f"decision {decision_id}")
    return ". ".join(parts)


def _resource_id(row: Any) -> Optional[str]:
    """The entity a tool call reads or writes, when its arguments or result name one."""
    if not isinstance(row, dict):
        return None
    customer = _customer_id(row)
    if customer:
        return customer
    for key in _RESOURCE_ID_KEYS:
        value = row.get(key)
        if value not in (None, "") and not isinstance(value, (dict, list)):
            return str(value)
    return None


def _read_scope() -> Optional[tuple]:
    """Freshness evidence belongs to one tenant, agent and run."""
    run = evidence.run_key(_trace_ids()[0])
    if not run:
        return None
    return (runtime_config.get_tenant_id() or "", _agent_id() or "", run)


def remember_tool_result(name: str, result: Any, arguments: Optional[Dict[str, Any]] = None) -> None:
    """Keep the last as_of or updated_at from a tool result for the current run."""
    if not name or not isinstance(result, dict):
        return
    scope = _read_scope()
    if scope is None:
        return
    for field in ("as_of", "updated_at"):
        value = result.get(field)
        if value not in (None, ""):
            entry = {"tool": str(name), "read_at": str(value), "field": field}
            resource_id = _resource_id(arguments) or _resource_id(result)
            if resource_id:
                entry["resource_id"] = resource_id

            def _put(reads: Optional[Dict[tuple, Dict[str, str]]]) -> Dict[tuple, Dict[str, str]]:
                reads = dict(reads or {})
                reads.pop((entry["tool"], resource_id), None)
                reads[(entry["tool"], resource_id)] = entry
                while len(reads) > evidence.MAX_READS_PER_SCOPE:
                    reads.pop(next(iter(reads)))
                return reads

            _last_reads.update(scope, scope[2], _put)
            return


def note_retrieval_attributes(trace_id: Optional[str], attributes: Optional[Dict[str, Any]]) -> None:
    """Record retrieval evidence only when a span already set it."""
    key = evidence.run_key(trace_id)
    if not key or not attributes:
        return
    chunk = attributes.get("traccia.retrieval.chunk_count")
    present_flag = attributes.get("traccia.retrieval.present")
    if chunk is None and present_flag is None:
        return

    def _merge(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        entry = dict(entry or {})
        if present_flag is True or chunk is not None:
            entry["present"] = True
        if chunk is not None:
            entry["chunk_count"] = chunk
        return entry

    _retrieval_by_trace.update(key, key, _merge)


def _freshness_context(resource_id: Optional[str] = None) -> list:
    """Reads from this run only, newest first; skip reads of a different entity than the one being written."""
    scope = _read_scope()
    if scope is None:
        return []
    reads = _last_reads.get(scope) or {}
    return [
        dict(item)
        for item in reversed(list(reads.values()))
        if not (resource_id and item.get("resource_id") and item["resource_id"] != resource_id)
    ]


def _retrieval_context(trace_id: Optional[str]) -> Dict[str, Any]:
    key = evidence.run_key(trace_id)
    if not key:
        return {"present": False}
    found = _retrieval_by_trace.get(key)
    if not found:
        return {"present": False}
    out: Dict[str, Any] = {"present": bool(found.get("present"))}
    if "chunk_count" in found:
        out["chunk_count"] = found["chunk_count"]
    return out


def note_prompt_attributes(trace_id: Optional[str], attributes: Optional[Dict[str, Any]]) -> None:
    """Remember a compiled Traccia prompt for later LLM checks in this trace.

    compile() stamps the parent span. The provider wrapper then starts a child
    span, and that child is current when the check runs.
    """
    if not attributes:
        return
    name = attributes.get("traccia.prompt.name")
    if not name:
        return
    key = evidence.run_key(trace_id)
    if not key:
        return
    entry = {"name": str(name)}
    label = attributes.get("traccia.prompt.label")
    version_id = attributes.get("traccia.prompt.version_id")
    if label:
        entry["label"] = str(label)
    if version_id:
        entry["version_id"] = str(version_id)
    _prompt_by_trace.update(key, key, lambda _old: entry)


def _prompt_from_attributes(attributes: Any) -> Dict[str, Any]:
    if not isinstance(attributes, dict):
        attributes = getattr(attributes, "_dict", None) if attributes is not None else None
    if not isinstance(attributes, dict):
        try:
            attributes = dict(attributes or {})
        except Exception:
            return {}
    prompt: Dict[str, Any] = {}
    name = attributes.get("traccia.prompt.name")
    label = attributes.get("traccia.prompt.label")
    version_id = attributes.get("traccia.prompt.version_id")
    if name:
        prompt["name"] = name
    if label:
        prompt["label"] = label
    if version_id:
        prompt["version_id"] = version_id
    return prompt


def _prompt_context() -> Dict[str, Any]:
    try:
        from traccia.context import get_current_span

        span = get_current_span()
    except Exception:
        span = None
    from_span = _prompt_from_attributes(getattr(span, "attributes", None) if span else None)
    if from_span.get("name"):
        return from_span
    trace_id = evidence.run_key(_trace_ids()[0])
    if not trace_id:
        return from_span
    remembered = dict(_prompt_by_trace.get(trace_id) or {})
    return remembered or from_span


def _customer_id(arguments: Dict[str, Any]) -> Optional[str]:
    customer = arguments.get("customer")
    if isinstance(customer, dict) and customer.get("id") not in (None, ""):
        return str(customer.get("id"))
    return None


def check_policy(
    *,
    action: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    resource: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST /api/v1/policy/check. Safe to call from custom tools; @govern uses this automatically."""
    api_key, endpoint = _credentials()
    if not api_key or not endpoint:
        return {"effect": "allow", "would_have": False, "reasons": ["missing_credentials"]}
    agent_id = _agent_id()
    if not agent_id:
        return {"effect": "allow", "would_have": False, "reasons": ["missing_agent_id"]}
    trace_id, span_id = _trace_ids()
    payload = {
        "principal": {"agent_id": agent_id},
        "action": action,
        "resource": resource or {},
        "context": dict(context or {}),
        "agent_id": agent_id,
        "trace_id": trace_id,
        "span_id": str(span_id) if span_id is not None else None,
    }
    payload["context"].setdefault("agent_id", agent_id)
    remaining = cached_remaining_budget(agent_id)
    if remaining is not None:
        payload["context"].setdefault("remaining_budget_usd", remaining)
    try:
        base = _derive_base_url(endpoint)
        response = _http_session.post(
            f"{base}{CHECK_PATH}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=2,
        )
        if response.status_code != 200:
            logger.warning(
                "policy check HTTP %s; allowing call (%s)",
                response.status_code,
                (response.text or "")[:180],
            )
            return {"effect": "allow", "would_have": False, "reasons": ["check_http_error"]}
        decision = response.json()
    except Exception as exc:
        logger.warning("policy check failed: %s; allowing call", exc)
        return {"effect": "allow", "would_have": False, "reasons": ["check_error"]}
    _cache_budget(agent_id, decision.get("remaining_budget_usd"))
    _stamp_span(decision)
    if decision.get("would_have"):
        return decision
    effect = decision.get("effect") or "allow"
    if effect == "queue":
        raise ApprovalPending(
            "This action is waiting for approval. Do not run the tool or retry the check.",
            approval_id=decision.get("approval_id"),
            expires_at=decision.get("approval_expires_at"),
            decision_id=decision.get("id"),
        )
    if effect == "deny":
        raise _enrich_blocked(AgentBlockedError(_blocked_message(decision)), decision)
    return decision


def settle_policy(decision: Optional[Dict[str, Any]], *, release: bool = False, actual_usd: Optional[float] = None) -> None:
    if not decision:
        return
    reserved = (decision.get("obligations") or {}).get("reserved_usd")
    if reserved is None:
        return
    api_key, endpoint = _credentials()
    agent_id = _agent_id()
    if not api_key or not endpoint or not agent_id:
        return
    trace_id, _span_id = _trace_ids()
    try:
        base = _derive_base_url(endpoint)
        _http_session.post(
            f"{base}{SETTLE_PATH}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "agent_id": agent_id,
                "trace_id": trace_id,
                "reserved_usd": reserved,
                "actual_usd": actual_usd,
                "release": release,
            },
            timeout=2,
        )
    except Exception as exc:
        logger.debug("policy settle failed: %s", exc)


def _estimate_tokens(kwargs: Dict[str, Any]) -> Optional[int]:
    try:
        from traccia.processors.token_counter import estimate_tokens_from_text
    except Exception:
        estimate_tokens_from_text = None
    messages = kwargs.get("messages")
    text = kwargs.get("prompt") or kwargs.get("input")
    blob = ""
    if isinstance(messages, list):
        parts = []
        for item in messages:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        blob = "\n".join(parts)
    elif isinstance(text, str):
        blob = text
    if not blob:
        return None
    if estimate_tokens_from_text:
        try:
            return int(estimate_tokens_from_text(blob))
        except Exception:
            pass
    return max(len(blob) // 4, 1)


def enforce_llm_call(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not runtime_config.pep_enabled():
        return None
    model = kwargs.get("model")
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    trace_id, _span_id = _trace_ids()
    context: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "input_tokens": _estimate_tokens(kwargs),
        "retrieval": _retrieval_context(trace_id),
    }
    prompt = _prompt_context()
    if prompt:
        context["prompt"] = prompt
    decision = check_policy(
        action={"type": "llm_call", "name": "llm_call", "model": model},
        context=context,
    )
    if (decision.get("effect") == "reshape") and not decision.get("would_have"):
        obligations = decision.get("obligations") or {}
        cheaper = obligations.get("cheaper_model") or obligations.get("fallback_model")
        if cheaper:
            kwargs["model"] = cheaper
        if obligations.get("clamp_max_tokens") is not None:
            kwargs["max_tokens"] = obligations["clamp_max_tokens"]
    return decision


def finish_llm_call(decision: Optional[Dict[str, Any]], *, release: bool = False, actual_usd: Optional[float] = None) -> None:
    settle_policy(decision, release=release, actual_usd=actual_usd)


def enforce_tool_call(name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if not runtime_config.pep_enabled():
        return None
    payload = arguments or {}
    context: Dict[str, Any] = {"input": payload, "tool_name": name}
    freshness = _freshness_context(_resource_id(payload))
    if freshness:
        context["freshness"] = freshness
    customer_id = _customer_id(payload)
    if customer_id:
        context["customer_id"] = customer_id
    return check_policy(
        action={"type": "tool_call", "name": name},
        context=context,
    )
