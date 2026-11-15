"""SDK policy enforcement point: check() on LLM and tool calls under @govern."""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import logging
import os
import queue
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, Optional, Set, Tuple

from traccia import client_config
from traccia.governance import evidence, fail_mode
from traccia.governance.policy import (
    AgentBlockedError,
    ApprovalPending,
    GovernanceConfigError,
    _derive_base_url,
    _http_session,
    rejected_error,
)
from traccia import runtime_config

logger = logging.getLogger("traccia.governance")

CHECK_PATH = "/api/v1/policy/check"
DECISION_EFFECTS = ("allow", "warn", "reshape", "queue", "deny")
SETTLE_PATH = "/api/v1/policy/settle"
_BUDGET_CACHE_TTL_S = 5.0
_budget_lock = threading.Lock()
_remaining_budget: Dict[str, tuple[float, float]] = {}
_last_reads = evidence.last_reads
_retrieval_by_trace = evidence.retrieval_by_trace
_prompt_by_trace = evidence.prompt_by_trace
_PROMPT_KEYS = ("traccia.prompt.name", "traccia.prompt.label", "traccia.prompt.version_id")


def _enrich_blocked(exc: AgentBlockedError, decision: Dict[str, Any]) -> AgentBlockedError:
    exc.decision_id = decision.get("id")
    exc.remaining_budget_usd = decision.get("remaining_budget_usd")
    exc.reasons = decision.get("reasons") or []
    return exc


def _credentials() -> tuple[Optional[str], Optional[str]]:
    return client_config.resolve_credentials()


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


def _span_ended(span: Any) -> bool:
    """Whether the span (a traccia Span, or the OTel span it wraps) has ended. An unsampled
    span never counts as ended: it has no end time, and keeps its attributes on its wrapper."""
    if getattr(span, "_ended", False) is True:
        return True
    end_time = getattr(getattr(span, "_otel_span", None), "end_time", None)
    return isinstance(end_time, int) and not isinstance(end_time, bool)


def _stamp_span(decision: Dict[str, Any]) -> None:
    try:
        from traccia.context import get_current_span

        span = get_current_span()
        if not span or _span_ended(span):
            # Ended: a check that returned after its call was cancelled. OTel drops the
            # attributes and warns, and get_current_span()'s new wrapper never knows it ended.
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
    """The customer a tool call reads or writes, when its arguments or result name one.

    Only customer identity counts. A generic "id" (a policy or document id) does not
    make a read customer-specific, so tool-level reads always reach the guard.
    """
    if not isinstance(row, dict):
        return None
    customer = _customer_id(row)
    if customer:
        return customer
    value = row.get("customer_id")
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


def valid_decision(decision: Any) -> bool:
    """Accept only a decision the SDK can act on; anything else counts as no answer."""
    return (
        isinstance(decision, dict)
        and decision.get("effect") in DECISION_EFFECTS
        and isinstance(decision.get("would_have"), bool)
        and fail_mode.valid_fields(decision)
    )


def _unresolved(ident: fail_mode.Identity, started_at: float, reason: str, detail: str) -> Dict[str, Any]:
    """No decision from Traccia: apply the fail mode (TRC-342).

    A rejection of the key after this check was sent wins over any fail mode.
    """
    rejected = fail_mode.rejected_since(ident, started_at)
    if rejected:
        raise rejected_error(rejected)
    mode = fail_mode.resolve(ident, fail_mode.local_mode())
    if mode == "closed":
        logger.error("policy check could not complete (%s); blocking call (policy fail mode: closed)", detail)
        raise AgentBlockedError(
            f"Policy check could not complete ({detail}). Blocking because the policy fail mode is closed.",
            reasons=[reason],
        )
    logger.warning("policy check could not complete (%s); allowing call (policy fail mode: open)", detail)
    return {"effect": "allow", "would_have": False, "reasons": [reason]}


def check_policy(
    *,
    action: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    resource: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST /api/v1/policy/check. Safe to call from custom tools; @govern uses this automatically.

    Deny raises AgentBlockedError and a queued approval raises ApprovalPending. When
    Traccia gives no answer, the fail mode applies (see traccia.governance.fail_mode).
    A rejected API key, or a missing API key, endpoint or agent ID, always raises.
    """
    api_key, endpoint = _credentials()
    if not api_key or not endpoint:
        raise GovernanceConfigError(
            "Traccia API key or endpoint not set. Policy checks need both "
            "(TRACCIA_API_KEY and TRACCIA_ENDPOINT, or init())."
        )
    agent_id = _agent_id()
    if not agent_id:
        raise GovernanceConfigError(
            "No agent_id on init, @govern, or TRACCIA_AGENT_ID. Policy checks need one."
        )
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
    base = _derive_base_url(endpoint)
    ident = fail_mode.identity(api_key, base)
    started_at = fail_mode._now()
    try:
        response = _http_session.post(
            f"{base}{CHECK_PATH}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=2,
        )
    except Exception as exc:
        return _unresolved(ident, started_at, "check_error", str(exc))
    if response.status_code in fail_mode.REJECT_STATUSES:
        fail_mode.note_rejection(ident, response.status_code)
        raise rejected_error(response.status_code)
    if response.status_code != 200:
        return _unresolved(
            ident, started_at, "check_http_error", f"HTTP {response.status_code}: {(response.text or '')[:180]}"
        )
    try:
        decision = response.json()
    except Exception as exc:
        return _unresolved(ident, started_at, "check_error", f"invalid response: {exc}")
    if not valid_decision(decision):
        return _unresolved(ident, started_at, "check_error", "invalid decision")
    rejected = fail_mode.rejected_since(ident, started_at)
    if rejected:
        # The key was rejected while this check was on its way: its answer, even an
        # allow, no longer holds.
        raise rejected_error(rejected)
    fail_mode.note_answer(ident, decision, started_at)
    _cache_budget(agent_id, decision.get("remaining_budget_usd"))
    _stamp_span(decision)
    if decision.get("would_have"):
        return decision
    effect = decision["effect"]
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


def reserved_usd(decision: Optional[Dict[str, Any]]) -> Optional[float]:
    """The USD amount the policy check reserved for a call, if any."""
    reserved = ((decision or {}).get("obligations") or {}).get("reserved_usd")
    return reserved if isinstance(reserved, (int, float)) and not isinstance(reserved, bool) else None


SettleTarget = Tuple[Optional[str], Optional[str], Optional[str]]


def settle_target() -> Optional[SettleTarget]:
    """The API key, endpoint and agent a policy check run now uses, or None when checks are
    off. Taken with the check and passed to settle_policy(), so the settle reaches the account
    and agent that reserved, even if it runs after stop_tracing() has cleared the credentials
    init() was given, or after a re-init."""
    try:
        if not runtime_config.pep_enabled():
            return None
        api_key, endpoint = _credentials()
        return api_key, endpoint, _agent_id()
    except Exception:
        return None


def settle_policy(
    decision: Optional[Dict[str, Any]],
    *,
    release: bool = False,
    actual_usd: Optional[float] = None,
    model: Optional[str] = None,
    vendor: Optional[str] = None,
    input_tokens: Optional[float] = None,
    output_tokens: Optional[float] = None,
    target: Optional[SettleTarget] = None,
) -> None:
    """Settle the call's reservation.

    Without ``actual_usd`` the server prices the call from ``model``, ``vendor`` and the
    token counts; with none of them it releases the reservation. ``target``: the
    settle_target() of the call's policy check; without it, the current credentials and agent.
    """
    if not decision:
        return
    reserved = (decision.get("obligations") or {}).get("reserved_usd")
    if reserved is None:
        return
    if target is not None:
        api_key, endpoint, agent_id = target
    else:
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
                "model": model,
                "vendor": vendor,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            timeout=2,
        )
    except Exception as exc:
        logger.debug("policy settle failed: %s", exc)


# Tokens one image is counted as in a prompt estimate, in the range vision models charge for a
# typical image. Image parts are counted by this, never by their (base64) URL.
IMAGE_PROMPT_TOKENS = 1000
_IMAGE_PART_TYPES = {"image_url", "image", "input_image"}


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _message_text(message: Any) -> tuple[str, int]:
    """(text, image count) of one message: its content (a string or content parts) plus any
    tool call arguments, which are part of the prompt when the conversation is sent back."""
    if isinstance(message, str):
        return message, 0
    content = _get(message, "content")
    texts = []
    images = 0
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue
            text = _get(part, "text")
            if isinstance(text, str):
                texts.append(text)
            elif _get(part, "type") in _IMAGE_PART_TYPES:
                images += 1
    tool_calls = _get(message, "tool_calls")
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            arguments = _get(_get(call, "function"), "arguments")
            if isinstance(arguments, str):
                texts.append(arguments)
    return "".join(texts), images


def _estimate_tokens(kwargs: Dict[str, Any]) -> Optional[int]:
    try:
        from traccia.processors.token_counter import estimate_tokens_from_text_with_source
    except Exception:
        estimate_tokens_from_text_with_source = None
    messages = kwargs.get("messages")
    text = kwargs.get("prompt") or kwargs.get("input")
    blob = ""
    images = 0
    if isinstance(messages, (list, tuple)):
        parts = []
        for item in messages:
            message_text, message_images = _message_text(item)
            parts.append(message_text)
            images += message_images
        blob = "\n".join(parts)
    elif isinstance(text, str):
        blob = text
    image_tokens = images * IMAGE_PROMPT_TOKENS
    if not blob:
        return image_tokens or None
    rough = max(len(blob) // 4, 1)
    if estimate_tokens_from_text_with_source:
        try:
            count, source = estimate_tokens_from_text_with_source(blob)
            if source == "estimated.tiktoken":
                return int(count) + image_tokens
            # Without tiktoken the counter splits on whitespace, which counts code, JSON or
            # unspaced text as a handful of tokens; never estimate below characters / 4.
            return max(int(count), rough) + image_tokens
        except Exception:
            pass
    return rough + image_tokens


def estimate_prompt_tokens(kwargs: Dict[str, Any]) -> Optional[int]:
    """Rough prompt size in tokens, as sent with the policy check."""
    return _estimate_tokens(kwargs)


# The OpenAI, Anthropic and Groq SDKs pass NOT_GIVEN / omit for a parameter left out
# (chat.completions.stream() forwards every one).
_OMITTED_TYPES = ("NotGiven", "Omit")


def _given(value: Any) -> Any:
    """*value*, or None when it is a provider SDK's omission sentinel."""
    return None if type(value).__name__ in _OMITTED_TYPES else value


def _gemini_body(request: Any) -> Optional[Dict[str, Any]]:
    """The body of a Gemini request= (a dict or a CreateInteractionRequest) as a dict, or None
    when it has another form, which could not be rewritten either."""
    if not isinstance(request, dict) and not hasattr(request, "model_validate"):
        return None
    body = _get(request, "body")
    if hasattr(body, "model_dump"):
        body = body.model_dump(exclude_none=True)
    return body if isinstance(body, dict) else None


def _with_gemini_body(request: Any, body: Dict[str, Any]) -> Any:
    """A copy of a Gemini request= with *body*; the caller's request is left as it is."""
    if isinstance(request, dict):
        return {**request, "body": body}
    return type(request).model_validate({**request.model_dump(exclude_none=True), "body": body})


def _token_cap(
    kwargs: Dict[str, Any], vendor: Optional[str], api: Optional[str]
) -> Tuple[Dict[str, Any], str]:
    """Where this provider API takes the output-token cap: (params, key)."""
    if vendor == "google_gemini":
        config = kwargs.get("generation_config")
        if hasattr(config, "model_dump"):
            config = config.model_dump(exclude_none=True)
        return (config if isinstance(config, dict) else {}), "max_output_tokens"
    if api == "responses":
        return kwargs, "max_output_tokens"
    if vendor == "anthropic":
        return kwargs, "max_tokens"
    for key in ("max_completion_tokens", "max_tokens"):
        if _given(kwargs.get(key)) is not None:
            return kwargs, key
    # OpenAI reasoning models reject max_tokens; every Chat model takes max_completion_tokens.
    return kwargs, "max_completion_tokens" if vendor == "openai" else "max_tokens"


def enforce_llm_call(
    kwargs: Dict[str, Any], vendor: Optional[str] = None, api: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Check policy before an LLM call. *vendor* is the span's llm.vendor; Spend Cap
    uses it to pick that provider's price. *api* is the span's llm.api ("responses"
    for the OpenAI Responses API); with *vendor* it picks the token-cap parameter."""
    if not runtime_config.pep_enabled():
        return None
    params = kwargs
    # Gemini also takes the whole call as request=, which the SDK rejects mixed with body fields.
    request = kwargs.get("request") if vendor == "google_gemini" else None
    if request is not None:
        params = _gemini_body(request) or {}
    model = _given(params.get("model"))
    cap_params, cap_key = _token_cap(params, vendor, api)
    max_tokens = _given(cap_params.get(cap_key))
    trace_id, _span_id = _trace_ids()
    context: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "input_tokens": _estimate_tokens(params),
        "retrieval": _retrieval_context(trace_id),
    }
    if vendor:
        context["vendor"] = vendor
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
        clamp = obligations.get("clamp_max_tokens")
        if request is None:
            if cheaper:
                kwargs["model"] = cheaper
            if clamp is not None:
                if cap_params is kwargs:
                    kwargs[cap_key] = clamp
                else:
                    kwargs["generation_config"] = {**cap_params, cap_key: clamp}
        elif cheaper or clamp is not None:
            reshaped = None
            if "model" in params:  # else unreadable, or an agent interaction (no generation_config)
                body = {**params}
                if cheaper:
                    body["model"] = cheaper
                if clamp is not None:
                    body["generation_config"] = {**cap_params, cap_key: clamp}
                try:
                    reshaped = _with_gemini_body(request, body)
                except Exception:  # the SDK's model rejected the rewritten body
                    reshaped = None
            if reshaped is None:
                settle_policy(decision, release=True)
                raise _enrich_blocked(
                    AgentBlockedError(
                        "Policy reshape (model or max_output_tokens) cannot be applied to this "
                        "Gemini request=; blocking the call"
                    ),
                    decision,
                )
            kwargs["request"] = reshaped
    return decision


def finish_llm_call(
    decision: Optional[Dict[str, Any]],
    *,
    release: bool = False,
    actual_usd: Optional[float] = None,
    model: Optional[str] = None,
    vendor: Optional[str] = None,
    input_tokens: Optional[float] = None,
    output_tokens: Optional[float] = None,
    target: Optional[SettleTarget] = None,
) -> None:
    settle_policy(
        decision,
        release=release,
        actual_usd=actual_usd,
        model=model,
        vendor=vendor,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        target=target,
    )


_settle_executor: Optional[ThreadPoolExecutor] = None
_settle_flush_hook = False
_pending_settles: Set[Future] = set()
# Reentrant, so code holding it can start a settle.
_settle_lock = threading.RLock()


def _settle_pool() -> ThreadPoolExecutor:
    global _settle_executor, _settle_flush_hook
    with _settle_lock:
        if _settle_executor is None:
            _settle_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="traccia-settle")
            if not _settle_flush_hook:
                atexit.register(flush_pending_settles)
                _settle_flush_hook = True
        return _settle_executor


def _settle_done(future: Future) -> None:
    with _settle_lock:
        _pending_settles.discard(future)


def finish_llm_call_in_background(decision: Optional[Dict[str, Any]], **kwargs: Any) -> None:
    """finish_llm_call() on a worker thread, so a slow policy server does not delay the caller.

    Runs in a copy of the current context, so it settles for the agent (and span) that made
    the call. flush_pending_settles() waits for it; it also runs at interpreter exit.
    """
    if reserved_usd(decision) is None:
        return  # nothing was reserved, so there is nothing to settle
    ctx = contextvars.copy_context()
    try:
        future = _settle_pool().submit(ctx.run, finish_llm_call, decision, **kwargs)
    except RuntimeError:
        # The interpreter is shutting down (e.g. a stream finalized at exit): settle here.
        ctx.run(finish_llm_call, decision, **kwargs)
        return
    with _settle_lock:
        _pending_settles.add(future)
    future.add_done_callback(_settle_done)


# Blocking work for async LLM calls (the policy check), run off the event loop.
_offloop_executor: Optional[ThreadPoolExecutor] = None
_offloop_lock = threading.Lock()


def _offloop_pool() -> ThreadPoolExecutor:
    global _offloop_executor
    with _offloop_lock:
        if _offloop_executor is None:
            _offloop_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="traccia-check")
        return _offloop_executor


async def run_off_loop(
    fn: Callable[..., Any],
    *args: Any,
    if_cancelled: Optional[Callable[[Optional[Future]], None]] = None,
) -> Any:
    """Await ``fn(*args)`` run on a worker thread, in a copy of the caller's context (its agent,
    run and span), so blocking I/O such as a slow policy server never stalls the event loop.

    The thread cannot be stopped. If the awaiting task is cancelled (asyncio.CancelledError, a
    wait_for() timeout), ``if_cancelled`` runs in that context copy: with fn's finished Future
    once fn returns, or at once with None if fn never started. It must not raise.
    """
    ctx = contextvars.copy_context()
    future = _offloop_pool().submit(ctx.run, fn, *args)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        if if_cancelled is not None:
            if future.cancel():
                ctx.run(if_cancelled, None)
            else:
                # Runs once fn has returned and left ctx, on the worker (or here, if it has).
                future.add_done_callback(lambda done: ctx.run(if_cancelled, done))
        raise


def _check_with_target(
    kwargs: Dict[str, Any], vendor: Optional[str], api: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[SettleTarget]]:
    target = settle_target()
    return enforce_llm_call(kwargs, vendor=vendor, api=api), target


def _release_late_check(done: Optional[Future]) -> None:
    """The caller was cancelled during its policy check: release what the check reserved, as
    the call will never be made. A background settle, so flush_pending_settles() waits for it."""
    if done is None:
        return  # the check never started: nothing reserved
    try:
        decision, target = done.result()
    except BaseException:
        return  # the check failed: nothing reserved
    try:
        finish_llm_call_in_background(decision, target=target, release=True)
    except Exception:
        pass


async def enforce_llm_call_off_loop(
    kwargs: Dict[str, Any], vendor: Optional[str] = None, api: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[SettleTarget]]:
    """enforce_llm_call() for an async LLM call, on a worker thread (see run_off_loop()). A
    reshape still rewrites ``kwargs``. Returns the decision and the settle_target() the check
    used; settle with finish_llm_call_in_background(decision, target=target, ...).

    A call cancelled during its check releases what the check reserves once it returns.
    """
    if not runtime_config.pep_enabled():
        return None, None  # no check, so no I/O: stay on the loop
    return await run_off_loop(_check_with_target, kwargs, vendor, api, if_cancelled=_release_late_check)


# Work handed off by garbage-collection finalizers (a stream dropped unfinished). The collector
# runs them on whatever thread it interrupts, possibly inside a lock that finalizing the call
# takes (an HTTP connection pool's, ThreadPoolExecutor.submit()'s) or while other code is half
# done, so they only put the work here: SimpleQueue.put() is safe there, as nothing else is. A
# worker thread runs it, and run_handed_off() runs what is left (flush_pending_settles()).
_handed_off: "queue.SimpleQueue[Callable[[], None]]" = queue.SimpleQueue()
_handoff_worker: Optional[threading.Thread] = None
_handoff_pid: Optional[int] = None


def hand_off(work: Callable[[], None]) -> None:
    """Run ``work`` off the caller's thread. Safe in a garbage-collection finalizer."""
    _handed_off.put(work)


def start_handoff_worker() -> None:
    """Start the hand-off worker in this process if it is not running. Call from ordinary
    code, before a finalizer can hand work off: starting a thread takes locks."""
    global _handoff_worker, _handoff_pid
    with _settle_lock:
        worker = _handoff_worker
        if worker is not None and _handoff_pid == os.getpid() and worker.is_alive():
            return
        worker = threading.Thread(target=_handoff_loop, name="traccia-handoff", daemon=True)
        worker.start()
        _handoff_worker, _handoff_pid = worker, os.getpid()


def _run_work(work: Callable[[], None]) -> None:
    try:
        work()
    except Exception as exc:
        logger.debug("handed-off work failed: %s", exc)


def _handoff_loop() -> None:
    while True:
        _run_work(_handed_off.get())


def run_handed_off() -> None:
    """Run the work handed off so far that the worker has not taken, on this thread."""
    while True:
        try:
            work = _handed_off.get_nowait()
        except queue.Empty:
            return
        _run_work(work)


_OFF_THREAD_GRACE_S = 0.25


def _run_off_this_thread(work: Callable[[], None], timeout: Optional[float]) -> None:
    """Run ``work`` on a new thread and wait for it, up to ``timeout`` seconds.

    Shutdown work (stopping a call, closing its response, starting its settle) takes locks
    that the calling thread may hold: stop_tracing() can run in a signal handler that
    interrupted this thread inside httpcore's connection pool or ThreadPoolExecutor.submit().
    On another thread that work waits for the lock instead of deadlocking, and this thread
    returns at the time limit. Runs here only when no thread can be started (interpreter
    exit).
    """
    done = threading.Event()

    def run() -> None:
        try:
            _run_work(work)
        finally:
            done.set()

    try:
        threading.Thread(target=run, name="traccia-shutdown", daemon=True).start()
    except RuntimeError:
        run()
        return
    # A moment past the limit, for what the work started just before it (as stopping a call
    # whose policy check it waited for) to end.
    done.wait(None if timeout is None else timeout + _OFF_THREAD_GRACE_S)


def flush_pending_settles(timeout: Optional[float] = 10.0) -> None:
    """Finish the calls handed off by finalizers, then wait (up to ``timeout`` seconds in all)
    for every background settle started so far. The work runs off this thread (see
    _run_off_this_thread())."""
    caller = threading.get_ident()
    _run_off_this_thread(lambda: _flush(timeout, caller), timeout)


def _flush(timeout: Optional[float], caller: int) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout

    def remaining() -> Optional[float]:
        return None if deadline is None else max(deadline - time.monotonic(), 0.0)

    run_handed_off()
    with _settle_lock:
        calls = list(_open_calls)
    for call in calls:
        # A dropped call is finished here if the worker has not taken it; one being finalized
        # elsewhere is waited for, as its settle starts when it ends.
        finish_pending = getattr(call, "finish_pending", None)
        if finish_pending is not None:
            try:
                finish_pending(remaining(), caller=caller)
            except Exception:
                pass
    with _settle_lock:
        pending = list(_pending_settles)
    if pending:
        wait(pending, timeout=remaining())


# LLM calls still in flight: an open stream, or a request that has not returned. Each has
# settle_at_shutdown(timeout, caller) and may have check_pending (True while its policy check
# runs) and finish_pending(timeout, caller); ``caller`` is the thread that asked, which the
# work runs off. A call stays here until its finalization has ended. Weak, so
# a call that is dropped and collected is not kept alive here.
_open_calls: "weakref.WeakSet[Any]" = weakref.WeakSet()
_open_calls_hook = False


def track_open_call(call: Any) -> None:
    """Register a call in flight; settle_open_calls() settles it if it is still open then."""
    global _open_calls_hook
    with _settle_lock:
        _open_calls.add(call)
        if not _open_calls_hook:
            # start_tracing()'s exit hook also settles open calls before it flushes spans, so
            # this one settles calls whichever of the two runs first, or without tracing.
            atexit.register(settle_open_calls)
            _open_calls_hook = True


def untrack_open_call(call: Any) -> None:
    with _settle_lock:
        _open_calls.discard(call)


def settle_open_calls(timeout: float = 10.0) -> None:
    """Settle the calls still in flight, then wait for every pending settle, all within
    ``timeout`` seconds. Each call is stopped where possible and charged what it reported,
    or its reservation when that is unknown. Run by stop_tracing() and at interpreter exit.

    Calls dropped unfinished whose finalization was handed off are finalized first, as
    dropped. A call whose policy check is still running is settled once the check returns,
    so the reservation it makes is settled; those calls go last, as they may wait for their
    checks (see settle_at_shutdown(timeout)). The work runs off this thread (see
    _run_off_this_thread()), and is waited for.
    """
    caller = threading.get_ident()
    _run_off_this_thread(lambda: _settle_open_calls(timeout, caller), timeout)


def _settle_open_calls(timeout: float, caller: int) -> None:
    deadline = time.monotonic() + timeout
    run_handed_off()
    with _settle_lock:
        calls = list(_open_calls)
    calls.sort(key=lambda call: bool(getattr(call, "check_pending", False)))
    for done, call in enumerate(calls):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "%d open LLM call(s) not settled before the shutdown time limit", len(calls) - done
            )
            break
        try:
            call.settle_at_shutdown(timeout=remaining, caller=caller)
        except Exception as exc:
            logger.debug("settling an open call at shutdown failed: %s", exc)
    _flush(max(deadline - time.monotonic(), 0.0), caller)


def _reset_after_fork() -> None:
    """In a forked child: the parent's worker threads are gone, its locks may be held, and its
    calls in flight are the parent's to settle."""
    global _settle_lock, _settle_executor, _handed_off, _handoff_worker, _handoff_pid
    global _offloop_lock, _offloop_executor
    _settle_lock = threading.RLock()
    _settle_executor = None
    _offloop_lock = threading.Lock()
    _offloop_executor = None
    _pending_settles.clear()
    _open_calls.clear()
    _handed_off = queue.SimpleQueue()
    _handoff_worker = _handoff_pid = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


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
