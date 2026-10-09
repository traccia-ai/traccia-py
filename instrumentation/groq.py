"""Groq monkey patching for chat.completions.create (sync, async and streaming)."""

from __future__ import annotations

import contextvars
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from traccia.instrumentation.openai import _compute_cost, _safe_get
from traccia.tracer.span import SpanStatus

_patched = False

_SPAN_NAME = "llm.groq.chat.completions"


def _get_tracer(name: str) -> Any:
    """Return a tracer from the global provider."""
    import traccia

    return traccia.get_tracer(name)


def _slim_messages(kwargs: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Return up to 50 messages reduced to role and string content."""
    messages = kwargs.get("messages")
    if not isinstance(messages, (list, tuple)):
        return None

    slim = []
    for m in list(messages)[:50]:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            content = str(content)
        slim.append({"role": m.get("role"), "content": content})
    return slim or None


def _prompt_text(slim: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Join slim messages into "role: content" lines."""
    if not slim:
        return None
    parts = [
        f"{m['role']}: {m['content']}" if m.get("role") else str(m["content"])
        for m in slim
        if m.get("content")
    ]
    return "\n".join(parts) or None


def _record_metrics(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    duration: Optional[float],
    cost: Optional[float],
) -> None:
    """Record LLM metrics if metrics are enabled."""
    try:
        from traccia.metrics.recorder import get_metrics_recorder

        recorder = get_metrics_recorder()
        if not recorder:
            return

        attrs: Dict[str, Any] = {"gen_ai.system": "groq"}
        if model:
            attrs["gen_ai.request.model"] = model

        # Per-run agent identity, so ingestion attributes the data point to the right agent.
        try:
            from traccia import runtime_config as _rc

            _aid = _rc.get_agent_id()
            _aname = _rc.get_agent_name()
            _env = _rc.get_env()
            if _aid:
                attrs["agent.id"] = _aid
                attrs["agent_id"] = _aid
            if _aname:
                attrs["agent.name"] = _aname
            if _env:
                attrs["environment"] = _env
        except Exception:
            pass

        if prompt_tokens is not None or completion_tokens is not None:
            recorder.record_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attributes=attrs,
            )
        if duration is not None:
            recorder.record_duration(duration, attributes=attrs)
        if cost is not None:
            recorder.record_cost(cost, attributes=attrs)
    except Exception:
        pass


def _start_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build the span attributes known before the call is made."""
    slim = _slim_messages(kwargs)
    attrs: Dict[str, Any] = {"llm.vendor": "groq"}
    if kwargs.get("model"):
        attrs["llm.model"] = kwargs["model"]
    if slim:
        attrs["llm.groq.messages"] = json.dumps(slim)[:1000]
    text = _prompt_text(slim)
    if text:
        attrs["llm.prompt"] = text
    return attrs


def _first_choice(obj: Any) -> Any:
    """Return choices[0] of a completion or chunk, or None."""
    choices = _safe_get(obj, "choices")
    return choices[0] if isinstance(choices, (list, tuple)) and choices else None


def _record_result(
    span: Any,
    model: Optional[str],
    resp_model: Optional[str],
    usage: Any,
    reason: Optional[str],
    completion: Optional[str],
    t0: float,
    decision: Optional[Dict[str, Any]],
) -> None:
    """Set usage, finish reason and completion on the span, then record metrics and settle cost."""
    from traccia.governance.pep import finish_llm_call

    if resp_model and "llm.model" not in span.attributes:
        span.set_attribute("llm.model", resp_model)

    pt = ct = None
    if usage:
        span.set_attribute("llm.usage.source", "provider_usage")
        pt = _safe_get(usage, "prompt_tokens")
        ct = _safe_get(usage, "completion_tokens")
        if pt is not None:
            span.set_attribute("llm.usage.prompt_tokens", pt)
            span.set_attribute("llm.usage.prompt_source", "provider_usage")
        if ct is not None:
            span.set_attribute("llm.usage.completion_tokens", ct)
            span.set_attribute("llm.usage.completion_source", "provider_usage")
        total = _safe_get(usage, "total_tokens")
        if total is not None:
            span.set_attribute("llm.usage.total_tokens", total)

    if reason:
        span.set_attribute("llm.finish_reason", reason)
    if completion:
        span.set_attribute("llm.completion", completion)

    cost = _compute_cost(resp_model or model, pt, ct)
    _record_metrics(resp_model or model, pt, ct, time.perf_counter() - t0, cost)
    finish_llm_call(decision, actual_usd=cost)


def _record_response(
    span: Any,
    resp: Any,
    model: Optional[str],
    t0: float,
    decision: Optional[Dict[str, Any]],
) -> None:
    """Record a non-streamed ChatCompletion on the span."""
    choice = _first_choice(resp)
    _record_result(
        span,
        model,
        _safe_get(resp, "model"),
        _safe_get(resp, "usage"),
        _safe_get(choice, "finish_reason"),
        _safe_get(choice, "message.content"),
        t0,
        decision,
    )


def _record_failure(
    span: Any,
    exc: BaseException,
    decision: Optional[Dict[str, Any]],
    model: Optional[str],
) -> None:
    """Release the governance reservation and mark the span and metrics as failed."""
    try:
        from traccia.governance.pep import finish_llm_call

        finish_llm_call(decision, release=True)
    except Exception:
        pass
    span.record_exception(exc)
    span.set_status(SpanStatus.ERROR, str(exc) or type(exc).__name__)
    try:
        from traccia.metrics.recorder import get_metrics_recorder

        rec = get_metrics_recorder()
        if rec:
            rec.record_exception(
                attributes={
                    "gen_ai.system": "groq",
                    "gen_ai.request.model": model or "unknown",
                }
            )
    except Exception:
        pass


class _StreamState:
    """Accumulates chunks of a streamed completion and closes the span exactly once."""

    def __init__(
        self,
        span: Any,
        model: Optional[str],
        t0: float,
        decision: Optional[Dict[str, Any]],
        ctx: contextvars.Context,
    ) -> None:
        self.span, self.model, self.t0, self.decision = span, model, t0, decision
        # The context of the create() call: agent identity, pep_enabled and this span
        # as the current span. Finishing inside it settles and records metrics for the
        # agent that made the call, wherever and whenever the stream is consumed.
        self.ctx = ctx
        self.parts: List[str] = []
        self.resp_model = self.usage = self.reason = None
        self.done = False
        self._lock = threading.Lock()

    def on_chunk(self, chunk: Any) -> None:
        """Collect content, finish reason and usage from one ChatCompletionChunk."""
        try:
            self.resp_model = self.resp_model or _safe_get(chunk, "model")
            # Groq puts usage on the last chunk under x_groq.usage; chunk.usage is set
            # instead when the caller passes stream_options={"include_usage": True}.
            usage = _safe_get(chunk, "usage") or _safe_get(chunk, "x_groq.usage")
            if usage:
                self.usage = usage
            choice = _first_choice(chunk)
            content = _safe_get(choice, "delta.content")
            if content:
                self.parts.append(content)
            reason = _safe_get(choice, "finish_reason")
            if reason:
                self.reason = reason
        except Exception:
            pass

    def finish(self, exc: Optional[BaseException] = None) -> None:
        """Record what was streamed (or *exc*) and end the span; later calls do nothing."""
        with self._lock:
            if self.done:
                return
            self.done = True
        try:
            self.ctx.run(self._finish, exc)
        except RuntimeError:
            # The context is already entered on this thread; finish in place.
            self._finish(exc)

    def _finish(self, exc: Optional[BaseException]) -> None:
        try:
            if exc is not None:
                _record_failure(self.span, exc, self.decision, self.model)
            else:
                _record_result(
                    self.span,
                    self.model,
                    self.resp_model,
                    self.usage,
                    self.reason,
                    "".join(self.parts) or None,
                    self.t0,
                    self.decision,
                )
        except Exception:
            pass
        finally:
            self.span.end()


class _SyncStream:
    """Proxy for groq.Stream; records the span when the stream is exhausted, closed or dropped."""

    def __init__(self, stream: Any, state: _StreamState) -> None:
        self._stream, self._state = stream, state

    def __iter__(self) -> "_SyncStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._state.finish()
            raise
        except BaseException as exc:  # includes KeyboardInterrupt mid-stream
            self._state.finish(exc)
            raise
        self._state.on_chunk(chunk)
        return chunk

    def __enter__(self) -> "_SyncStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying stream and end the span."""
        try:
            self._stream.close()
        finally:
            self._state.finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __del__(self) -> None:
        # Abandoned before exhaustion or close(), e.g. `break` out of the loop.
        try:
            self._state.finish()
        except Exception:
            pass


class _AsyncStream:
    """Proxy for groq.AsyncStream; records the span when the stream is exhausted, closed or dropped."""

    def __init__(self, stream: Any, state: _StreamState) -> None:
        self._stream, self._state = stream, state

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._state.finish()
            raise
        except BaseException as exc:  # includes asyncio.CancelledError
            self._state.finish(exc)
            raise
        self._state.on_chunk(chunk)
        return chunk

    async def __aenter__(self) -> "_AsyncStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying stream and end the span."""
        try:
            await self._stream.close()
        finally:
            self._state.finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __del__(self) -> None:
        # Abandoned before exhaustion or close(), e.g. `break` out of the loop.
        try:
            self._state.finish()
        except Exception:
            pass


def _wrap_sync(create_fn: Callable) -> Callable:
    """Wrap Completions.create; returns *create_fn* unchanged if already wrapped."""
    if getattr(create_fn, "_agent_trace_patched", False):
        return create_fn

    def wrapped(self, *args, **kwargs):
        from traccia.context.context import pop_span, push_span
        from traccia.governance.pep import enforce_llm_call

        t0 = time.perf_counter()
        # Not a `with` block: a streamed call must keep the span open after we return.
        span = _get_tracer("groq").start_span(
            _SPAN_NAME, attributes=_start_attrs(kwargs)
        )
        token = push_span(span)
        decision = None
        try:
            decision = enforce_llm_call(kwargs)
            if kwargs.get("model"):
                span.set_attribute("llm.model", kwargs["model"])
            resp = create_fn(self, *args, **kwargs)
            if kwargs.get("stream") is True:
                span.set_attribute("llm.streaming", True)
                state = _StreamState(
                    span, kwargs.get("model"), t0, decision, contextvars.copy_context()
                )
                return _SyncStream(resp, state)
            _record_response(span, resp, kwargs.get("model"), t0, decision)
            span.end()
            return resp
        except BaseException as exc:
            # BaseException, not Exception: asyncio.CancelledError and KeyboardInterrupt
            # must also end the span and release the governance reservation.
            _record_failure(span, exc, decision, kwargs.get("model"))
            span.end()
            raise
        finally:
            pop_span(token)

    wrapped._agent_trace_patched = True
    return wrapped


def _wrap_async(create_fn: Callable) -> Callable:
    """Wrap AsyncCompletions.create; returns *create_fn* unchanged if already wrapped."""
    if getattr(create_fn, "_agent_trace_patched", False):
        return create_fn

    async def wrapped(self, *args, **kwargs):
        from traccia.context.context import pop_span, push_span
        from traccia.governance.pep import enforce_llm_call

        t0 = time.perf_counter()
        # Not a `with` block: a streamed call must keep the span open after we return.
        span = _get_tracer("groq").start_span(
            _SPAN_NAME, attributes=_start_attrs(kwargs)
        )
        token = push_span(span)
        decision = None
        try:
            decision = enforce_llm_call(kwargs)
            if kwargs.get("model"):
                span.set_attribute("llm.model", kwargs["model"])
            resp = await create_fn(self, *args, **kwargs)
            if kwargs.get("stream") is True:
                span.set_attribute("llm.streaming", True)
                state = _StreamState(
                    span, kwargs.get("model"), t0, decision, contextvars.copy_context()
                )
                return _AsyncStream(resp, state)
            _record_response(span, resp, kwargs.get("model"), t0, decision)
            span.end()
            return resp
        except BaseException as exc:
            # BaseException, not Exception: asyncio.CancelledError and KeyboardInterrupt
            # must also end the span and release the governance reservation.
            _record_failure(span, exc, decision, kwargs.get("model"))
            span.end()
            raise
        finally:
            pop_span(token)

    wrapped._agent_trace_patched = True
    return wrapped


def patch_groq() -> bool:
    """
    Patch Groq chat.completions.create for sync and async clients.

    Each call gets an ``llm.groq.chat.completions`` span with the prompt, completion,
    token usage and cost. With ``stream=True`` the span stays open until the returned
    stream is exhausted, closed or garbage collected.

    Returns:
        True if Groq is patched (now or by an earlier call), False if the groq
        package is not installed.
    """
    global _patched
    if _patched:
        return True
    try:
        from groq.resources.chat.completions import AsyncCompletions, Completions
    except Exception:
        return False

    Completions.create = _wrap_sync(Completions.create)
    AsyncCompletions.create = _wrap_async(AsyncCompletions.create)
    _patched = True
    return True
