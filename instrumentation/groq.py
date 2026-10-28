"""Groq (groq) monkey patching for chat.completions.create, sync and async.

Patches ``Completions.create`` and ``AsyncCompletions.create`` so every call gets an
``llm.groq.chat.completions`` span and goes through the governance policy check.

- Streaming (``stream=True``): the returned ``Stream``/``AsyncStream`` is groq's own object.
  Its iterator is wrapped, so the span stays open until the stream is read to the end,
  stopped early, closed, fails, or is garbage collected, and then records the completion
  and token usage.
- ``with_raw_response`` keeps working: the parsed body is read through the response's own
  cache, so the caller's ``parse()`` returns the same object.
- Finalization (span attributes, governance settlement, metrics) runs in a copy of the
  ``create()`` call's context, so it belongs to the agent and span that made the call,
  whichever context consumes the stream.
- Settlement charges the Spend Cap what the call cost: its token usage priced at Groq's price
  (or a pricing override), an estimate priced the same way when the response or stream was
  read but carried no usage, or the full reservation when the cost is unknown: nothing of the
  response was read through the SDK, the stream was dropped unfinished, or the call was still
  open at shutdown. Only a call that failed before any output releases its reservation. The
  settle request runs on a worker thread, so a slow policy server does not delay the caller.
- A stream dropped unfinished, or still open when stop_tracing() runs or the interpreter
  exits, has its HTTP response closed so Groq stops generating it.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import math
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

from traccia.tracer.span import SpanStatus

_patched = False

VENDOR = "groq"
SPAN_NAME = "llm.groq.chat.completions"
# groq sets this request header to "stream" for with_streaming_response, whose body the
# caller reads; for with_raw_response it is "true" and the body is already read.
_RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"
_SUCCESS = "success"
_ERROR = "error"
# A stream garbage-collected before it was finished or closed.
_ABANDONED = "abandoned"
# A call still in flight when stop_tracing() runs or the interpreter exits.
_SHUTDOWN = "shutdown"
_INTERRUPTED = (_ABANDONED, _SHUTDOWN)
_UNSET = object()
# Response closes scheduled for abandoned async streams; held until they finish.
_closing: "set[asyncio.Task]" = set()


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """Safely get a nested attribute or dict key (dot-separated path)."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur if cur is not None else default


def _first_choice(obj: Any) -> Any:
    choices = _safe_get(obj, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        return choices[0]
    return None


def _read_tokens(usage: Any) -> Optional[Tuple[int, int, int]]:
    """Prompt, completion and cached prompt token counts from a Groq usage object, if it has
    prompt and completion counts. Cached tokens (``prompt_tokens_details.cached_tokens``,
    billed at the cached input rate) are part of the prompt count; 0 when Groq reports none."""
    prompt = _safe_get(usage, "prompt_tokens")
    completion = _safe_get(usage, "completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    if isinstance(prompt, bool) or isinstance(completion, bool):
        return None
    cached = _safe_get(usage, "prompt_tokens_details.cached_tokens")
    if not isinstance(cached, int) or isinstance(cached, bool):
        cached = 0
    return prompt, completion, min(max(cached, 0), max(prompt, 0))


def _output_chars(message: Any) -> int:
    """Characters of output in a message or delta: its content, reasoning (billed as output
    tokens by reasoning models) and tool call arguments."""
    chars = 0
    for field in ("content", "reasoning"):
        text = _safe_get(message, field)
        if isinstance(text, str):
            chars += len(text)
    tool_calls = _safe_get(message, "tool_calls")
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            arguments = _safe_get(call, "function.arguments")
            if isinstance(arguments, str):
                chars += len(arguments)
    return chars


def _slim_messages(kwargs: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """JSON-friendly, small copies of the request messages."""
    messages = kwargs.get("messages")
    if not isinstance(messages, (list, tuple)):
        return None
    slim = []
    for message in list(messages)[:50]:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = str(content)
        slim.append({"role": message.get("role"), "content": content})
    return slim or None


def _prompt_text(slim: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not slim:
        return None
    parts = []
    for message in slim:
        content = message.get("content")
        if content:
            role = message.get("role")
            parts.append(f"{role}: {content}" if role else str(content))
    return "\n".join(parts) if parts else None


def _request_attributes(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {"llm.vendor": VENDOR}
    model = kwargs.get("model")
    if isinstance(model, str) and model:
        attributes["llm.model"] = model
    slim = _slim_messages(kwargs)
    if slim:
        try:
            attributes["llm.groq.messages"] = json.dumps(slim)[:2000]
        except Exception:
            attributes["llm.groq.messages"] = str(slim)[:2000]
    prompt = _prompt_text(slim)
    if prompt:
        attributes["llm.prompt"] = prompt
    if kwargs.get("stream") is True:
        attributes["llm.streaming"] = True
    return attributes


def _compute_cost(
    model: Optional[str], prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
) -> Optional[float]:
    """USD cost at Groq's price, or None when the pricing table has no Groq entry for the model
    (the server then prices the tokens). Only Groq's own entries, or a user override that fits
    them, are used (see pricing_matcher), from the table spans are priced with, so the settled cost and
    the span's llm.cost.usd agree. ``cached_tokens`` are part of ``prompt_tokens``."""
    if not model:
        return None
    try:
        from traccia.cost_resolver import get_resolver
        from traccia.processors.cost_engine import compute_cost as _compute

        return _compute(
            model,
            prompt_tokens - cached_tokens,
            completion_tokens,
            get_resolver().pricing_table,
            cache_read_tokens=cached_tokens,
            vendor=VENDOR,
        )
    except Exception:
        return None


def _metric_attributes(model: Optional[str]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {"gen_ai.system": VENDOR}
    if model:
        attributes["gen_ai.request.model"] = model
    try:
        from traccia import runtime_config as _rc

        agent_id = _rc.get_agent_id()
        agent_name = _rc.get_agent_name()
        env = _rc.get_env()
        if agent_id:
            attributes["agent.id"] = agent_id
            attributes["agent_id"] = agent_id
        if agent_name:
            attributes["agent.name"] = agent_name
        if env:
            attributes["environment"] = env
    except Exception:
        pass
    return attributes


def _record_metrics(kind: str, model: Optional[str], tokens: Optional[Tuple[int, int, int]], duration: float, cost: Optional[float]) -> None:
    """Token, duration, cost and exception metrics. Tokens and cost only from Groq's usage."""
    try:
        from traccia.metrics.recorder import get_metrics_recorder

        recorder = get_metrics_recorder()
        if not recorder:
            return
        attributes = _metric_attributes(model)
        if tokens:
            recorder.record_token_usage(
                prompt_tokens=tokens[0], completion_tokens=tokens[1], attributes=attributes
            )
        recorder.record_duration(duration, attributes=attributes)
        if cost is not None:
            recorder.record_cost(cost, attributes=attributes)
        if kind == _ERROR:
            recorder.record_exception(attributes=attributes)
    except Exception:
        pass


class _CallState:
    """Span and governance state for one create() call. Holds no reference to the stream,
    only its HTTP response, so a stream's garbage-collection finalizer can hold it."""

    def __init__(self, span: Any, kwargs: Dict[str, Any]) -> None:
        self.span = span
        self.decision: Optional[Dict[str, Any]] = None
        # Context of the create() call: run identity, PEP flag, and this span as current.
        self.ctx: Optional[contextvars.Context] = None
        self.request_model: Optional[str] = None
        self.prompt_tokens = _estimate_prompt_tokens(kwargs)
        self._t0 = time.perf_counter()
        # Reentrant: a garbage-collection finalizer can run on this thread mid-check.
        self._lock = threading.RLock()
        self._done = False
        self._abandoned: Optional[weakref.finalize] = None
        # The stream's HTTP response, closed to stop a stream that is abandoned or still open
        # at shutdown.
        self._response: Any = None
        self._async = False
        self._parts: List[str] = []
        self._chars = 0
        self._chunks = 0
        self._usage: Any = None
        self._resp_model: Any = None
        self._finish_reason: Any = None

    # -- streams -------------------------------------------------------------

    def on_chunk(self, chunk: Any) -> None:
        """Collect content, finish reason and usage from one ChatCompletionChunk."""
        self._chunks += 1
        try:
            self._resp_model = self._resp_model or _safe_get(chunk, "model")
            # Groq puts usage on the last chunk under x_groq.usage; chunk.usage is set instead
            # with stream_options={"include_usage": True}.
            usage = _safe_get(chunk, "usage") or _safe_get(chunk, "x_groq.usage")
            if usage is not None:
                self._usage = usage
            choice = _first_choice(chunk)
            delta = _safe_get(choice, "delta")
            content = _safe_get(delta, "content")
            if isinstance(content, str) and content:
                self._parts.append(content)
            self._chars += _output_chars(delta)
            finish_reason = _safe_get(choice, "finish_reason")
            if finish_reason:
                self._finish_reason = finish_reason
        except Exception:
            # Never let span bookkeeping break the caller's stream.
            pass

    def trace_stream(self, stream: Any) -> None:
        """Wrap a Stream so the span ends when the stream does."""
        stream._iterator = _traced_iterator(stream._iterator, self)
        original_close = stream.close

        def close() -> None:
            try:
                original_close()
            finally:
                self.finalize(_SUCCESS)

        stream.close = close
        self._response = getattr(stream, "response", None)
        self._abandoned = weakref.finalize(stream, self.finalize, _ABANDONED)

    def trace_async_stream(self, stream: Any) -> None:
        """Wrap an AsyncStream so the span ends when the stream does."""
        stream._iterator = _traced_async_iterator(stream._iterator, self)
        original_close = stream.close

        async def close() -> None:
            try:
                await original_close()
            finally:
                self.finalize(_SUCCESS)

        stream.close = close
        self._response = getattr(stream, "response", None)
        self._async = True
        self._abandoned = weakref.finalize(stream, self.finalize, _ABANDONED)

    def settle_at_shutdown(self) -> None:
        """Stop and settle the call if it is still open (see pep.settle_open_calls)."""
        self.finalize(_SHUTDOWN)

    def _abort(self) -> None:
        """Close the stream's HTTP response, so Groq stops generating (and billing) it."""
        response = self._response
        if response is None:
            return
        try:
            if not self._async:
                response.close()
                return
            # An async response closes on its own event loop; without one running here (at
            # interpreter exit) it cannot be closed.
            task = asyncio.get_running_loop().create_task(response.aclose())
            _closing.add(task)
            task.add_done_callback(_closing.discard)
        except Exception:
            pass

    # -- finalization --------------------------------------------------------

    def finalize(self, kind: str, response: Any = _UNSET, error: Optional[BaseException] = None) -> None:
        """Record the outcome, settle the reservation, record metrics, end the span.

        Only the first call does anything. Never raises.
        """
        with self._lock:
            if self._done:
                return
            self._done = True
        if self._abandoned is not None:
            self._abandoned.detach()
        _untrack(self)

        def run() -> None:
            self._finish(kind, response, error)

        if self.ctx is None:
            run()
            return
        try:
            self.ctx.run(run)
        except RuntimeError:
            # The context is already entered on this thread; run in the current one.
            run()

    def _finish(self, kind: str, response: Any, error: Optional[BaseException]) -> None:
        span = self.span
        try:
            if kind in _INTERRUPTED:
                self._abort()
                span.set_attribute("llm.interrupted", kind)
            if kind == _SUCCESS and response is not _UNSET:
                choice = _first_choice(response)
                message = _safe_get(choice, "message")
                usage = _safe_get(response, "usage")
                resp_model = _safe_get(response, "model")
                chars = _output_chars(message)
                read = True
                content = _safe_get(message, "content")
                completion = content if isinstance(content, str) else None
                finish_reason = _safe_get(choice, "finish_reason")
            else:
                # A stream, or a call whose response was not read. After an error this keeps
                # the text and any usage that arrived, as a stopped loop does.
                usage = self._usage
                resp_model = self._resp_model
                chars = self._chars
                read = self._chunks > 0
                completion = "".join(self._parts) or None
                finish_reason = self._finish_reason if kind == _SUCCESS else None

            tokens = _read_tokens(usage)
            self._record_span(tokens, usage, resp_model, completion, finish_reason)
            if kind == _ERROR:
                if error is not None:
                    span.record_exception(error)
                span.set_status(SpanStatus.ERROR, str(error) if error is not None else "error")

            model = resp_model if isinstance(resp_model, str) and resp_model else self.request_model
            settle = self._settlement(kind, tokens, model, chars, read)
            _record_metrics(
                kind,
                model,
                tokens,
                time.perf_counter() - self._t0,
                settle.get("actual_usd") if tokens else None,
            )
            try:
                from traccia.governance.pep import finish_llm_call_in_background

                # Started here, inside the call's context, so it settles for the agent and
                # span that made the call.
                finish_llm_call_in_background(self.decision, **settle)
            except Exception:
                pass
        except Exception:
            # Recording is best-effort; the span still ends below.
            pass
        finally:
            try:
                span.end()
            except Exception:
                pass

    def _record_span(self, tokens, usage, resp_model, completion, finish_reason) -> None:
        span = self.span
        if isinstance(resp_model, str) and resp_model:
            # The model Groq ran and billed, which settlement prices too, so the span's cost
            # agrees with the settled cost. A different requested name is kept alongside.
            requested = getattr(span, "attributes", {}).get("llm.model")
            if requested and requested != resp_model:
                span.set_attribute("llm.request.model", requested)
            span.set_attribute("llm.model", resp_model)
        if usage is not None:
            span.set_attribute("llm.usage.source", "provider_usage")
            if tokens:
                # Prompt tokens are the uncached input, as the cost processor and the platform
                # price them; cached input is priced at its own rate.
                span.set_attribute("llm.usage.prompt_tokens", tokens[0] - tokens[2])
                span.set_attribute("llm.usage.input_tokens", tokens[0] - tokens[2])
                if tokens[2]:
                    span.set_attribute("llm.usage.cache_read_tokens", tokens[2])
                span.set_attribute("llm.usage.prompt_source", "provider_usage")
                span.set_attribute("llm.usage.completion_tokens", tokens[1])
                span.set_attribute("llm.usage.output_tokens", tokens[1])
                span.set_attribute("llm.usage.completion_source", "provider_usage")
            total = _safe_get(usage, "total_tokens")
            if isinstance(total, int):
                span.set_attribute("llm.usage.total_tokens", total)
        if completion:
            span.set_attribute("llm.completion", completion[:4096])
        if finish_reason:
            span.set_attribute("llm.finish_reason", str(finish_reason))

    def _settlement(
        self, kind: str, tokens: Optional[Tuple[int, int, int]], model: Optional[str], chars: int, read: bool
    ) -> Dict[str, Any]:
        """What to settle the reservation with. Groq bills the tokens it generated, also for
        a stream that was stopped, closed or failed partway. ``read``: whether any of the
        response reached the caller."""
        from traccia.governance.pep import reserved_usd

        if tokens:
            return {
                "actual_usd": _compute_cost(model, tokens[0], tokens[1], tokens[2]),
                "model": model,
                "vendor": VENDOR,
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
            }
        if kind in _INTERRUPTED:
            # Dropped unfinished, or still open at shutdown: Groq bills what it generated until
            # the request stopped, which is unknown without usage, so keep the reservation.
            return {"actual_usd": reserved_usd(self.decision)}
        if read or chars > 0:
            # The response was read but carried no usage (or the stream stopped before it):
            # estimate the output from the text, at least one token, and the prompt from the
            # request. Priced like reported usage, so a pricing override applies; the server
            # prices it when the table has no Groq entry, and only when both counts are present.
            input_tokens = self.prompt_tokens or 0
            output_tokens = max(math.ceil(chars / 4), 1)
            return {
                "actual_usd": _compute_cost(model, input_tokens, output_tokens),
                "model": model,
                "vendor": VENDOR,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        if kind == _SUCCESS:
            # The call completed but nothing of the response was read through the SDK, so its
            # cost is unknown: charge the full reservation rather than nothing.
            return {"actual_usd": reserved_usd(self.decision)}
        return {"release": True}


def _track(state: _CallState) -> None:
    try:
        from traccia.governance.pep import track_open_call

        track_open_call(state)
    except Exception:
        pass


def _untrack(state: _CallState) -> None:
    try:
        from traccia.governance.pep import untrack_open_call

        untrack_open_call(state)
    except Exception:
        pass


def _estimate_prompt_tokens(kwargs: Dict[str, Any]) -> Optional[int]:
    try:
        from traccia.governance.pep import estimate_prompt_tokens

        return estimate_prompt_tokens(kwargs)
    except Exception:
        return None


def _traced_iterator(inner: Any, state: _CallState):
    """Record each chunk; finalize when the stream ends, fails, or is stopped early."""
    try:
        for chunk in inner:
            state.on_chunk(chunk)
            yield chunk
    except GeneratorExit:
        # Only garbage collection closes this generator: Stream.close() closes the response,
        # not the iterator. So the stream was dropped unfinished.
        state.finalize(_ABANDONED)
        raise
    except BaseException as exc:
        state.finalize(_ERROR, error=exc)
        raise
    state.finalize(_SUCCESS)


async def _traced_async_iterator(inner: Any, state: _CallState):
    """Async version of _traced_iterator."""
    try:
        async for chunk in inner:
            state.on_chunk(chunk)
            yield chunk
    except GeneratorExit:
        state.finalize(_ABANDONED)
        raise
    except BaseException as exc:
        state.finalize(_ERROR, error=exc)
        raise
    state.finalize(_SUCCESS)


def _stream_classes() -> Tuple[type, ...]:
    try:
        from groq._streaming import AsyncStream, Stream

        return Stream, AsyncStream
    except Exception:
        return ()


def _response_classes() -> Tuple[type, ...]:
    try:
        from groq._response import APIResponse, AsyncAPIResponse

        return APIResponse, AsyncAPIResponse
    except Exception:
        return ()


def _is_streamed_response(resp: Any) -> bool:
    """True for with_streaming_response, whose body the caller reads."""
    try:
        return resp.http_response.request.headers.get(_RAW_RESPONSE_HEADER) == "stream"
    except Exception:
        return False


def _trace_result(resp: Any, state: _CallState) -> None:
    """Finalize the span from a sync create() result, or keep it open for a stream."""
    streams = _stream_classes()
    responses = _response_classes()
    if streams and isinstance(resp, streams):
        state.trace_stream(resp)
        return
    if responses and isinstance(resp, responses):
        if _is_streamed_response(resp):
            state.finalize(_SUCCESS)
            return
        try:
            parsed = resp.parse()  # cached by the response; the caller's parse() returns it
        except Exception:
            state.finalize(_SUCCESS)
            return
        _trace_result(parsed, state)
        return
    state.finalize(_SUCCESS, response=resp)


async def _trace_async_result(resp: Any, state: _CallState) -> None:
    """Async version of _trace_result."""
    streams = _stream_classes()
    responses = _response_classes()
    if streams and isinstance(resp, streams):
        state.trace_async_stream(resp)
        return
    if responses and isinstance(resp, responses):
        if _is_streamed_response(resp):
            state.finalize(_SUCCESS)
            return
        try:
            parsed = await resp.parse()
        except Exception:
            state.finalize(_SUCCESS)
            return
        await _trace_async_result(parsed, state)
        return
    state.finalize(_SUCCESS, response=resp)


def _start(kwargs: Dict[str, Any]) -> Tuple[Any, _CallState, Any]:
    """Start the span, make it current, and capture the call's context."""
    from traccia.context import push_span

    span = _get_tracer("groq").start_span(SPAN_NAME, attributes=_request_attributes(kwargs))
    state = _CallState(span, kwargs)
    _track(state)
    token = push_span(span)
    state.ctx = contextvars.copy_context()
    return span, state, token


def _after_check(span: Any, state: _CallState, kwargs: Dict[str, Any]) -> None:
    # The policy check may have swapped the model (reshape).
    model = kwargs.get("model")
    if isinstance(model, str) and model:
        state.request_model = model
        span.set_attribute("llm.model", model)


def _build_sync_wrapper(original_create: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original_create)
    def sync_wrapped(self, *args, **kwargs):
        from traccia.context import pop_span
        from traccia.governance.pep import enforce_llm_call

        span, state, token = _start(kwargs)
        try:
            try:
                state.decision = enforce_llm_call(kwargs, vendor=VENDOR)
                _after_check(span, state, kwargs)
                resp = original_create(self, *args, **kwargs)
            except BaseException as exc:
                state.finalize(_ERROR, error=exc)
                raise
        finally:
            pop_span(token)
        try:
            _trace_result(resp, state)
        except Exception:
            state.finalize(_SUCCESS)
        return resp

    sync_wrapped._agent_trace_patched = True
    return sync_wrapped


def _build_async_wrapper(original_create: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original_create)
    async def async_wrapped(self, *args, **kwargs):
        from traccia.context import pop_span
        from traccia.governance.pep import enforce_llm_call

        span, state, token = _start(kwargs)
        try:
            try:
                state.decision = enforce_llm_call(kwargs, vendor=VENDOR)
                _after_check(span, state, kwargs)
                resp = await original_create(self, *args, **kwargs)
            except BaseException as exc:
                state.finalize(_ERROR, error=exc)
                raise
        finally:
            pop_span(token)
        try:
            await _trace_async_result(resp, state)
        except Exception:
            state.finalize(_SUCCESS)
        return resp

    async_wrapped._agent_trace_patched = True
    return async_wrapped


def patch_groq() -> bool:
    """Patch groq's chat.completions.create (sync and async). Returns True if patched.

    Soft-fails (returns False) if groq isn't installed. Clients created before patching are
    covered too; a ``with_raw_response``/``with_streaming_response`` accessor read before
    patching keeps the unpatched method.
    """
    global _patched
    if _patched:
        return True
    try:
        from groq.resources.chat.completions import AsyncCompletions, Completions
    except Exception:
        return False

    hit = False
    for cls, build in ((Completions, _build_sync_wrapper), (AsyncCompletions, _build_async_wrapper)):
        original = getattr(cls, "create", None)
        if original is None:
            continue
        if not getattr(original, "_agent_trace_patched", False):
            setattr(cls, "create", build(original))
        hit = True
    _patched = hit
    return hit


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)
