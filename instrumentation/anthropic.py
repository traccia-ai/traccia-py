"""Instrumentation for Anthropic Messages, including both streaming APIs."""
from __future__ import annotations

import functools
import importlib
import inspect
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Mapping, Optional, Tuple

from traccia.tracer.span import SpanStatus

_patched = False
_MAX_ITEMS = 50


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_number(value: Any) -> bool:
    """True for real numeric token counts, excluding bool (a subclass of int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def _limit() -> int:
    try:
        from traccia import runtime_config
        return int(runtime_config.get_attr_truncation_limit() or 1000)
    except Exception:
        return 1000


def _safe_value(value: Any, depth: int = 0) -> Any:
    """Return bounded JSON-safe data, omitting credential/header/binary fields."""
    if depth > 4 or isinstance(value, (bytes, bytearray, memoryview)):
        return "[omitted]"
    if isinstance(value, str):
        return value[:_limit()]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        blocked = {"api_key", "apikey", "authorization", "headers"}
        return {str(k): _safe_value(v, depth + 1) for k, v in list(value.items())[:_MAX_ITEMS]
                if str(k).lower() not in blocked}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v, depth + 1) for v in list(value)[:_MAX_ITEMS]]
    for method in ("model_dump", "to_dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return _safe_value(fn(), depth + 1)
            except Exception:
                pass
    return str(value)[:_limit()]


def _json_attr(value: Any) -> Optional[str]:
    if value is None:
        return None
    limit = _limit()
    try:
        encoded = json.dumps(_safe_value(value), ensure_ascii=False)
    except Exception:
        try:
            return json.dumps({"repr": str(value)[:limit]}, ensure_ascii=False)
        except Exception:
            return None
    if len(encoded) <= limit:
        return encoded
    try:
        return json.dumps({"truncated": True, "preview": encoded[:limit]}, ensure_ascii=False)
    except Exception:
        return None


def _request_model(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    return kwargs.get("model") or _safe_get(args, "0.model")


def _request_payload(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    messages = kwargs.get("messages")
    if messages is None and len(args) >= 2:
        messages = args[1]
    return {k: v for k, v in (("messages", messages), ("system", kwargs.get("system")),
                              ("tools", kwargs.get("tools"))) if v is not None}


def _content_text(content: Any) -> Optional[str]:
    if not isinstance(content, (list, tuple)):
        return str(content) if content is not None else None
    parts = []
    for block in content:
        typ = _field(block, "type")
        if typ == "text" and _field(block, "text") is not None:
            parts.append(str(_field(block, "text")))
        elif typ == "tool_use":
            name, bid = _field(block, "name"), _field(block, "id")
            parts.append("[tool_use{}{}]".format(" " + str(name) if name else "",
                                                 " (" + str(bid) + ")" if bid else ""))
    return "\n".join(parts) or None


def _usage(resp: Any) -> Dict[str, Any]:
    usage = _field(resp, "usage")
    if usage is None:
        return {}
    result: Dict[str, Any] = {}
    for name in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                 "cache_read_input_tokens", "thinking_tokens"):
        value = _field(usage, name)
        if _is_number(value):
            result[name] = value

    cache_creation = _field(usage, "cache_creation")
    if _is_number(cache_creation):
        result["cache_creation"] = cache_creation
    elif cache_creation is not None:
        for name in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
            value = _field(cache_creation, name)
            if _is_number(value):
                result["cache_creation_" + name] = value
    details = _field(usage, "output_tokens_details") or _field(usage, "thinking")
    thinking = _field(details, "thinking_tokens") if details is not None else None
    if _is_number(thinking):
        result["thinking_tokens"] = thinking
    return result


def _compute_cost(model: Any, input_tokens: Any, output_tokens: Any) -> Optional[float]:
    if not model or input_tokens is None or output_tokens is None:
        return None
    try:
        from traccia.processors.cost_engine import compute_cost
        from traccia.pricing_config import load_pricing
        return compute_cost(model, input_tokens, output_tokens, load_pricing())
    except Exception:
        return None


def _record_metrics(
    model: Any,
    usage: Mapping[str, Any],
    duration: float,
    cost: Optional[float] = None,
    error: bool = False,
) -> None:
    try:
        from traccia.metrics.recorder import get_metrics_recorder
        recorder = get_metrics_recorder()
        if not recorder:
            return
        attrs = {"gen_ai.system": "anthropic"}
        if model:
            attrs["gen_ai.request.model"] = model
        try:
            from traccia import runtime_config as rc
            for key, getter in (("agent.id", rc.get_agent_id), ("agent.name", rc.get_agent_name),
                                ("environment", rc.get_env)):
                value = getter()
                if value:
                    attrs[key] = value
                    if key == "agent.id":
                        attrs["agent_id"] = value
        except Exception:
            pass
        if usage:
            recorder.record_token_usage(
                usage.get("input_tokens"), usage.get("output_tokens"), attrs
            )
        recorder.record_duration(duration, attrs)
        if cost is not None:
            recorder.record_cost(cost, attrs)
        if error:
            recorder.record_exception({**attrs, "gen_ai.request.model": model or "unknown"})
    except Exception:
        pass


def _start_attrs(model: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {"llm.vendor": "anthropic", "gen_ai.system": "anthropic"}
    try:
        if model:
            attrs.update({"llm.model": model, "gen_ai.request.model": model})
        payload = _request_payload(args, kwargs)
        messages = payload.get("messages")
        if isinstance(messages, (list, tuple)):
            prompt_parts = []
            for message in messages[:_MAX_ITEMS]:
                role, content = _field(message, "role"), _field(message, "content")
                text = _content_text(content)
                if text:
                    prompt_parts.append((str(role) + ": " if role else "") + text)
            prompt = "\n".join(prompt_parts)
            if prompt:
                attrs["llm.prompt"] = prompt[:_limit()]
                attrs["gen_ai.prompt"] = prompt[:_limit()]
        for key, attr in (("messages", "llm.anthropic.messages"),
                          ("system", "llm.anthropic.system"),
                          ("tools", "llm.anthropic.tools")):
            encoded = _json_attr(payload.get(key))
            if encoded:
                attrs[attr] = encoded
    except Exception:
        # Request-attribute capture must never break the wrapped SDK call.
        pass
    return attrs


_UNSET = object()


@dataclass
class _StreamResult:
    response: Any
    completion: Optional[str]
    usage: Dict[str, Any]
    stop_reason: Any


class _StreamTraceState:
    """Accumulate Anthropic stream state behind one small internal interface."""

    def __init__(self) -> None:
        self._initial_response: Any = None
        self._last_message: Any = None
        self._completion_parts: list[str] = []
        self._usage: Dict[str, Any] = {}
        self._stop_reason: Any = _UNSET

    def observe_event(self, event: Any) -> None:
        if event is None:
            return

        event_type = _field(event, "type")
        message = _field(event, "message")
        if message is not None:
            self._last_message = message
            if event_type == "message_start":
                self._initial_response = message

        if event_type == "message_delta":
            delta = _field(event, "delta")
            stop_reason = _field(delta, "stop_reason")
            if stop_reason is not None:
                self._stop_reason = stop_reason
            self._usage.update(_usage(event))
        elif event_type == "content_block_start":
            block = _field(event, "content_block")
            if _field(block, "type") == "tool_use":
                name, block_id = _field(block, "name"), _field(block, "id")
                label = "[tool_use{}{}]".format(
                    " " + str(name) if name else "",
                    " (" + str(block_id) + ")" if block_id else "",
                )
                self._completion_parts.append(label)
        elif event_type == "content_block_delta":
            delta = _field(event, "delta")
            if _field(delta, "type") == "text_delta":
                text = _field(delta, "text")
                if text is not None:
                    self._completion_parts.append(str(text))

    def observe_text(self, item: Any) -> None:
        if isinstance(item, str):
            self._completion_parts.append(item)
            return
        text = _field(item, "text")
        if text is not None and _field(item, "type") in {"text", "text_delta"}:
            self._completion_parts.append(str(text))

    def resolve(
        self,
        response: Any = _UNSET,
        snapshot: Any = None,
        completion: Any = _UNSET,
    ) -> _StreamResult:
        selected_response = None
        if response is not _UNSET and response is not None and not isinstance(response, str):
            selected_response = response
        elif snapshot is not None:
            selected_response = snapshot
        elif self._last_message is not None:
            selected_response = self._last_message
        else:
            selected_response = self._initial_response

        response_content = _field(selected_response, "content")
        if completion is _UNSET:
            if response_content:
                resolved_completion = _content_text(response_content)
            else:
                resolved_completion = "".join(self._completion_parts) or None
        else:
            resolved_completion = completion

        resolved_usage = _usage(selected_response)
        resolved_usage.update(self._usage)
        stop_reason = self._stop_reason
        if stop_reason is _UNSET:
            stop_reason = _field(selected_response, "stop_reason", _UNSET)

        return _StreamResult(
            response=selected_response,
            completion=resolved_completion,
            usage=resolved_usage,
            stop_reason=stop_reason,
        )


def _finish(
    span: Any,
    response: Any,
    request_model: Any,
    started: float,
    completion: Optional[str] = None,
    usage: Optional[Mapping[str, Any]] = None,
    stop_reason: Any = _UNSET,
) -> None:
    response_model = _field(response, "model") or request_model
    if response_model:
        span.set_attribute("llm.model", response_model)
        span.set_attribute("gen_ai.response.model", response_model)
    response_id = _field(response, "id")
    if response_id:
        span.set_attribute("llm.response.id", response_id)
        span.set_attribute("gen_ai.response.id", response_id)
    stop = _field(response, "stop_reason") if stop_reason is _UNSET else stop_reason
    if stop is not None:
        span.set_attribute("llm.stop_reason", stop)
        span.set_attribute("gen_ai.response.finish_reasons", [str(stop)])
    content = completion if completion else _content_text(_field(response, "content"))
    if content:
        span.set_attribute("llm.completion", content[:_limit()])
        span.set_attribute("gen_ai.response.content", content[:_limit()])
    response_usage = dict(_usage(response))
    if usage:
        response_usage.update(usage)
    for key, value in response_usage.items():
        if not (_is_number(value) or isinstance(value, str)):
            continue
        span.set_attribute("llm.usage." + key, value)
        span.set_attribute("gen_ai.usage." + key, value)
    if response_usage.get("input_tokens") is not None:
        span.set_attribute("llm.usage.prompt_tokens", response_usage["input_tokens"])
        span.set_attribute("llm.usage.prompt_source", "provider_usage")
    if response_usage.get("output_tokens") is not None:
        span.set_attribute("llm.usage.completion_tokens", response_usage["output_tokens"])
        span.set_attribute("llm.usage.completion_source", "provider_usage")
    if response_usage:
        span.set_attribute("llm.usage.source", "provider_usage")
    _record_metrics(
        response_model,
        response_usage,
        time.perf_counter() - started,
        _compute_cost(
            response_model,
            response_usage.get("input_tokens"),
            response_usage.get("output_tokens"),
        ),
    )


class _StreamProxy:
    """Forward SDK stream protocols while reporting through shared stream state."""

    def __init__(
        self,
        stream: Any,
        span: Any,
        started: float,
        request_model: Any,
        span_exit: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._stream, self._span, self._started = stream, span, started
        self._request_model, self._done = request_model, False
        self._span_exit = span_exit
        self._entered_stream: Any = None
        self._final_snapshot: Any = None
        self._trace_state = _StreamTraceState()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        value = getattr(self._stream, name)
        if name == "text_stream":
            return self._wrap_text_stream(value)
        if name in {"get_final_message", "get_final_text", "until_done"}:
            return self._wrap_terminal_method(name, value)
        return value

    def __del__(self) -> None:
        try:
            if not self._done:
                self._finish()
        except Exception:
            pass

    def _snapshot(self) -> Any:
        if self._final_snapshot is not None:
            return self._final_snapshot
        try:
            return getattr(self._stream, "current_message_snapshot", None)
        except Exception:
            return None

    def _wrap_terminal_method(self, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                result = method(*args, **kwargs)
            except BaseException as exc:
                self._finish(exc)
                raise
            if inspect.isawaitable(result):
                async def await_result() -> Any:
                    try:
                        resolved = await result
                    except BaseException as exc:
                        self._finish(exc)
                        raise
                    self._finish_result(name, resolved)
                    return resolved

                return await_result()
            self._finish_result(name, result)
            return result

        return wrapped

    def _finish_result(self, name: str, result: Any) -> None:
        if name == "get_final_message":
            self._final_snapshot = result
            self._finish(response=result)
        elif name == "get_final_text":
            self._finish(response=self._snapshot(), completion=result)
        else:
            self._finish(response=self._snapshot())

    def _wrap_text_stream(self, stream: Any) -> Any:
        if hasattr(stream, "__aiter__") and not hasattr(stream, "__next__"):
            async def async_text_stream() -> AsyncIterator[Any]:
                try:
                    async for item in stream:
                        self._trace_state.observe_text(item)
                        yield item
                except BaseException as exc:
                    self._finish(exc)
                    raise
                else:
                    self._final_snapshot = self._snapshot()
                    self._finish(response=self._final_snapshot)

            return async_text_stream()

        def text_stream() -> Iterator[Any]:
            try:
                for item in stream:
                    self._trace_state.observe_text(item)
                    yield item
            except BaseException as exc:
                self._finish(exc)
                raise
            else:
                self._final_snapshot = self._snapshot()
                self._finish(response=self._final_snapshot)

        return text_stream()

    def _event(self, event: Any) -> None:
        self._trace_state.observe_event(event)
        if _field(event, "type") == "message_stop":
            self._finish(response=self._snapshot())

    def _finish(
        self,
        exc: Optional[BaseException] = None,
        response: Any = _UNSET,
        completion: Any = _UNSET,
    ) -> None:
        if self._done:
            return
        self._done = True
        result = self._trace_state.resolve(
            response=response,
            snapshot=self._snapshot(),
            completion=completion,
        )
        _close_stream_span(
            self._span,
            self._started,
            self._request_model,
            self._span_exit,
            exc,
            result.response,
            completion=result.completion,
            usage=result.usage,
            stop_reason=result.stop_reason,
        )

    def __iter__(self) -> Iterator[Any]:
        try:
            for event in self._stream:
                self._event(event)
                yield event
        except BaseException as exc:
            self._finish(exc)
            raise
        else:
            self._finish()

    def __next__(self) -> Any:
        try:
            event = next(self._stream)
            self._event(event)
            return event
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise

    def close(self) -> Any:
        try:
            result = self._stream.close()
        except BaseException as exc:
            self._finish(exc)
            raise

        if inspect.isawaitable(result):
            async def await_close() -> Any:
                try:
                    resolved = await result
                except BaseException as exc:
                    self._finish(exc)
                    raise
                self._finish()
                return resolved

            return await_close()

        self._finish()
        return result

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for event in self._stream:
                self._event(event)
                yield event
        except BaseException as exc:
            self._finish(exc)
            raise
        else:
            self._finish()

    async def __anext__(self) -> Any:
        try:
            event = await self._stream.__anext__()
            self._event(event)
            return event
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise

    def __enter__(self) -> "_StreamProxy":
        try:
            entered = self._stream.__enter__()
            self._entered_stream = self._stream
            if entered is not None:
                self._stream = entered
        except BaseException as exc:
            self._finish(exc)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> Any:
        context_manager = self._entered_stream if self._entered_stream is not None else self._stream
        try:
            return context_manager.__exit__(exc_type, exc, tb)
        except BaseException as exit_exc:
            self._finish(exit_exc)
            raise
        finally:
            self._finish(exc)

    async def __aenter__(self) -> "_StreamProxy":
        try:
            entered = await self._stream.__aenter__()
            self._entered_stream = self._stream
            if entered is not None:
                self._stream = entered
        except BaseException as exc:
            self._finish(exc)
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> Any:
        context_manager = self._entered_stream if self._entered_stream is not None else self._stream
        try:
            return await context_manager.__aexit__(exc_type, exc, tb)
        except BaseException as exit_exc:
            self._finish(exit_exc)
            raise
        finally:
            self._finish(exc)


def _make_stream(
    stream: Any,
    model: Any,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    span_state: Optional[Tuple[Any, float, Optional[Callable[..., Any]]]] = None,
) -> Any:
    if span_state is None:
        span_state = _open_stream_span(model, args, kwargs or {})
    span, started, span_exit = span_state
    if span is None:
        return stream
    return _StreamProxy(stream, span, started, model, span_exit=span_exit)


def _open_stream_span(
    model: Any,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Tuple[Any, float, Optional[Callable[..., Any]]]:
    started = time.perf_counter()
    try:
        tracer = _get_tracer("anthropic")
        attrs = _start_attrs(model, args, kwargs)
        if hasattr(tracer, "start_span"):
            span = tracer.start_span("llm.anthropic.messages", attributes=attrs)
            span.__enter__()
            return span, started, None
        context_manager = tracer.start_as_current_span("llm.anthropic.messages", attributes=attrs)
        span = context_manager.__enter__()
        return span, started, context_manager.__exit__
    except Exception:
        return None, started, None


def _close_stream_span(
    span: Any,
    started: float,
    model: Any,
    span_exit: Optional[Callable[..., Any]],
    exc: Optional[BaseException],
    response: Any,
    completion: Optional[str] = None,
    usage: Optional[Mapping[str, Any]] = None,
    stop_reason: Any = _UNSET,
) -> None:
    try:
        if exc is not None:
            span.record_exception(exc)
            span.set_status(SpanStatus.ERROR, str(exc))
            _record_metrics(model, {}, time.perf_counter() - started, error=True)
        else:
            _finish(
                span,
                response,
                model,
                started,
                completion=completion,
                usage=usage,
                stop_reason=stop_reason,
            )
    except Exception as instrumentation_error:
        try:
            span.record_exception(instrumentation_error)
            span.set_status(SpanStatus.ERROR, str(instrumentation_error))
        except Exception:
            pass
    finally:
        try:
            if span_exit:
                span_exit(
                    None if exc is None else type(exc),
                    exc,
                    getattr(exc, "__traceback__", None),
                )
            else:
                span.__exit__(
                    None if exc is None else type(exc),
                    exc,
                    getattr(exc, "__traceback__", None),
                )
        except Exception:
            try:
                span.end()
            except Exception:
                pass


def _record_call_error(span: Any, exc: BaseException, model: Any, started: float) -> None:
    span.record_exception(exc)
    span.set_status(SpanStatus.ERROR, str(exc))
    _record_metrics(model, {}, time.perf_counter() - started, error=True)


def _safe_finish(
    span: Any,
    response: Any,
    request_model: Any,
    started: float,
    **kwargs: Any,
) -> None:
    """Run _finish but never let response-attribute capture break the caller."""
    try:
        _finish(span, response, request_model, started, **kwargs)
    except Exception:
        pass


def _abort_stream_span(
    span_state: Tuple[Any, float, Optional[Callable[..., Any]]],
    model: Any,
    exc: BaseException,
) -> None:
    span, started, span_exit = span_state
    if span is not None:
        _close_stream_span(span, started, model, span_exit, exc, None)


@contextmanager
def _trace_call(
    model: Any,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Iterator[Tuple[Any, float]]:
    started = time.perf_counter()
    span_cm: Any = None
    span: Any = None
    try:
        tracer = _get_tracer("anthropic")
        span_cm = tracer.start_as_current_span(
            "llm.anthropic.messages", attributes=_start_attrs(model, args, kwargs)
        )
        span = span_cm.__enter__()
    except Exception:
        # Tracing setup failed - run the call untraced rather than raising.
        span_cm, span = None, None
    exc_info: Tuple[Any, Any, Any] = (None, None, None)
    try:
        yield span, started
    except BaseException as exc:
        exc_info = (type(exc), exc, getattr(exc, "__traceback__", None))
        if span is not None:
            try:
                _record_call_error(span, exc, model, started)
            except Exception:
                pass
        raise
    finally:
        if span_cm is not None:
            try:
                span_cm.__exit__(*exc_info)
            except Exception:
                try:
                    span.end()
                except Exception:
                    pass


def _call(
    fn: Callable[..., Any], self: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    model = _request_model(args, kwargs)
    with _trace_call(model, args, kwargs) as (span, started):
        response = fn(self, *args, **kwargs)
        if span is not None:
            _safe_finish(span, response, model, started)
        return response


def _call_stream(
    fn: Callable[..., Any], self: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    model = _request_model(args, kwargs)
    span_state = _open_stream_span(model, args, kwargs)
    try:
        stream = fn(self, *args, **kwargs)
    except BaseException as exc:
        _abort_stream_span(span_state, model, exc)
        raise
    return _make_stream(stream, model, args, kwargs, span_state)


async def _call_async(
    fn: Callable[..., Any], self: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    model = _request_model(args, kwargs)
    with _trace_call(model, args, kwargs) as (span, started):
        response = await fn(self, *args, **kwargs)
        if span is not None:
            _safe_finish(span, response, model, started)
        return response


async def _call_stream_async(
    fn: Callable[..., Any], self: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    model = _request_model(args, kwargs)
    span_state = _open_stream_span(model, args, kwargs)
    try:
        stream = await fn(self, *args, **kwargs)
    except BaseException as exc:
        _abort_stream_span(span_state, model, exc)
        raise
    return _make_stream(stream, model, args, kwargs, span_state)


def _wrap_create(fn: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    if getattr(fn, "_agent_trace_patched", False):
        return fn
    if is_async:
        @functools.wraps(fn)
        async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return await _call_stream_async(fn, self, args, kwargs)
            return await _call_async(fn, self, args, kwargs)
    else:
        @functools.wraps(fn)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return _call_stream(fn, self, args, kwargs)
            return _call(fn, self, args, kwargs)
    wrapped._agent_trace_patched = True
    return wrapped


def _wrap_stream(fn: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(fn, "_agent_trace_patched", False):
        return fn
    @functools.wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = _request_model(args, kwargs)
        span_state = _open_stream_span(model, args, kwargs)
        try:
            result = fn(self, *args, **kwargs)
        except BaseException as exc:
            _abort_stream_span(span_state, model, exc)
            raise
        if inspect.isawaitable(result):
            async def await_stream() -> Any:
                try:
                    stream = await result
                except BaseException as exc:
                    _abort_stream_span(span_state, model, exc)
                    raise
                return _make_stream(stream, model, args, kwargs, span_state)
            return await_stream()
        return _make_stream(result, model, args, kwargs, span_state)
    wrapped._agent_trace_patched = True
    return wrapped


def _patch_resource_class(resource_cls: Any, is_async: bool) -> bool:
    changed, create = False, getattr(resource_cls, "create", None)
    if create is not None and not getattr(create, "_agent_trace_patched", False):
        resource_cls.create = _wrap_create(create, is_async); changed = True
    stream = getattr(resource_cls, "stream", None)
    if stream is not None and not getattr(stream, "_agent_trace_patched", False):
        resource_cls.stream = _wrap_stream(stream); changed = True
    return changed or create is not None or stream is not None


def patch_anthropic() -> bool:
    """Patch Anthropic message resources for sync, async, and streaming calls."""
    global _patched
    if _patched:
        return True
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    patched = False
    for module_name in (
        "anthropic.resources.messages.messages",
        "anthropic.resources.beta.messages.messages",
    ):
        try:
            module = importlib.import_module(module_name)
            for class_name, is_async in (("Messages", False), ("AsyncMessages", True)):
                resource_cls = getattr(module, class_name, None)
                if resource_cls is not None:
                    patched = _patch_resource_class(resource_cls, is_async) or patched
        except (ImportError, AttributeError):
            continue
    _patched = patched
    return patched


def _get_tracer(name: str) -> Any:
    import traccia
    return traccia.get_tracer(name)
