"""@observe decorator for instrumenting functions."""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import inspect
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional
from traccia.tracer.span import SpanStatus

# traccia.governance is imported inside the functions that use it: importing it here would run
# traccia.governance.govern, which imports observe from this module while it is still loading.
if TYPE_CHECKING:
    from traccia.governance.idempotency import IdempotencySpec


def _first_self_or_cls(func: Callable[..., Any]) -> Optional[str]:
    """``self`` or ``cls`` when it is the function's first positional parameter, else None."""
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return None
    if params and params[0].name in ("self", "cls") and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return params[0].name
    return None


def _receiver_name(func: Callable[..., Any]) -> Optional[str]:
    """``self`` or ``cls`` when it is the first parameter of a method (a function defined in a
    class body), else None: a plain function keeps such a parameter as data, unless called
    through a class it was attached to later (see _is_attached_receiver())."""
    parts = (getattr(func, "__qualname__", "") or "").split(".")
    if len(parts) < 2 or parts[-2] == "<locals>":
        return None
    return _first_self_or_cls(func)


def _is_attached_receiver(value: Any, name: str, wrapper: Any) -> bool:
    """Whether ``value``, passed as ``self`` or ``cls``, is the instance or class this tool was
    called through: the tool (or a classmethod or staticmethod around it) is an attribute of its
    class. Covers a function attached to a class afterwards (``Payments.refund =
    observe(...)(_refund)``, a mixin), whose qualified name has no class."""
    owner = value if name == "cls" else type(value)
    if not isinstance(owner, type):
        return False
    try:
        if name == "self":
            # Bound to this one instance (``obj.refund = types.MethodType(tool, obj)``).
            for attr in vars(value).values():
                if getattr(attr, "__self__", None) is value and _leads_to(attr, wrapper):
                    return True
    except Exception:  # noqa: BLE001 - no __dict__, or an odd instance
        pass
    try:
        classes = owner.__mro__
    except Exception:  # noqa: BLE001 - an odd class: keep the value as data
        return False
    for c in classes:
        try:
            attrs = list(vars(c).values())
        except Exception:  # noqa: BLE001
            continue
        for attr in attrs:
            if _leads_to(attr, wrapper):
                return True
    return False


def _leads_to(attr: Any, wrapper: Any, depth: int = 8) -> bool:
    """Whether ``attr`` is ``wrapper``, wraps it as a method (``__func__``), or reaches it through a
    ``functools.wraps`` chain (``__wrapped__``), as a tool attached under another decorator is."""
    for _ in range(depth):
        if attr is None:
            return False
        if attr is wrapper:
            return True
        try:
            func = getattr(attr, "__func__", None)
            if func is wrapper:
                return True
            nxt = getattr(attr, "__wrapped__", None)
            if nxt is None and func is not None:
                nxt = getattr(func, "__wrapped__", None)
        except Exception:  # noqa: BLE001 - a getattr that raises
            return False
        attr = nxt
    return False


def _tool_arguments(bound: inspect.BoundArguments, receiver: Optional[str]) -> Dict[str, Any]:
    """A tool's arguments as policy checks, read evidence and idempotency keys see them:
    all of them except a leading ``self``/``cls``."""
    return {k: v for k, v in bound.arguments.items() if k != receiver}


def _build_idempotency_spec(
    func: Callable[..., Any],
    *,
    name: Optional[str],
    as_type: str,
    idempotent: bool,
    key_fields: Optional[Iterable[str]],
    unconfirmed_on: Optional[Iterable[type]],
    release_on: Optional[Iterable[type]],
    release_unconfirmed: bool,
    idempotency_scope: Optional[str],
    key_extra: Optional[Callable[..., Any]],
    in_flight_wait_s: Optional[float],
    key_types: Optional[Dict[str, str]],
) -> Optional[IdempotencySpec]:
    """Validate the idempotency options once, when the tool is decorated, and build the tool's
    spec; None for a tool that is not idempotent. The signature is read once: a callable that has
    none skips the checks that need it.

    Type hints are not read for ``key_types``: Node has no runtime types, so keys would differ.
    """
    from traccia.governance.idempotency import KEY_TYPES, IdempotencySpec, coerce_key_types, fixed_default_forms

    if key_fields is None:
        fields = None
    elif isinstance(key_fields, str):
        fields = (key_fields,)
    else:
        fields = tuple(key_fields)
    if fields is not None and not all(isinstance(f, str) for f in fields):
        raise TypeError("key_fields must be parameter names (strings)")
    if fields is not None and not idempotent:
        raise ValueError("key_fields requires idempotent=True")
    params = None
    var_keyword = None
    if idempotent:
        if as_type != "tool":
            raise ValueError("idempotent=True requires as_type='tool'")
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            raise TypeError("idempotent tools cannot be generators in v1")
        if fields is not None and not fields:
            raise ValueError("key_fields must name at least one parameter")
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            params = None
        if params is not None:
            var_keyword = next((p.name for p in params.values() if p.kind is inspect.Parameter.VAR_KEYWORD), None)
            if fields is not None and var_keyword is None:
                for field in fields:
                    n = field[:-1] if field.endswith("?") else field  # "cc?" marks an optional field
                    if n not in params:
                        raise ValueError(f"key_fields: unknown parameter '{n}'")

    unconfirmed_types = tuple(unconfirmed_on or ())
    release_types = tuple(release_on or ())
    if (unconfirmed_on is not None or release_on is not None or release_unconfirmed) and not idempotent:
        raise ValueError("unconfirmed_on, release_on and release_unconfirmed require idempotent=True")
    if any(not (isinstance(t, type) and issubclass(t, BaseException)) for t in unconfirmed_types):
        raise TypeError("unconfirmed_on must list exception classes")
    if any(not (isinstance(t, type) and issubclass(t, BaseException)) for t in release_types):
        raise TypeError("release_on must list exception classes")
    if unconfirmed_types and release_unconfirmed:
        raise ValueError("pass unconfirmed_on or release_unconfirmed=True, not both")

    scope = "agent"
    if idempotency_scope is not None:
        if not idempotent:
            raise ValueError("idempotency_scope requires idempotent=True")
        if idempotency_scope not in ("agent", "workspace"):
            raise ValueError("idempotency_scope must be 'agent' or 'workspace'")
        scope = idempotency_scope

    if key_extra is not None and not idempotent:
        raise ValueError("key_extra requires idempotent=True")
    if key_extra is not None and not callable(key_extra):
        raise TypeError("key_extra must be callable: it gets the tool's arguments and returns extra key values")
    if key_extra is not None and (
        inspect.iscoroutinefunction(key_extra)
        or inspect.iscoroutinefunction(getattr(key_extra, "__call__", None))
    ):
        raise TypeError("key_extra must return its values directly, not a coroutine: an async key_extra cannot key the call")

    if in_flight_wait_s is not None and not idempotent:
        raise ValueError("in_flight_wait_s requires idempotent=True")
    if in_flight_wait_s is not None and not in_flight_wait_s >= 0:
        raise ValueError("in_flight_wait_s must be 0 or more seconds")

    if idempotent and not name:
        # The tool name is part of the key. A function's own name is not stable enough:
        # two modules can each have a refund(), and renaming or moving one changes its keys.
        raise ValueError("idempotent tools need a name: pass @observe(name='...')")

    if key_types is not None and not idempotent:
        raise ValueError("key_types requires idempotent=True")
    if not idempotent:
        return None
    resolved_types = {}
    for field, kind in (key_types or {}).items():
        if kind not in KEY_TYPES:
            raise ValueError(f"key_types: '{field}' must be 'string', 'number' or 'boolean'")
        if params and var_keyword is None and field not in params:
            raise ValueError(f"key_types: unknown parameter '{field}'")
        resolved_types[field] = kind

    defaults = {
        p.name: p.default
        for p in (params.values() if params is not None else ())
        if p.default is not inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    # Converted too: a default that key_types turns into text (version=1 under "string") must
    # still equal itself, or it would stay in the key.
    defaults = coerce_key_types(defaults, resolved_types)
    wait = {} if in_flight_wait_s is None else {"in_flight_wait_s": float(in_flight_wait_s)}
    return IdempotencySpec(
        name=name,
        key_fields=fields,
        var_keyword=var_keyword,
        scope=scope,
        key_types=resolved_types,
        defaults=defaults,
        default_forms=fixed_default_forms(defaults),
        unconfirmed_types=unconfirmed_types,
        release_types=release_types,
        release_unconfirmed=release_unconfirmed,
        key_extra=key_extra,
        **wait,
    )


def _note_retrieval(span: Any) -> None:
    try:
        from traccia.governance.pep import note_retrieval_attributes

        attributes = getattr(span, "attributes", None) or {}
        context = getattr(span, "context", None)
        trace_id = getattr(context, "trace_id", None) if context else None
        if isinstance(trace_id, int):
            trace_id = format(trace_id, "032x")
        note_retrieval_attributes(str(trace_id) if trace_id else None, dict(attributes))
    except Exception:
        return


def _capture_args(bound_args: inspect.BoundArguments, skip: Iterable[str]) -> Dict[str, Any]:
    """Capture function arguments, converting complex types to OTel-compatible types."""
    captured = {}
    for name, value in bound_args.arguments.items():
        if name in skip:
            continue
        # Skip 'self' - it's an object, not a valid OTel attribute
        if name == "self":
            continue
        captured[name] = _convert_to_otel_type(value)
    return captured


def _convert_to_otel_type(value: Any) -> Any:
    """``value`` as an OTel attribute value (bool, str, bytes, int, float, or a list of those), truncated."""
    if isinstance(value, (bool, str, bytes, int, float)) or value is None:
        return value
    
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            if isinstance(item, (bool, str, bytes, int, float)) or item is None:
                converted.append(item)
            else:
                converted.append(str(item)[:1000])
        return converted[:100]
    
    if isinstance(value, dict):
        try:
            import json
            json_str = json.dumps(value, default=str)[:1000]
            return json_str
        except Exception:
            return str(value)[:1000]
    
    return str(value)[:1000]


def _infer_type_from_attributes(attributes: Dict[str, Any]) -> Optional[str]:
    """"llm" or "tool" when the attributes name one, else None (the span stays "span")."""
    if any(key in attributes for key in ["llm.model", "llm.vendor", "model"]):
        return "llm"
    
    if any(key in attributes for key in ["tool.name", "tool"]):
        return "tool"
    
    return None


def _extract_llm_attributes(span_attrs: Dict[str, Any], bound_args: inspect.BoundArguments) -> None:
    """Copy model, temperature, max_tokens and messages/prompt arguments into ``span_attrs``; never raises."""
    try:
        args_dict = dict(bound_args.arguments)
        
        if "model" in args_dict and "llm.model" not in span_attrs:
            span_attrs["llm.model"] = str(args_dict["model"])
        
        if "temperature" in args_dict and "llm.temperature" not in span_attrs:
            temp = args_dict["temperature"]
            if isinstance(temp, (int, float)):
                span_attrs["llm.temperature"] = temp
        
        if "max_tokens" in args_dict and "llm.max_tokens" not in span_attrs:
            max_tok = args_dict["max_tokens"]
            if isinstance(max_tok, int):
                span_attrs["llm.max_tokens"] = max_tok
        
        if "messages" in args_dict and "llm.prompt" not in span_attrs:
            messages = args_dict["messages"]
            if isinstance(messages, (list, str)):
                prompt_str = _convert_to_otel_type(messages)
                span_attrs["llm.prompt"] = prompt_str
        elif "prompt" in args_dict and "llm.prompt" not in span_attrs:
            prompt = args_dict["prompt"]
            if isinstance(prompt, str):
                span_attrs["llm.prompt"] = prompt[:1000]
    
    except Exception:
        # Extraction must never interrupt span creation.
        pass


def _validate_guardrail_span(span_attrs: Dict[str, Any]) -> None:
    """Log warnings for guardrail-typed spans missing recommended attributes."""
    try:
        from traccia.guardrails.helpers import validate_guardrail_attributes
        import logging
        # triggered may be set after the function returns (bool auto-trigger)
        warnings = validate_guardrail_attributes(span_attrs, require_triggered=False)
        if warnings:
            _logger = logging.getLogger("traccia.guardrails")
            for w in warnings:
                _logger.warning(w)
    except Exception:
        pass


@contextlib.asynccontextmanager
async def _async_null():
    """contextlib.nullcontext() for ``async with`` (which Python 3.9 lacks)."""
    yield


def observe(
    name: Optional[str] = None,
    *,
    attributes: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
    as_type: str = "span",
    skip_args: Optional[Iterable[str]] = None,
    skip_result: bool = False,
    idempotent: bool = False,
    key_fields: Optional[Iterable[str]] = None,
    unconfirmed_on: Optional[Iterable[type]] = None,
    release_on: Optional[Iterable[type]] = None,
    release_unconfirmed: bool = False,
    idempotency_scope: Optional[str] = None,
    key_extra: Optional[Callable[..., Any]] = None,
    in_flight_wait_s: Optional[float] = None,
    key_types: Optional[Dict[str, str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate a function to create a span around its execution.

    - Supports sync and async functions.
    - Captures errors and records exception events.
    - Optionally captures arguments/results (skip controls).
    - ``idempotent=True`` (tools only): the call gets an idempotency key, readable inside
      the tool with ``traccia.idempotency_key()``, built from all arguments or only
      ``key_fields``. The key is committed when the tool returns. When it raises, the key is
      held as unconfirmed for the window, since the work may have happened, unless the tool
      says the failure is definite: it raised ``traccia.DefiniteFailure`` or an error of a class
      in ``release_on``. That releases the key so a retry runs. ``unconfirmed_on`` lists classes
      that hold the call wherever they appear in the chain, even below such an error.
      ``release_unconfirmed=True`` lets a retry run after an unclear outcome, with the same
      downstream key, for a tool whose downstream API dedupes retries by the key.
      ``idempotency_scope="workspace"`` gives every agent in the workspace the same key for
      the same call, so one agent repeating another's action is caught (default: per agent).
      ``key_extra`` is called with the tool's own arguments (including ``self``) and returns
      non-argument values added to the key (the account on ``self``, the tenant); None adds nothing.
      ``in_flight_wait_s`` (default 30): how long a call denied because the same call is still
      running waits and checks again, so it runs if that call fails; 0 raises at once.
      ``key_types`` ({name: "string" | "number" | "boolean"}) converts those arguments before
      keying, so 123 and "123" key the same. Type hints are not read.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or func.__name__
        skip_args_set = set(skip_args or [])
        tags_list = [str(tag) for tag in tags] if tags is not None else []
        spec = _build_idempotency_spec(
            func,
            name=name,
            as_type=as_type,
            idempotent=idempotent,
            key_fields=key_fields,
            unconfirmed_on=unconfirmed_on,
            release_on=release_on,
            release_unconfirmed=release_unconfirmed,
            idempotency_scope=idempotency_scope,
            key_extra=key_extra,
            in_flight_wait_s=in_flight_wait_s,
            key_types=key_types,
        )
        static_receiver = _receiver_name(func)
        late_receiver = _first_self_or_cls(func) if static_receiver is None else None

        def receiver_of(bound: inspect.BoundArguments) -> Optional[str]:
            if late_receiver is None or late_receiver not in bound.arguments:
                return static_receiver
            # ``wrapper``: the function this decorator returns, assigned below before any call.
            attached = _is_attached_receiver(bound.arguments[late_receiver], late_receiver, wrapper)
            return late_receiver if attached else None

        def _tool_call(bound: inspect.BoundArguments, arguments: Dict[str, Any], is_async: bool = False) -> Any:
            from traccia.governance.idempotency import Unkeyable, coerce_key_types, drop_defaults
            from traccia.governance.pep import async_governed_tool_call, governed_tool_call

            key_arguments = None
            if spec is not None:
                # Keys see **kwargs entries as plain arguments, so key_fields can name them.
                var_keyword = spec.var_keyword
                merged = {k: v for k, v in arguments.items() if k != var_keyword}
                extra_kwargs = (arguments.get(var_keyword) or {}) if var_keyword is not None else {}
                clash = sorted(k for k in extra_kwargs if k in merged)
                if clash:
                    # A positional-only or *args parameter and a **kwargs entry share a name; one
                    # would overwrite the other, so two different calls could share a key.
                    key_arguments = Unkeyable(
                        f"**{var_keyword} entry '{clash[0]}' has the same name as a parameter")
                else:
                    merged.update(extra_kwargs)
                    # Declared types first ("1" for an int becomes 1), then arguments left at
                    # their default are left out (Node has no defaults).
                    key_arguments = drop_defaults(
                        coerce_key_types(merged, spec.key_types), spec.defaults, spec.default_forms)
            governed = async_governed_tool_call if is_async else governed_tool_call
            return governed(span_name, arguments, spec, key_arguments=key_arguments, bound_args=bound)

        is_coro = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = _get_tracer(func.__module__ or "default")
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()

            span_attrs = dict(attributes or {})
            if tags_list:
                span_attrs["span.tags"] = tags_list
            
            span_attrs.update(_capture_args(bound, skip_args_set))
            
            inferred_type = as_type
            if as_type == "span":
                detected_type = _infer_type_from_attributes(span_attrs)
                if detected_type:
                    inferred_type = detected_type
            
            span_attrs["span.type"] = inferred_type
            
            if inferred_type == "llm":
                _extract_llm_attributes(span_attrs, bound)

            if inferred_type == "guardrail":
                _validate_guardrail_span(span_attrs)

            with tracer.start_as_current_span(span_name, attributes=span_attrs) as span:
                try:
                    # The tool's arguments, read once per call: the class a method is called
                    # through can gain the tool later (see receiver_of()).
                    arguments = _tool_arguments(bound, receiver_of(bound)) if inferred_type == "tool" else None
                    tool_cm = _tool_call(bound, arguments) if inferred_type == "tool" else contextlib.nullcontext()
                    with tool_cm as call:
                        result = func(*args, **kwargs)
                        if call is not None and (inspect.isawaitable(result) or isinstance(result, concurrent.futures.Future)):
                            # A plain def that returned a coroutine or future: settle the
                            # idempotency key when the work finishes.
                            pending = call.defer(result)
                            if pending is not result:
                                # Its result is not known yet: nothing to remember or stamp.
                                return pending
                    if inferred_type == "tool":
                        from traccia.governance.pep import remember_tool_result

                        remember_tool_result(span_name, result, arguments)
                    _note_retrieval(span)
                    # A guardrail's bool result sets guardrail.triggered unless attributes= set it.
                    if inferred_type == "guardrail" and isinstance(result, bool):
                        if span.attributes.get("guardrail.triggered") is None:
                            span.set_attribute("guardrail.triggered", result)
                    if not skip_result:
                        otel_result = _convert_to_otel_type(result)
                        span.set_attribute("result", otel_result)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(SpanStatus.ERROR, str(exc))
                    
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_attribute("error.message", str(exc))
                    
                    tb = traceback.format_exc()
                    span.set_attribute("error.stack_trace", tb[:2000])
                    
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = _get_tracer(func.__module__ or "default")
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()

            span_attrs = dict(attributes or {})
            if tags_list:
                span_attrs["span.tags"] = tags_list
            
            span_attrs.update(_capture_args(bound, skip_args_set))
            
            inferred_type = as_type
            if as_type == "span":
                detected_type = _infer_type_from_attributes(span_attrs)
                if detected_type:
                    inferred_type = detected_type
            
            span_attrs["span.type"] = inferred_type
            
            if inferred_type == "llm":
                _extract_llm_attributes(span_attrs, bound)

            if inferred_type == "guardrail":
                _validate_guardrail_span(span_attrs)

            async with tracer.start_as_current_span(span_name, attributes=span_attrs) as span:
                try:
                    arguments = _tool_arguments(bound, receiver_of(bound)) if inferred_type == "tool" else None
                    tool_cm = _tool_call(bound, arguments, is_async=True) if inferred_type == "tool" else _async_null()
                    async with tool_cm:
                        result = await func(*args, **kwargs)
                    if inferred_type == "tool":
                        from traccia.governance.pep import remember_tool_result

                        remember_tool_result(span_name, result, arguments)
                    _note_retrieval(span)
                    if inferred_type == "guardrail" and isinstance(result, bool):
                        if span.attributes.get("guardrail.triggered") is None:
                            span.set_attribute("guardrail.triggered", result)
                    if not skip_result:
                        otel_result = _convert_to_otel_type(result)
                        span.set_attribute("result", otel_result)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(SpanStatus.ERROR, str(exc))
                    
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_attribute("error.message", str(exc))
                    
                    tb = traceback.format_exc()
                    span.set_attribute("error.stack_trace", tb[:2000])
                    
                    raise

        wrapper = async_wrapper if is_coro else sync_wrapper
        return wrapper

    return decorator


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)

