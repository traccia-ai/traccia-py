"""Governed tool calls: the policy check, idempotency key and settle around one tool call."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import inspect
import logging
import random
import time
import weakref
from concurrent.futures import Future
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, NamedTuple, Optional, Tuple, Union

from traccia import runtime_config
from traccia.governance import pep
from traccia.governance.asyncio_internals import never_ran, peek_exception, unstarted_task_frame
from traccia.governance.idempotency import (
    KEY_PATTERN,
    KEY_PREFIX,
    OUTCOME_ABANDON,
    OUTCOME_COMMIT,
    OUTCOME_RELEASE,
    OUTCOME_UNCONFIRMED,
    IdempotencySpec,
    Outcome,
    Unkeyable,
    bound_key,
    derive_key,
    is_unconfirmed,
)
from traccia.governance.policy import IDEMPOTENCY_IN_FLIGHT, AgentBlockedError, IdempotencyRepeat
from traccia.governance.serialization import json_safe

logger = logging.getLogger("traccia.governance")


def enforce_tool_call(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    idempotency: Optional[Dict[str, str]] = None,
    idempotency_scope: str = "agent",
) -> Optional[Dict[str, Any]]:
    """Check a tool call. ``idempotency`` ({"key": ...} or {"unkeyable": reason}) marks it idempotent;
    ``idempotency_scope="workspace"`` shares its key across the workspace's agents."""
    if not runtime_config.pep_enabled():
        return None
    payload = arguments or {}
    context: Dict[str, Any] = {"input": json_safe(payload), "tool_name": name}
    freshness = pep._freshness_context(pep._resource_id(payload))
    if freshness:
        context["freshness"] = freshness
    customer_id = pep._customer_id(payload)
    if customer_id:
        context["customer_id"] = customer_id
    action: Dict[str, Any] = {"type": "tool_call", "name": name}
    if idempotency is not None:
        action["idempotent"] = True
        context["idempotency"] = idempotency
        if idempotency_scope == "workspace":
            action["idempotency_scope"] = "workspace"
    return pep.check_policy(action=action, context=context)


class CallKey(NamedTuple):
    """An idempotent call's key and its offline downstream key (see _offline_downstream_key());
    or, for a call that cannot be keyed, the Unkeyable that ``idempotency_key()`` raises inside
    it, whose message is the reason the check sends."""

    key: Optional[str] = None
    offline: Optional[str] = None
    unkeyable: Optional[Unkeyable] = None

    def wire(self) -> Dict[str, str]:
        """The idempotency field the check sends."""
        if self.key is None:
            return {"unkeyable": str(self.unkeyable)}
        wire = {"key": self.key}
        if self.offline != self.key:
            # A new attempt adopts it unless an earlier attempt's key was spent, so a call with
            # checks off and its checked retry pass the same key.
            wire["fallback_key"] = self.offline
        return wire


def _tool_key(
    spec: Optional[IdempotencySpec],
    arguments: Dict[str, Any],
    key_arguments: Optional[Union[Dict[str, Any], Unkeyable]] = None,
    bound_args: Optional[inspect.BoundArguments] = None,
) -> Tuple[Optional[CallKey], Optional[Dict[str, str]]]:
    """The call key and the idempotency field the check sends, both None for a non-idempotent
    tool. Cheap (no I/O)."""
    call_key = _call_key(spec, arguments, key_arguments, bound_args)
    return call_key, None if call_key is None else call_key.wire()


def _call_key(
    spec: Optional[IdempotencySpec],
    arguments: Dict[str, Any],
    key_arguments: Optional[Union[Dict[str, Any], Unkeyable]],
    bound_args: Optional[inspect.BoundArguments],
) -> Optional[CallKey]:
    """The key is derived from ``key_arguments`` (default: ``arguments``). A workspace-scoped key
    leaves the agent out, so every agent derives the same one. ``spec.key_extra`` is called with
    the tool's own arguments, ``bound_args``."""
    if spec is None:
        return None
    if isinstance(key_arguments, Unkeyable):
        # Checked before key_extra runs: the call is unkeyable whatever it returns.
        return CallKey(unkeyable=key_arguments)
    agent = None if spec.scope == "workspace" else pep._agent_id()
    extra = None
    if spec.key_extra is not None:
        try:
            extra = spec.key_extra(*bound_args.args, **bound_args.kwargs)
        except Exception as exc:
            # Its message may hold raw values (a tenant id, an account): send only the type.
            return CallKey(unkeyable=Unkeyable(f"{type(exc).__name__} while computing key_extra"))
    try:
        key = derive_key(
            agent, spec.name, arguments if key_arguments is None else key_arguments, spec.key_fields, extra
        )
    except Unkeyable as u:
        return CallKey(unkeyable=u)
    except Exception as exc:
        # e.g. a raising model_dump(). Its message may hold raw values: send only the type.
        return CallKey(unkeyable=Unkeyable(f"{type(exc).__name__} while keying arguments"))
    return CallKey(key, _offline_downstream_key(key))


# A call denied because the same call is in flight waits and checks again: if the first call
# fails, this one runs instead of being lost; if it succeeds, this one gets the normal repeat.
# Delays double from IN_FLIGHT_FIRST_DELAY_S up to IN_FLIGHT_MAX_DELAY_S.
IN_FLIGHT_FIRST_DELAY_S = 0.5
IN_FLIGHT_MAX_DELAY_S = 4.0
_monotonic = time.monotonic
_sleep = time.sleep
_async_sleep = asyncio.sleep


def _recheck_wire(wire: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The idempotency field for a check made while waiting out an in-flight call: flagged, so the
    server neither counts it as another tool call nor stores or alerts on it while the answer is
    still "in flight"."""
    return None if wire is None else {**wire, "recheck": True}


def _jittered(delay: float) -> float:
    """``delay`` scaled by a random 0.5-1.5: callers waiting out the same in-flight call otherwise
    re-check in lockstep (0.5, 1, 2, 4 s), hitting the server in waves during a burst."""
    return delay * random.uniform(0.5, 1.5)


def _in_flight_pauses(wait_s: float) -> Iterator[float]:
    """The pauses before each re-check of a call denied as in flight: doubling and jittered, and
    only those that end within ``wait_s`` from now. The caller sleeps each one and checks again."""
    deadline = _monotonic() + wait_s

    def pauses() -> Iterator[float]:
        delay = IN_FLIGHT_FIRST_DELAY_S
        while True:
            pause = _jittered(delay)
            if _monotonic() + pause > deadline:
                return
            yield pause
            delay = min(delay * 2, IN_FLIGHT_MAX_DELAY_S)

    return pauses()


def _next_pause(exc: IdempotencyRepeat, pauses: Iterator[float]) -> Optional[float]:
    """How long to wait before checking again after ``exc``, or None to raise it: only a call
    still in flight is waited out, and only within the wait."""
    return next(pauses, None) if exc.reason == IDEMPOTENCY_IN_FLIGHT else None


# Idempotent calls running in this context, so a tool that calls itself with the same key does not
# wait out its own in-flight reservation. Entries are marked done when the call ends, since a task
# or thread the tool spawned holds a copy of this context.
_active_calls: contextvars.ContextVar[tuple] = contextvars.ContextVar("traccia_active_idem_calls", default=())


class _ActiveCall:
    __slots__ = ("key", "running")

    def __init__(self, key: str) -> None:
        self.key = key
        self.running = True


def _reentrant(call_key: Optional[CallKey]) -> bool:
    """Whether this context is already running the same idempotent call (the tool called itself
    with the same arguments). Such a call is still checked, without waiting: a Block policy
    denies it as in flight at once, while with no Block policy (Observe, Warn, none) it runs,
    as the server says."""
    return (call_key is not None and call_key.key is not None and runtime_config.pep_enabled()
            and any(c.key == call_key.key and c.running for c in _active_calls.get()))


@contextmanager
def _active(call_key: Optional[CallKey]) -> Iterator[None]:
    if call_key is None or call_key.key is None:
        yield
        return
    call = _ActiveCall(call_key.key)
    token = _active_calls.set(_active_calls.get() + (call,))
    try:
        yield
    finally:
        call.running = False
        _active_calls.reset(token)


def _on_event_loop_thread() -> bool:
    """Whether an asyncio event loop is running on this thread (the check is per thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _check_tool(
    name: str, arguments: Dict[str, Any], wire: Optional[Dict[str, str]], spec: Optional[IdempotencySpec]
) -> Tuple[Optional[Dict[str, Any]], Optional[pep.SettleTarget]]:
    target = pep.settle_target()
    scope = spec.scope if spec is not None else "agent"
    return enforce_tool_call(name, arguments, idempotency=wire, idempotency_scope=scope), target


def _prepare(
    spec: Optional[IdempotencySpec],
    arguments: Dict[str, Any],
    key_arguments: Optional[Union[Dict[str, Any], Unkeyable]],
    bound_args: Optional[inspect.BoundArguments],
) -> Tuple[Optional[CallKey], Optional[Dict[str, str]], float]:
    """The call key, the idempotency field the check sends, and how long an in-flight denial is
    waited out: never for a call that would wait out its own reservation."""
    call_key, wire = _tool_key(spec, arguments, key_arguments, bound_args)
    if spec is None or _reentrant(call_key):
        return call_key, wire, 0.0
    return call_key, wire, spec.in_flight_wait_s


@contextmanager
def governed_tool_call(
    name: str,
    arguments: Dict[str, Any],
    spec: Optional[IdempotencySpec] = None,
    *,
    key_arguments: Optional[Union[Dict[str, Any], Unkeyable]] = None,
    bound_args: Optional[inspect.BoundArguments] = None,
) -> Iterator["ToolCallSettle"]:
    """Check a tool call, run the block, then settle what the check reserved.

    With a ``spec`` (an idempotent tool) the key is derived from ``key_arguments`` (default:
    ``arguments``) and the downstream key is bound for ``traccia.idempotency_key()``, even with
    policy checks off. Success commits the key; an exception settles it by
    ``_failure_outcome`` and is re-raised. The settle finishes before the call returns or raises.

    Yields a ToolCallSettle: a sync tool that returned an awaitable passes it to ``defer()``
    and returns what that gives back, so the key settles when the work finishes.
    """
    call_key, wire, wait_s = _prepare(spec, arguments, key_arguments, bound_args)
    if wait_s > 0 and _on_event_loop_thread():
        # A sync tool called from async code: sleeping would freeze the event loop, so an
        # in-flight denial raises at once.
        wait_s = 0.0
    if runtime_config.pep_enabled():
        pauses = _in_flight_pauses(wait_s)
        wire_now = wire
        while True:
            try:
                decision, target = _check_tool(name, arguments, wire_now, spec)
                break
            except IdempotencyRepeat as exc:
                pause = _next_pause(exc, pauses)
                if pause is None:
                    raise
            _sleep(pause)
            wire_now = _recheck_wire(wire)
    else:
        decision, target = None, None
    downstream_key = _resolve_downstream_key(call_key, decision)
    call = ToolCallSettle(decision, target, downstream_key, spec)
    try:
        with _active(call_key), bound_key(downstream_key):
            yield call
    except BaseException as exc:
        _settle_quietly(decision, target, _settle_outcome(exc, spec))
        raise
    if not call._deferred:
        _settle_quietly(decision, target, OUTCOME_COMMIT)


class ToolCallSettle:
    """The settle of one governed sync tool call, which can wait for an awaitable it returned."""

    def __init__(
        self,
        decision: Optional[Dict[str, Any]],
        target: Optional[pep.SettleTarget],
        downstream_key: Optional[Union[str, Unkeyable]],
        spec: Optional[IdempotencySpec],
    ) -> None:
        self._decision = decision
        self._target = target
        self._downstream_key = downstream_key
        self._spec = spec
        self._deferred = False  # settled when the awaitable the tool returned finishes
        self._started = False  # the coroutine defer() returned has started
        self._awaitable: Any = None  # the awaitable that coroutine awaits
        self._unstarted_frame: Any = None  # a returned task's coroutine frame, before its first step
        self._pending_finalizer: Optional[weakref.finalize] = None  # abandons a returned future dropped unfinished

    def defer(self, awaitable: Any) -> Any:
        """Settle when ``awaitable`` finishes instead of now; returns what to hand back (a task or
        future as it is, since it runs by itself; anything else wrapped in a coroutine).

        Never raises: the work may already be running, so on an internal error the call settles
        as unconfirmed and the awaitable is handed back as it is.
        """
        try:
            return self._defer(awaitable)
        except Exception:  # noqa: BLE001 - bookkeeping must never decide the call's outcome
            logger.warning("traccia: could not track the awaitable an idempotent tool returned; "
                           "its key is held as unconfirmed", exc_info=True)
            self._deferred = True
            _settle_tool_in_background(self._decision, self._target, OUTCOME_UNCONFIRMED)
            return awaitable

    def _defer(self, awaitable: Any) -> Any:
        """The body of :meth:`defer`.

        Without it, a plain ``def`` tool that returns a coroutine or future would commit its key
        before the work started. A coroutine dropped before its first step abandons the
        reservation when collected, so a retry does not wait out the lease.
        """
        if self._downstream_key is None:
            return awaitable  # not idempotent: nothing waits on the work
        self._deferred = True
        if isinstance(awaitable, Future):
            # A concurrent.futures.Future (executor.submit, run_coroutine_threadsafe): the work runs
            # elsewhere, so settle when it ends, not when the tool returned it.
            awaitable.add_done_callback(self._settle_concurrent_done)
            return awaitable
        if isinstance(awaitable, asyncio.Future):
            # A task or future runs by itself, awaited or not: settle when it ends and hand it back
            # unwrapped, as a wrapping coroutine could be dropped while the work goes on.
            self._unstarted_frame = unstarted_task_frame(awaitable)
            awaitable.add_done_callback(self._settle_when_done)
            if not isinstance(awaitable, asyncio.Task):
                # A plain future dropped uncompleted never runs its done callback: abandon when it
                # is collected. Not for a task, which may have started its work.
                pep.start_handoff_worker()
                self._pending_finalizer = weakref.finalize(awaitable, pep.hand_off, self._abandon_never_completed)
            return awaitable
        self._awaitable = awaitable
        finishing = self._finish(awaitable)
        # The finalizer may run while this thread holds a lock that settling takes, so it only
        # hands the work off; the worker is started here, in ordinary code.
        pep.start_handoff_worker()
        weakref.finalize(finishing, pep.hand_off, self._abandon_if_never_started)
        return finishing

    def _settle(self, exc: Optional[BaseException]) -> None:
        _settle_tool_in_background(self._decision, self._target, _settle_outcome(exc, self._spec))

    def _settle_concurrent_done(self, future: Any) -> None:
        self._on_done(future, asyncio_future=False)

    def _abandon_never_completed(self) -> None:
        _abandon(self._decision, self._target)

    def _settle_when_done(self, future: Any) -> None:
        if self._pending_finalizer is not None:
            self._pending_finalizer.detach()  # it completed: settled here, never abandoned
            self._pending_finalizer = None
        self._on_done(future, asyncio_future=True)

    def _on_done(self, future: Any, asyncio_future: bool) -> None:
        """Settle by how a returned future ended: a concurrent.futures.Future, or (``asyncio_future``)
        an asyncio task or future."""
        if future.cancelled():
            frame, self._unstarted_frame = self._unstarted_frame, None
            if frame is not None and never_ran(frame):
                # Cancelled before its first step: the work never ran, so abandon.
                _abandon(self._decision, self._target)
                return
            # Cancelled while running: not known whether it did its work. A concurrent future is
            # never known not to have run: run_coroutine_threadsafe's can be cancelled mid-run.
            exc: Optional[BaseException] = asyncio.CancelledError()
        elif asyncio_future:
            exc = peek_exception(future)  # asyncio still logs an error nobody else reads
        else:
            exc = future.exception()  # done: returns at once
        self._settle(exc)

    def _abandon_if_never_started(self) -> None:
        if self._started:
            return
        close = getattr(self._awaitable, "close", None)
        if callable(close):
            try:
                close()  # the work never ran: no "never awaited" warning for it either
            except Exception:  # noqa: BLE001 - best effort
                pass
        _abandon(self._decision, self._target)

    async def _finish(self, awaitable: Any) -> Any:
        self._started = True
        try:
            with bound_key(self._downstream_key):
                value = await awaitable
        except BaseException as exc:
            await _settle_tool_off_loop(self._decision, self._target, _settle_outcome(exc, self._spec))
            raise
        await _settle_tool_off_loop(self._decision, self._target, OUTCOME_COMMIT)
        return value


def _failure_outcome(exc: BaseException, spec: Optional[IdempotencySpec]) -> Outcome:
    """How a failed call settles its idempotency key.

    ``release`` when the tool says the failure is definite (``DefiniteFailure`` or a ``release_on``
    type; see is_unconfirmed()): the retry runs with a new
    downstream key. Any other failure may still have done its side effect: ``unconfirmed``, so a
    retry inside the window is not run blind, or with ``release_unconfirmed`` (a downstream that
    dedupes by the key) ``abandon``, so the retry runs with the same downstream key and the
    downstream drops it if the first one landed.
    """
    hold = spec.unconfirmed_types if spec is not None else ()
    release = spec.release_types if spec is not None else ()
    try:
        maybe_done = is_unconfirmed(exc, hold, release)
    except Exception:  # noqa: BLE001 - classifying must never cost the settle
        maybe_done = True
    if maybe_done:
        return OUTCOME_ABANDON if spec is not None and spec.release_unconfirmed else OUTCOME_UNCONFIRMED
    return OUTCOME_RELEASE


def _settle_outcome(exc: Optional[BaseException], spec: Optional[IdempotencySpec]) -> Outcome:
    """The outcome of a call that ended with ``exc``: commit when it is None."""
    return OUTCOME_COMMIT if exc is None else _failure_outcome(exc, spec)


def _resolve_downstream_key(
    call_key: Optional[CallKey], decision: Optional[Dict[str, Any]]
) -> Optional[Union[str, Unkeyable]]:
    """What ``idempotency_key()`` returns, stamped on the span: the per-attempt key the check
    handed back, else the offline key (keyed with the API key, so a receiver cannot reverse it
    by guessing the arguments). An unkeyable call returns its Unkeyable."""
    if call_key is None or call_key.key is None:
        return None if call_key is None else call_key.unkeyable
    if _unanswered(decision):
        raise AgentBlockedError(
            "Policy check could not complete, so this idempotent call's downstream key is unknown. "
            "Blocking instead of sending a key Traccia may have replaced.",
            reasons=list(decision.get("reasons") or []) + [DOWNSTREAM_KEY_UNKNOWN],
        )
    downstream_key = ((decision or {}).get("obligations") or {}).get("idempotency_downstream_key")
    if not (isinstance(downstream_key, str) and KEY_PATTERN.fullmatch(downstream_key)):
        downstream_key = call_key.offline
    pep._stamp_attribute(pep.DOWNSTREAM_KEY_ATTRIBUTE, downstream_key)
    return downstream_key


DOWNSTREAM_KEY_UNKNOWN = "idempotency_downstream_key_unknown"
_UNANSWERED_REASONS = frozenset({"check_error", "check_http_error"})


def _unanswered(decision: Optional[Dict[str, Any]]) -> bool:
    """Whether the decision is the fail-open allow given when Traccia did not answer."""
    reasons = (decision or {}).get("reasons") or []
    return decision is not None and "id" not in decision and bool(_UNANSWERED_REASONS.intersection(reasons))


def _offline_downstream_key(call_key: str) -> str:
    """The downstream key without an answer from Traccia: ``idk1_`` + HMAC-SHA256(API key,
    call key), or the call key when no API key is set."""
    try:
        api_key, _endpoint = pep._credentials()
    except Exception:
        api_key = None
    if not api_key:
        return call_key
    return KEY_PREFIX + hmac.new(api_key.encode("utf-8"), call_key.encode("utf-8"), hashlib.sha256).hexdigest()


def _settle_tool_in_background(
    decision: Optional[Dict[str, Any]], target: Optional[pep.SettleTarget], outcome: Outcome
) -> None:
    """_settle_quietly() on a settle worker; flush_pending_settles() waits for it."""
    if not decision:
        return
    pep._submit_settle(_settle_quietly, decision, target, outcome)


def _abandon(decision: Optional[Dict[str, Any]], target: Optional[pep.SettleTarget]) -> None:
    """Settle a call that never ran: the reservation is abandoned in the background, and a retry
    keeps the attempt (and downstream key) it inherited from an unclear one."""
    _settle_tool_in_background(decision, target, OUTCOME_ABANDON)


# The tool was cancelled during its policy check: abandon what the check reserved.
_abandon_late_check = pep._late_release(_abandon)


async def _settle_tool_off_loop(
    decision: Optional[Dict[str, Any]], target: Optional[pep.SettleTarget], outcome: Outcome
) -> None:
    """Await the settle on a worker thread. If the caller is cancelled before the settle
    started, it runs in the background instead, so it is never lost."""
    if not decision:
        return

    def _if_cancelled(done: Optional[Future]) -> None:
        if done is None:
            _settle_tool_in_background(decision, target, outcome)

    def _settle() -> None:
        _settle_quietly(decision, target, outcome)

    await pep.run_off_loop(_settle, if_cancelled=_if_cancelled)


@asynccontextmanager
async def async_governed_tool_call(
    name: str,
    arguments: Dict[str, Any],
    spec: Optional[IdempotencySpec] = None,
    *,
    key_arguments: Optional[Union[Dict[str, Any], Unkeyable]] = None,
    bound_args: Optional[inspect.BoundArguments] = None,
) -> AsyncIterator[None]:
    """governed_tool_call() for an async tool: the policy check and the settle run on a worker
    thread (pep.run_off_loop()), so a slow policy server never stalls the event loop. The settle is
    still awaited, so it has finished before the tool call returns or raises. A tool cancelled
    during its check abandons what the check reserved once the check returns.
    """
    call_key, wire, wait_s = _prepare(spec, arguments, key_arguments, bound_args)
    if runtime_config.pep_enabled():
        pauses = _in_flight_pauses(wait_s)
        wire_now = wire
        while True:
            try:
                decision, target = await pep.run_off_loop(
                    _check_tool, name, arguments, wire_now, spec, if_cancelled=_abandon_late_check
                )
                break
            except IdempotencyRepeat as exc:
                pause = _next_pause(exc, pauses)
                if pause is None:
                    raise
            await _async_sleep(pause)
            wire_now = _recheck_wire(wire)
    else:
        decision, target = None, None  # no check, so no I/O: stay on the loop
    downstream_key = _resolve_downstream_key(call_key, decision)
    try:
        with _active(call_key), bound_key(downstream_key):
            yield
    except BaseException as exc:
        await _settle_tool_off_loop(decision, target, _settle_outcome(exc, spec))
        raise
    await _settle_tool_off_loop(decision, target, OUTCOME_COMMIT)


def _settle_quietly(
    decision: Optional[Dict[str, Any]], target: Optional[pep.SettleTarget], outcome: Outcome
) -> None:
    """Settle a tool call; never let a settle error replace the tool's result or exception."""
    if not decision:
        return
    try:
        pep.settle_policy(decision, target=target, outcome=outcome)
    except Exception:
        logger.debug("tool settle failed", exc_info=True)
