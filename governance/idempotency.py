"""Canonical idempotency keys for tool calls.

Keys must match the Node SDK byte for byte: see tests/fixtures/idempotency_vectors.json.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import json
import logging
import math
import re
import types
import unicodedata
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Sequence, Tuple, Union

from traccia.governance.serialization import MAX_DEPTH

KEY_PREFIX = "idk1_"
# A call key or downstream key: the prefix and a SHA-256 or HMAC-SHA256 hex digest. Use fullmatch.
KEY_PATTERN = re.compile(re.escape(KEY_PREFIX) + "[0-9a-f]{64}")

# How a tool call settles its idempotency key. ``release`` and ``abandon`` both let a retry run:
# ``release`` gives it a new downstream key, ``abandon`` keeps the one the attempt had.
# ``unconfirmed`` holds the key for the window, as the call may have happened.
Outcome = Literal["commit", "release", "unconfirmed", "abandon"]
OUTCOME_COMMIT: Outcome = "commit"
OUTCOME_RELEASE: Outcome = "release"
OUTCOME_UNCONFIRMED: Outcome = "unconfirmed"
OUTCOME_ABANDON: Outcome = "abandon"

# How long an idempotent tool denied because the same call is in flight waits and checks again.
DEFAULT_IN_FLIGHT_WAIT_S = 30.0


class Unkeyable(ValueError):
    """The arguments cannot be turned into a stable idempotency key."""


@dataclasses.dataclass(frozen=True)
class IdempotencySpec:
    """An idempotent tool's settings, validated once when the tool is decorated.

    ``defaults``: each parameter's default, already converted by ``key_types``, so an argument
    left at its default is left out of the key; ``default_forms``: the canonical form of each
    one that can never change (see fixed_default_forms()). ``key_extra`` is called with the
    tool's own arguments. ``var_keyword``: the name of the tool's ``**kwargs`` parameter, if it
    has one.
    """

    name: str
    key_fields: Optional[Tuple[str, ...]] = None
    var_keyword: Optional[str] = None
    scope: str = "agent"
    key_types: Dict[str, str] = dataclasses.field(default_factory=dict)
    defaults: Dict[str, Any] = dataclasses.field(default_factory=dict)
    default_forms: Dict[str, str] = dataclasses.field(default_factory=dict)
    unconfirmed_types: Tuple[type, ...] = ()
    release_types: Tuple[type, ...] = ()
    release_unconfirmed: bool = False
    key_extra: Optional[Callable[..., Any]] = None
    in_flight_wait_s: float = DEFAULT_IN_FLIGHT_WAIT_S


_bound: ContextVar[Optional[Union[str, Unkeyable]]] = ContextVar("traccia_idempotency_key", default=None)


def _nfc(s: str, reason: str) -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        raise Unkeyable(reason) from None
    return unicodedata.normalize("NFC", s)


def _quote(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _is_numpy_scalar(v: Any) -> bool:
    """A numpy integer, floating or bool scalar, recognised without importing numpy."""
    cls = type(v)
    if cls.__module__ != "numpy" or getattr(v, "ndim", None) != 0:
        return False
    return any(c.__module__ == "numpy" and c.__name__ in ("integer", "floating", "bool_", "bool")
               for c in cls.__mro__)


class _KeyedNone:
    """An explicit None the key keeps, as null (see drop_defaults)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "None"


KEYED_NONE = _KeyedNone()
# What key_types trims around a value: ASCII whitespace only, as the Node SDK does. str.strip()
# also strips U+0085 and U+001C-001F and JS trim() strips U+FEFF, so either default would split keys.
_KEY_SPACE = " \t\n\r\f\v"
# Decimals (and key_types numbers) whose exponent is beyond this are not keyed as numbers.
_MAX_DECIMAL_EXPONENT = 400


def _enc(v: Any, path: str, depth: int) -> str:
    where = path or "value"
    if v is None or v is KEYED_NONE:
        return "null"
    if _is_numpy_scalar(v):
        v = v.item()  # numpy int, float and bool key like the Python value
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, enum.Enum):
        return _enc(v.value, path, depth)
    if isinstance(v, int):
        try:
            return str(int(v))
        except ValueError:  # beyond sys.get_int_max_str_digits(), 4300 digits by default
            raise Unkeyable(f"{where} is an int too long to write out") from None
    if isinstance(v, float):
        if not math.isfinite(v):
            raise Unkeyable(f"{where} is {v!r}")
        if v.is_integer():
            # The exact digits of the double, like JS BigInt(n).toString(), even beyond 2**53
            # (where a JS number may already have rounded an id: pass those as str).
            return str(int(v))
        text = repr(float(v))  # shortest round-trip digits, the same digits JS picks
        if "e" in text:
            # Python writes 1e-05 where JS writes 0.00001: plain digits on both sides.
            text = format(Decimal(text), "f")
        return text
    if isinstance(v, Decimal):
        # A JSON number in plain digits, trailing zeros stripped: Decimal("10.50") keys as 10.5.
        if not v.is_finite():
            raise Unkeyable(f"{where} is a non-finite Decimal")
        if v == 0:
            return "0"
        if abs(v.adjusted()) > _MAX_DECIMAL_EXPONENT:
            # In plain digits Decimal("1E+200000000") is a 200 MB string, and no JS number holds it.
            raise Unkeyable(f"{where} is a Decimal beyond 1e{_MAX_DECIMAL_EXPONENT}")
        sign, digits, exp = v.as_tuple()  # strip trailing zeros without any decimal context
        while digits[-1] == 0:
            digits, exp = digits[:-1], exp + 1
        return format(Decimal((sign, digits, exp)), "f")
    if isinstance(v, str):
        return _quote(_nfc(v, f"{where} has a lone surrogate"))
    if isinstance(v, datetime):
        if v.utcoffset() is None:
            # A naive datetime keys as its wall time with no zone marker, so it never equals an
            # aware time (those end in Z).
            return '"%04d-%02d-%02dT%02d:%02d:%02d.%03d"' % (
                v.year, v.month, v.day, v.hour, v.minute, v.second, v.microsecond // 1000)
        try:
            u = v.astimezone(timezone.utc)
        except OverflowError:
            raise Unkeyable(f"{where} is a datetime outside the UTC range") from None
        return '"%04d-%02d-%02dT%02d:%02d:%02d.%03dZ"' % (
            u.year, u.month, u.day, u.hour, u.minute, u.second, u.microsecond // 1000)
    if isinstance(v, date):
        return _quote(v.isoformat())
    if isinstance(v, uuid.UUID):
        return _quote(str(v))
    kind = type(v).__name__
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        names = {f.name for f in dataclasses.fields(v)}
        extra = sorted(k for k in _instance_data(v) if k not in names)
        if extra:
            # An attribute set outside the fields (from an InitVar in __post_init__, say) would be
            # left out, so two different values could share a key.
            raise Unkeyable(f"{where} is {kind} with attribute '{extra[0]}' that is not a field")
        v = {name: getattr(v, name) for name in (f.name for f in dataclasses.fields(v))}
    elif not isinstance(v, (list, tuple, dict, type)) and callable(getattr(v, "model_dump", None)):
        v = v.model_dump(mode="python")
    if not isinstance(v, (list, tuple, dict)):
        raise Unkeyable(f"{where} is {kind}")
    if depth >= MAX_DEPTH:
        raise Unkeyable(f"{where} is nested more than {MAX_DEPTH} levels deep")
    if isinstance(v, dict):
        return _object(v, path, where, depth + 1)
    return "[" + ",".join(_enc(x, f"{path}[{i}]", depth + 1) for i, x in enumerate(v)) + "]"


def _class_attr(cls: type, name: str) -> Any:
    for c in cls.__mro__:
        if name in vars(c):
            return vars(c)[name]
    return None


def _caches_here(attr: Any) -> bool:
    """Whether a class attribute caches its value in the instance ``__dict__`` under its own name:
    ``functools.cached_property``, or any non-data descriptor that is not a plain method
    (django's, werkzeug's and the ``cached-property`` package's cached_property)."""
    if isinstance(attr, functools.cached_property):
        return True
    if attr is None or isinstance(attr, (types.FunctionType, classmethod, staticmethod, property)):
        return False
    kind = type(attr)
    return hasattr(kind, "__get__") and not hasattr(kind, "__set__") and not hasattr(kind, "__delete__")


def _instance_data(v: Any) -> list:
    """The names of the attributes a dataclass instance holds: its ``__dict__`` entries, less the
    values a cached property cached there (derived from the fields, and present only once read,
    so keying them would make a value keyable or not by read order), plus the ``__slots__`` it
    fills (a slot that is not a field holds data the key would leave out)."""
    cls = type(v)
    try:
        held = [k for k in vars(v) if not _caches_here(_class_attr(cls, k))]
    except TypeError:  # no __dict__
        held = []
    for c in cls.__mro__:
        slots = vars(c).get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot.startswith("__") and not slot.endswith("__"):
                slot = "_" + c.__name__.lstrip("_") + slot  # a private name, as Python stores it
            if slot not in ("__dict__", "__weakref__") and slot not in held and hasattr(v, slot):
                held.append(slot)
    return held


def _object(d: Dict[Any, Any], path: str, where: str, depth: int) -> str:
    items: Dict[str, Any] = {}
    for k, val in d.items():
        if not isinstance(k, str):
            raise Unkeyable(f"{where} has a {type(k).__name__} key")
        key = _nfc(k, f"{where} has a key with a lone surrogate")
        if key in items:
            raise Unkeyable(f"{where} has two keys that are equal after NFC normalization")
        items[key] = val
    return "{" + ",".join(
        _quote(k) + ":" + _enc(items[k], f"{path}.{k}" if path else k, depth) for k in sorted(items)
    ) + "}"


def canonical_json(value: Any) -> str:
    """Canonical JSON for ``value``; raises ``Unkeyable`` if it has no stable form."""
    return _enc(value, "", 0)


KEY_TYPES = ("string", "number", "boolean")
_PLAIN_NUMBER = re.compile(r"-?[0-9]+(\.[0-9]+)?")


def _coerce(value: Any, kind: str) -> Any:
    if _is_numpy_scalar(value):
        value = value.item()  # np.int64(123) converts like 123
    if kind == "string" and isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            return _enc(value, "", 0)  # its canonical number text: 123 -> "123", 10.0 -> "10"
        except ValueError:  # Unkeyable is one
            # NaN, a Decimal beyond the bound, an int too long to write out: left as given, so the
            # key says why it cannot be keyed, as without key_types. key_types never fails a call.
            return value
    if kind in ("number", "boolean") and isinstance(value, str):
        text = value.strip(_KEY_SPACE)
        if kind == "number" and _PLAIN_NUMBER.fullmatch(text):
            number = Decimal(text)  # keys as a normalized number: "007.50" -> 7.5
            # Beyond the bound it stays text: key_types never makes a call unkeyable.
            return number if number == 0 or abs(number.adjusted()) <= _MAX_DECIMAL_EXPONENT else value
        if kind == "boolean" and text.lower() in ("true", "false"):
            return text.lower() == "true"
    return value


def coerce_key_types(arguments: Dict[str, Any], key_types: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """``arguments`` with the named fields converted to their declared type, for the key only.

    So 123 and "123" key the same. ``string`` turns a number into its canonical text; ``number``
    turns a plain decimal string (no exponent) into a number; ``boolean`` turns "true"/"false"
    into a bool. Anything else is left as it is; the tool still gets the original values.
    """
    if not key_types:
        return arguments
    out = dict(arguments)
    for name, kind in key_types.items():
        if name in out:
            out[name] = _coerce(out[name], kind)
    return out


# Defaults whose canonical form can never change. A tuple of them is one too.
_FIXED_TYPES = (type(None), bool, int, float, str, Decimal, uuid.UUID, date)


def _fixed(value: Any) -> bool:
    kind = type(value)
    if kind is tuple:
        return all(type(v) in _FIXED_TYPES for v in value)
    if kind is datetime:
        return value.tzinfo is None or type(value.tzinfo) is timezone
    return kind in _FIXED_TYPES


def fixed_default_forms(defaults: Dict[str, Any]) -> Dict[str, str]:
    """The canonical form of each default that is an immutable value, read once when the tool is
    decorated. Any other default (a list, a dict, an object) may change between calls, so
    drop_defaults() canonicalises it on every call, as it does one that cannot be keyed."""
    forms = {}
    for name, value in defaults.items():
        if _fixed(value):
            try:
                forms[name] = canonical_json(value)
            except Exception:  # noqa: BLE001 - not keyable: compared on every call
                pass
    return forms


def drop_defaults(
    arguments: Dict[str, Any], defaults: Dict[str, Any], forms: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """``arguments`` with those left at their parameter's default set to None (left out of the key).

    A call keys the same whether a default was passed or not, as a Node call that omits the
    field does. Equal means the same object (so a sentinel default never makes the call
    unkeyable) or the same canonical form. ``forms``: the defaults' canonical forms already known
    (see fixed_default_forms()). An explicit None for a parameter whose default is something else
    (``set_spend_limit("acct", None)`` removes the limit) is kept in the key as null.
    """
    def at_default(name: str) -> bool:
        if name not in defaults:
            return False
        value, default = arguments[name], defaults[name]
        if value is default:
            return True
        try:
            form = canonical_json(value)
            return form == (forms[name] if forms and name in forms else canonical_json(default))
        except Exception:  # unkeyable on either side: keep it, and let derive_key say why
            return False

    # None rather than removed: the key drops None either way, but a key_fields field left at
    # its default still counts as given, not as missing.
    def keyed(name: str, value: Any) -> Any:
        if at_default(name):
            return None
        if value is None and name in defaults and defaults[name] is not None:
            return KEYED_NONE
        return value

    return {k: keyed(k, v) for k, v in arguments.items()}


_warned_missing: set = set()


def _warn_missing_once(tool: str, name: str) -> None:
    if (tool, name) in _warned_missing:
        return
    _warned_missing.add((tool, name))
    logging.getLogger(__name__).warning(
        "idempotent tool %r: key_fields names %r, which this call does not have, so it is left out "
        "of the key; write '%s?' if it may be left out, or fix the name", tool, name, name)


def derive_key(agent_id: Optional[str], tool: str, arguments: Dict[str, Any],
               key_fields: Optional[Sequence[str]] = None, extra: Any = None) -> str:
    """``idk1_`` + SHA-256 of the canonical call. ``None`` arguments are dropped.

    Same bytes as ``canonical_json({"agent", "args", "tool", "v": 1})``, built here so
    that reasons name argument paths (``client is object``) rather than ``args.client``.
    ``extra`` (from a tool's ``key_extra``: inputs that are not arguments, such as the
    account on ``self`` or the current tenant) adds an ``"extra"`` member; ``None`` adds
    nothing, so keys without it are unchanged.
    """
    if key_fields is None:
        names = list(arguments)
    else:
        names = []
        for field in key_fields:
            name, optional = (field[:-1], True) if field.endswith("?") else (field, False)
            if name in arguments:
                names.append(name)
            elif not optional:
                # Left out, like 'cc?', so whether the call is keyable never depends on what
                # this one call happened to carry; the warning points at a typo.
                _warn_missing_once(tool, name)
    # With no key field set, every such call shares one key: only key_fields tell calls apart, so
    # a changing request id or timestamp never slips a repeat past.
    args = {k: arguments[k] for k in names if arguments[k] is not None}
    agent = _enc("" if agent_id is None else agent_id, "agent", 1)
    extra_member = "" if extra is None else ',"extra":' + _enc(extra, "key_extra", 1)
    material = '{"agent":%s,"args":%s%s,"tool":%s,"v":1}' % (
        agent, _enc(args, "", 1), extra_member, _enc(tool, "tool", 1))
    return KEY_PREFIX + hashlib.sha256(material.encode("utf-8")).hexdigest()


class DefiniteFailure(Exception):
    """Raise from an idempotent tool to say its call surely did not happen, so the key is freed
    and a retry runs: ``raise traccia.DefiniteFailure("card declined") from err``. Raise it only
    where no request can have done the work: it vouches for everything below it in the chain."""


# How many errors one classification reads looking for an ``unconfirmed_on`` type (causes, handled
# contexts, exception group members). A walk cut short holds the call.
_WALK_BUDGET = 1000


def _attr(obj: object, name: str) -> object:
    """``getattr`` that never raises: a property on a third-party error must not cost the settle."""
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001
        return None


def _chained(error: BaseException) -> list:
    """The errors linked to ``error``: its cause, the error it was raised while handling (also
    behind ``from None``) and, for an exception group, its members."""
    linked = [_attr(error, "__cause__"), _attr(error, "__context__")]
    if any(c.__name__ == "BaseExceptionGroup" for c in type(error).__mro__):
        members = _attr(error, "exceptions")
        if isinstance(members, (tuple, list)):
            linked.extend(members)
    return linked


def is_unconfirmed(exc: BaseException, unconfirmed_on: Sequence[type] = (),
                   release_on: Sequence[type] = ()) -> bool:
    """Whether a tool call that raised ``exc`` may have done its side effect, so its key is held.

    Held unless the error the tool raised is the author's say that the call did not happen: a
    ``DefiniteFailure`` or a ``release_on`` type, with no ``unconfirmed_on`` type anywhere in its
    chain. Nothing else frees a call: an error describes only the last request, and a client that
    retried by itself, followed a redirect, or a tool that made several requests, may have done the
    work before it. An exception group is judged as itself, not by its members (a task group leaves
    out the siblings it cancelled).
    """
    if not isinstance(exc, (DefiniteFailure, *release_on)):
        return True
    hold_types = tuple(unconfirmed_on)
    if not hold_types:
        return False
    stack: list = [exc]
    seen: set = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        if len(seen) >= _WALK_BUDGET:
            return True  # errors still unread: one of them may be an unconfirmed_on type
        seen.add(id(current))
        if isinstance(current, hold_types):
            return True
        stack.extend(_chained(current))
    return False


def idempotency_key() -> str:
    """The key of the idempotent tool call running now."""
    value = _bound.get()
    if value is None:
        raise RuntimeError("idempotency_key() called outside an idempotent tool")
    if isinstance(value, Unkeyable):
        raise value.with_traceback(None)
    return value


@contextmanager
def bound_key(value: Optional[Union[str, Unkeyable]]) -> Iterator[None]:
    """Make ``idempotency_key()`` return ``value`` (or raise it) inside the block. None changes
    nothing: a tool that is not idempotent leaves the key of the call around it."""
    if value is None:
        yield
        return
    token = _bound.set(value)
    try:
        yield
    finally:
        _bound.reset(token)
