"""Canonical idempotency keys and the unconfirmed-error classifier.

The shared vector file pins the keys both SDKs derive; the tests here cover what it cannot hold:
Python-only values, reasons, the key accessor and is_unconfirmed().
"""

import asyncio
import dataclasses
import enum
import functools
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, List, Optional
from uuid import UUID

import pytest

import traccia
from traccia.governance.idempotency import (
    _WALK_BUDGET,
    Unkeyable,
    bound_key,
    canonical_json,
    coerce_key_types,
    derive_key,
    drop_defaults,
    idempotency_key,
    is_unconfirmed,
)

VECTOR_FILE = Path(__file__).parent / "fixtures" / "idempotency_vectors.json"
VECTORS = json.loads(VECTOR_FILE.read_text(encoding="utf-8"))
CASES = VECTORS["cases"]


def _native(value: Any) -> Any:
    """Turn the vector file's `$tag` objects into native Python values."""
    if isinstance(value, list):
        return [_native(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        (tag, raw), = value.items()
        if tag in ("$datetime", "$naive_datetime"):
            return datetime.fromisoformat(raw)
        if tag == "$decimal":
            return Decimal(raw)
        if tag == "$uuid":
            return UUID(raw)
        if tag == "$set":
            return set(raw)
        if tag == "$bigint":
            return int(raw)
        if tag == "$float":
            return float(raw)
    return {k: _native(v) for k, v in value.items()}


def _numpy():
    return pytest.importorskip("numpy")


# --- shared vectors ------------------------------------------------------------------------------


def test_vector_file():
    """The file is shared with the Node SDK; both pin the same LF-normalised hash."""
    digest = hashlib.sha256(VECTOR_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == "f2e98b6f8db27fedc3c0569f4b4e3e5b97e61784b27029249618d40b0c7e1def"
    assert VECTORS["version"] == 1
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names))
    for case in CASES:
        assert ("canonical" in case) != bool(case.get("unkeyable"))
        if "canonical" in case:
            assert json.loads(case["canonical"])["v"] == 1


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_shared_vectors(case):
    args = _native(case["args"])
    if "key_types" in case:
        args = coerce_key_types(args, case["key_types"])
    extra = {"extra": _native(case["extra"])} if "extra" in case else {}
    if case.get("unkeyable"):
        with pytest.raises(Unkeyable):
            derive_key(case["agent"], case["tool"], args, case.get("key_fields"), **extra)
        return
    expected = "idk1_" + hashlib.sha256(case["canonical"].encode("utf-8")).hexdigest()
    assert derive_key(case["agent"], case["tool"], args, case.get("key_fields"), **extra) == expected


# --- canonical form of Python-only values --------------------------------------------------------


class _Color(enum.Enum):
    RED = "r"


@pytest.mark.parametrize("value, canonical", [
    # A Python int is arbitrary precision, so it keys by its exact digits (as a Node bigint).
    pytest.param(10 ** 30, "1000000000000000000000000000000", id="int beyond 2**64"),
    pytest.param(float(2 ** 53 - 1), "9007199254740991", id="integral float"),
    # A float beyond 2**53 keys by the exact digits of the double, as a JS number does.
    pytest.param(2.0 ** 60, "1152921504606846976", id="integral float beyond 2**53"),
    # A Node tool gets money as a JS number: a Decimal keys as the same number.
    pytest.param(Decimal("-1.50"), "-1.5", id="Decimal drops trailing zeros"),
    pytest.param(Decimal("0.00005"), "0.00005", id="small Decimal in plain digits"),
    # Python writes 1e-05 where JS writes 0.00001: both key as plain digits.
    pytest.param(-2.5e-10, "-0.00000000025", id="tiny float in plain digits"),
    pytest.param(5e-324, "0." + "0" * 323 + "5", id="smallest float in plain digits"),
    # A naive datetime has no zone marker, so it never equals an aware one (those end in Z).
    pytest.param(datetime(2026, 1, 2, 3, 4, 5, 678000), '"2026-01-02T03:04:05.678"', id="naive datetime"),
    pytest.param(datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=timezone.utc), '"2026-01-02T03:04:05.678Z"',
                 id="aware datetime"),
    pytest.param(datetime(2026, 1, 2, 3, 4, 5, 678999, tzinfo=timezone.utc), '"2026-01-02T03:04:05.678Z"',
                 id="aware datetime truncated to milliseconds"),
    pytest.param(_Color.RED, '"r"', id="enum by value"),
    pytest.param(date(2026, 1, 2), '"2026-01-02"', id="date"),
    pytest.param((1, False), "[1,false]", id="tuple as a list"),
])
def test_canonical_json(value, canonical):
    assert canonical_json(value) == canonical


def test_decimal_ignores_ambient_context():
    long = Decimal("1.23456789012345678901234567890100")
    want = "1.234567890123456789012345678901"
    assert canonical_json(long) == want
    with localcontext() as ctx:
        ctx.prec = 6
        assert canonical_json(long) == want
        assert canonical_json(Decimal("1E+2")) == "100"
        assert canonical_json(Decimal("123456789.000")) == "123456789"


@dataclasses.dataclass
class _Mail:
    to: str
    tags: List[str]
    cc: Optional[str] = None


@dataclasses.dataclass
class _Charge:
    """An InitVar sets an attribute that is not a field: Charge(5, "usd") and Charge(5, "jpy")
    would share a key."""

    amount: int
    currency: dataclasses.InitVar[str]

    def __post_init__(self, currency):
        self.cents_per_unit = 1 if currency == "jpy" else 100


@dataclasses.dataclass
class _Totalled:
    amount: int
    total: int = dataclasses.field(init=False, default=0)


@dataclasses.dataclass
class _Account:
    """The tenant lives in a private slot that is not a field."""

    __slots__ = ("account", "__tenant")
    account: str

    def __init__(self, account, tenant):
        self.account = account
        self.__tenant = tenant


@dataclasses.dataclass
class _Slotted:  # what dataclass(slots=True) builds (Python 3.10+): a slot per field
    __slots__ = ("account",)
    account: str


@dataclasses.dataclass(frozen=True)
class _Line:
    sku: str
    qty: int

    @functools.cached_property
    def total(self):
        return self.qty * 5


class _cached_property:
    """The shape of django's, werkzeug's and the cached-property package's cached_property: a
    non-data descriptor that caches in __dict__ under its own name."""

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


@dataclasses.dataclass
class _Order:
    order_id: str

    @_cached_property
    def label(self):
        return self.order_id.upper()

    def note(self):
        return "a method"


def _read(value, attribute):
    getattr(value, attribute)  # caches the property in the instance __dict__
    return value


def _shadowed_method():
    order = _Order("o1")
    order.note = "hidden"  # instance data that shadows a plain method: data the key cannot see
    return order


def _pydantic_mail():
    pydantic = pytest.importorskip("pydantic")

    class Mail(pydantic.BaseModel):
        to: str
        tags: List[str]
        cc: Optional[str] = None

    return Mail(to="a@x", tags=["x", "y"])


@pytest.mark.parametrize("make_args, reason", [
    # Paths are relative to the arguments, never the internal material ("args.").
    pytest.param(lambda: {"client": object()}, "client is object", id="object"),
    pytest.param(lambda: {"items": [1, {"c": b"x"}]}, "items[1].c is bytes", id="bytes deep inside"),
    pytest.param(lambda: {"s": "a" + chr(0xD800)}, "s has a lone surrogate", id="lone surrogate"),
    # the reason never repeats the surrogate itself
    pytest.param(lambda: {"m": {chr(0xD800): 1}}, "m has a key with a lone surrogate", id="lone surrogate key"),
    pytest.param(lambda: {"m": {chr(0xE9): 1, "e" + chr(0x301): 2}},
                 "m has two keys that are equal after NFC normalization", id="NFC key collision"),
    pytest.param(lambda: {"m": {1: "a"}}, "m has a int key", id="non-string key"),
    pytest.param(lambda: {"x": float("inf")}, "x is inf", id="infinite float"),
    pytest.param(lambda: {"x": Decimal("NaN")}, "x is a non-finite Decimal", id="NaN Decimal"),
    pytest.param(lambda: {"at": datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=5)))},
                 "at is a datetime outside the UTC range", id="datetime outside the UTC range"),
    # Decimal("1E+200000000") would be written out as a 200 MB string.
    pytest.param(lambda: {"a": Decimal("1E+401")}, "a is a Decimal beyond 1e400", id="Decimal beyond 1e400"),
    pytest.param(lambda: {"a": Decimal("1E-401")}, "a is a Decimal beyond 1e400", id="Decimal below 1e-400"),
    pytest.param(lambda: {"a": Decimal("1E+200000000")}, "a is a Decimal beyond 1e400", id="huge Decimal"),
    pytest.param(lambda: {"amount": 10 ** 5000}, "amount is an int too long to write out", id="int too long"),
    pytest.param(lambda: {"c": _Charge(5, "usd")}, "c is _Charge with attribute 'cents_per_unit' that is not a field",
                 id="dataclass attribute that is not a field"),
    pytest.param(lambda: {"a": _Account("acct", "tenant-A")}, "a is _Account with attribute '_Account__tenant' that is not a field",
                 id="dataclass slot that is not a field"),
    pytest.param(lambda: {"o": _shadowed_method()}, "o is _Order with attribute 'note' that is not a field",
                 id="instance data shadowing a method"),
    pytest.param(lambda: {"a": _numpy().array([1, 2])}, "a is ndarray", id="numpy array"),
])
def test_unkeyable(make_args, reason):
    with pytest.raises(Unkeyable) as info:
        derive_key("a1", "t", make_args())
    assert str(info.value) == reason


@pytest.mark.parametrize("make_pair, same", [
    pytest.param(lambda: ({"amt": Decimal("10.50")}, {"amt": 10.5}), True, id="Decimal and float"),
    pytest.param(lambda: ({"amt": Decimal("100.00")}, {"amt": 100}), True, id="Decimal and int"),
    pytest.param(lambda: ({"amt": Decimal("10.5")}, {"amt": "10.5"}), False, id="a number and a string differ"),
    pytest.param(lambda: ({"a": Decimal("0E+999999")}, {"a": 0}), True, id="zero with a huge exponent"),
    pytest.param(lambda: ({"a": Decimal("1E+400")}, {"a": 10 ** 400}), True, id="Decimal at the bound"),
    pytest.param(lambda: ({"mail": _Mail("a@x", ["x", "y"])}, {"mail": {"to": "a@x", "tags": ["x", "y"], "cc": None}}),
                 True, id="dataclass by its fields"),
    pytest.param(lambda: ({"mail": _pydantic_mail()}, {"mail": {"to": "a@x", "tags": ["x", "y"], "cc": None}}),
                 True, id="pydantic model by its fields"),
    pytest.param(lambda: ({"c": _Totalled(5)}, {"c": {"amount": 5, "total": 0}}), True, id="init=False field"),
    pytest.param(lambda: ({"a": _Slotted("acct")}, {"a": {"account": "acct"}}), True, id="slotted dataclass"),
    # Reading a cached property stores it in __dict__: keying must not depend on read order.
    pytest.param(lambda: ({"l": _read(_Line("s", 2), "total")}, {"l": _Line("s", 2)}), True,
                 id="functools.cached_property read or not"),
    pytest.param(lambda: ({"o": _read(_Order("o1"), "label")}, {"o": _Order("o1")}), True,
                 id="other cached_property read or not"),
    pytest.param(lambda: ({"n": _numpy().int64(7)}, {"n": 7}), True, id="numpy int"),
    pytest.param(lambda: ({"b": _numpy().bool_(True)}, {"b": True}), True, id="numpy bool"),
    pytest.param(lambda: ({"f": _numpy().float32(0.5)}, {"f": 0.5}), True, id="numpy float"),
])
def test_same_key(make_pair, same):
    left, right = make_pair()
    assert (derive_key("a1", "t", left) == derive_key("a1", "t", right)) is same


@pytest.mark.parametrize("make_value, kind, converted", [
    # Beyond the bound a number stays text: key_types never makes a call unkeyable.
    pytest.param(lambda: "1" + "0" * 401, "number", None, id="number beyond the bound"),
    # Values key_types cannot convert stay as given, so the key says why it cannot be keyed.
    pytest.param(lambda: Decimal("NaN"), "string", None, id="NaN Decimal"),
    pytest.param(lambda: Decimal("sNaN"), "string", None, id="signalling NaN Decimal"),
    pytest.param(lambda: Decimal("1E+401"), "string", None, id="Decimal beyond the bound"),
    pytest.param(lambda: 10 ** 5000, "string", None, id="int too long to write out"),
    pytest.param(lambda: float("inf"), "string", None, id="infinite float"),
    pytest.param(lambda: _numpy().int64(123), "string", "123", id="numpy int converts like an int"),
])
def test_coerce_key_types(make_value, kind, converted):
    value = make_value()
    out = coerce_key_types({"amount": value}, {"amount": kind})["amount"]
    if converted is not None:
        assert out == converted
        return
    assert out is value
    if kind == "string":
        with pytest.raises(Unkeyable):
            derive_key("a1", "t", {"amount": out})


def _canonical_key(agent, tool, args_json):
    material = '{"agent":"%s","args":%s,"tool":"%s","v":1}' % (agent, args_json, tool)
    return "idk1_" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def test_drop_defaults_keeps_an_explicit_none_for_a_non_none_default():
    # set_spend_limit("acct", None) removes the limit; set_spend_limit("acct") keeps the default 100.
    defaults = {"limit": 100, "note": None}

    def key(**arguments):
        return derive_key("a1", "set_spend_limit", drop_defaults(arguments, defaults))

    assert key(account="acct", limit=None) != key(account="acct", limit=100)
    assert key(account="acct", limit=100) == derive_key("a1", "set_spend_limit", {"account": "acct"})
    assert key(account="acct", limit=None) == _canonical_key("a1", "set_spend_limit", '{"account":"acct","limit":null}')
    # A None default, or no default at all, still drops None.
    assert key(account="acct", note=None) == key(account="acct")
    assert derive_key("a1", "t", drop_defaults({"x": None}, {})) == derive_key("a1", "t", {})


def test_key_fields_naming_a_field_the_call_lacks(caplog):
    """The keys are pinned by the key_fields vectors; this covers the typo warning."""
    from traccia.governance import idempotency as idem

    idem._warned_missing.clear()  # warned once per process: start from none
    with caplog.at_level("WARNING", logger="traccia.governance.idempotency"):
        derive_key("a1", "t", {"to": "a", "cc": "b"}, ["to", "too"])
        derive_key("a1", "t", {"to": "a"}, ["to", "too"])
    # reported once per tool and field
    assert ["'too'" in r.getMessage() for r in caplog.records] == [True]
    caplog.clear()
    with caplog.at_level("WARNING", logger="traccia.governance.idempotency"):
        derive_key("a1", "t", {"to": "a"}, ["to", "cc?"])  # 'cc?' may be missing: no warning
    assert caplog.records == []


# --- the key accessor ----------------------------------------------------------------------------


def _traceback_depth(exc):
    tb, n = exc.__traceback__, 0
    while tb is not None:
        tb, n = tb.tb_next, n + 1
    return n


def test_idempotency_key_accessor():
    with pytest.raises(RuntimeError, match="outside an idempotent tool"):
        idempotency_key()
    with bound_key("outer"):
        assert idempotency_key() == "outer"
        with bound_key("inner"):
            assert idempotency_key() == "inner"
        assert idempotency_key() == "outer"
        with pytest.raises(ValueError):
            with bound_key("boom"):
                raise ValueError("x")
        assert idempotency_key() == "outer"
        bad = Unkeyable("client is object")
        with bound_key(bad):
            depths = []
            for _ in range(3):
                with pytest.raises(Unkeyable, match="client is object") as info:
                    idempotency_key()
                assert info.value is bad
                depths.append(_traceback_depth(info.value))
            assert depths[0] == depths[1] == depths[2]  # tracebacks don't pile up
    with pytest.raises(RuntimeError):
        idempotency_key()


def test_exported_from_package():
    import traccia

    assert traccia.idempotency_key is idempotency_key
    assert traccia.Unkeyable is Unkeyable
    assert {"idempotency_key", "Unkeyable"} <= set(traccia.__all__)


# --- is_unconfirmed ------------------------------------------------------------------------------


def err(name, module="myapp.errors", bases=(Exception,), **attrs):
    """An error of a class named ``name`` from ``module``, carrying ``attrs``."""
    exc = type(name, bases, {"__module__": module})("x")
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


class _Declined(Exception):
    pass


class _MaybeSent(Exception):
    pass


def _raised(fn, *args):
    try:
        fn(*args)
    except BaseException as exc:  # noqa: BLE001 - the test inspects any raised error
        return exc
    raise AssertionError("expected an exception")


def _raise_while_handling(first, then, *, chain="context"):
    """``then`` raised while handling ``first``: implicitly, ``from first`` or ``from None``."""
    def run():
        try:
            raise first
        except BaseException as exc:  # noqa: BLE001
            if chain == "cause":
                raise then from exc
            if chain == "none":
                raise then from None
            raise then
    return _raised(run)


def _eg(message, errors):
    try:
        return ExceptionGroup(message, errors)  # noqa: F821 - Python 3.11+
    except NameError:
        pytest.skip("ExceptionGroup needs Python 3.11")


@pytest.mark.parametrize("make, expected", [
    # held: the tool did not say the failure is definite. An error describes only the last request:
    # a client that retried by itself or followed a redirect may have done the work before it.
    pytest.param(lambda: ValueError("bad input"), True, id="the tool's own error"),
    pytest.param(ConnectionRefusedError, True, id="a refused connection"),
    pytest.param(lambda: err("ConnectError", "httpx"), True, id="httpx ConnectError"),
    pytest.param(lambda: err("RateLimitError", "openai", status_code=429), True, id="a 429"),
    pytest.param(lambda: err("InvalidRequestError", "stripe", http_status=400), True, id="a 400"),
    pytest.param(lambda: TimeoutError("read timed out"), True, id="a timeout"),
    pytest.param(asyncio.CancelledError, True, id="asyncio cancel"),
    pytest.param(KeyboardInterrupt, True, id="KeyboardInterrupt"),
    # the author's say
    pytest.param(lambda: traccia.DefiniteFailure("card declined"), False, id="DefiniteFailure"),
    pytest.param(lambda: type("Declined", (traccia.DefiniteFailure,), {})("x"), False, id="a DefiniteFailure subclass"),
    pytest.param(lambda: _raise_while_handling(TimeoutError("read"), traccia.DefiniteFailure("declined"), chain="cause"),
                 False, id="DefiniteFailure vouches for what is below it"),
    # an error raised below the author's say does not count: only the error the tool raised does
    pytest.param(lambda: _raise_while_handling(traccia.DefiniteFailure("x"), ValueError("then")), True,
                 id="an error raised while handling a DefiniteFailure"),
    # a group is judged as itself: a task group leaves out the siblings it cancelled
    pytest.param(lambda: _eg("fan-out", [traccia.DefiniteFailure("a"), traccia.DefiniteFailure("b")]), True,
                 id="a group of DefiniteFailures"),
])
def test_is_unconfirmed(make, expected):
    assert is_unconfirmed(make()) is expected


def test_release_on_and_unconfirmed_on():
    assert is_unconfirmed(_Declined("x"), release_on=(_Declined,)) is False
    assert is_unconfirmed(_Declined("x"), release_on=(_MaybeSent,)) is True
    assert is_unconfirmed(err("Sub", bases=(_Declined,)), release_on=(_Declined,)) is False  # subclasses count
    # unconfirmed_on wins, on the error itself or anywhere below it
    assert is_unconfirmed(_Declined("x"), unconfirmed_on=(_Declined,), release_on=(_Declined,)) is True
    below = [
        _raise_while_handling(_MaybeSent("sent"), traccia.DefiniteFailure("x"), chain="cause"),
        _raise_while_handling(_MaybeSent("sent"), traccia.DefiniteFailure("x")),
        _raise_while_handling(_MaybeSent("sent"), traccia.DefiniteFailure("x"), chain="none"),
    ]
    for exc in below:
        assert is_unconfirmed(exc, unconfirmed_on=(_MaybeSent,)) is True
        assert is_unconfirmed(exc) is False
    group = _eg("fan-out", [_Declined("a"), _MaybeSent("b")])
    group_type = type(group)
    assert is_unconfirmed(group, release_on=(group_type,)) is False  # the author's say on the group
    assert is_unconfirmed(group, unconfirmed_on=(_MaybeSent,), release_on=(group_type,)) is True


def test_the_unconfirmed_on_search_is_bounded():
    def chain(n):
        top = traccia.DefiniteFailure("top")
        current = top
        for _ in range(n):
            below = ValueError("x")
            current.__cause__ = below
            current = below
        return top

    assert is_unconfirmed(chain(10), unconfirmed_on=(_MaybeSent,)) is False
    assert is_unconfirmed(chain(_WALK_BUDGET + 5), unconfirmed_on=(_MaybeSent,)) is True  # unread: held
    looped = traccia.DefiniteFailure("a")
    other = ValueError("b")
    looped.__cause__, other.__cause__ = other, looped
    assert is_unconfirmed(looped, unconfirmed_on=(_MaybeSent,)) is False  # a cycle ends


def test_a_group_whose_members_cannot_be_read_still_settles():
    class Hostile(_eg("probe", [ValueError()]).__class__):
        @property
        def exceptions(self):
            raise RuntimeError("lazy")

    group = Hostile("x", [ValueError()])
    assert is_unconfirmed(group, unconfirmed_on=(_MaybeSent,), release_on=(Hostile,)) is False
