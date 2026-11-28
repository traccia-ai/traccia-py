"""JSON forms of tool arguments: the payload a policy check sends, and (in idempotency) the
canonical text a key is derived from. Both stop at the same nesting depth."""

from __future__ import annotations

import dataclasses
import enum
import math
import uuid
from datetime import date
from decimal import Decimal
from typing import Any

MAX_DEPTH = 32  # containers, counting the outer one; derive_key's material uses two


def json_safe(value: Any, depth: int = 0) -> Any:
    """``value`` in a form ``json.dumps(allow_nan=False)`` accepts, for a check's ``context.input``.

    Values JSON can already encode come back unchanged, so their payload is the same bytes.
    Decimal and UUID become strings, dates ISO strings, Enums their value, dataclasses and
    pydantic models dicts, tuples and sets lists, NaN and infinities null (as in Node).
    Anything else becomes ``"<TypeName>"``: never ``str()``/``repr()``, which could hold secrets.
    """
    if value is None or isinstance(value, (str, int)):  # bool is an int
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth >= MAX_DEPTH:
        return "<…>"
    if isinstance(value, dict):
        return {
            k if k is None or isinstance(k, (str, int, float)) else f"<{type(k).__name__}>": json_safe(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v, depth + 1) for v in value]
    if isinstance(value, enum.Enum):
        return json_safe(value.value, depth + 1)
    if isinstance(value, (Decimal, uuid.UUID)):
        return str(value)
    if isinstance(value, date):  # and datetime
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe({f.name: getattr(value, f.name) for f in dataclasses.fields(value)}, depth + 1)
    if not isinstance(value, type) and callable(getattr(value, "model_dump", None)):
        try:
            return json_safe(value.model_dump(mode="json"), depth + 1)
        except Exception:
            pass
    return f"<{type(value).__name__}>"
