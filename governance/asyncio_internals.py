"""What a governed tool reads from asyncio and CPython internals, kept in one place.

Neither has a public API: whether a cancelled task's coroutine ever ran (its frame's last
instruction against the entry that opens the body, CPython 3.11+; where that cannot be read,
the task counts as having run), and a finished future's error without marking it retrieved
(``Future._exception``), so asyncio still logs it.
"""

from __future__ import annotations

import dis
import functools
import inspect
import types
from typing import Any, Optional


def unstarted_task_frame(task: Any) -> Any:
    """The frame of a task's coroutine that has not taken its first step yet, else None."""
    try:
        get_coro = getattr(task, "get_coro", None)
        coro = get_coro() if callable(get_coro) else None
        # Native coroutines only: a Task also accepts any collections.abc.Coroutine (Cython,
        # mypyc, hand-written), which has no cr_running or cr_frame.
        if not isinstance(coro, types.CoroutineType) or inspect.getcoroutinestate(coro) != inspect.CORO_CREATED:
            return None  # already running (an eager task), or not a native coroutine: not known
        return coro.cr_frame
    except Exception:  # noqa: BLE001 - not known: a cancel then counts as unconfirmed
        return None


def never_ran(frame: Any) -> bool:
    """Whether a coroutine frame never got past its entry: a task cancelled before its first step
    has the error thrown in at the RESUME that opens the body, which leaves its last instruction
    at that RESUME or (3.14+) the one after it. Body code that ran and suspended stopped at an
    await, which lies further on. Unknown on an interpreter without RESUME (before 3.11): treated
    as having run."""
    entry = _entry_offset(frame.f_code)
    return entry is not None and frame.f_lasti <= entry


@functools.lru_cache(maxsize=256)
def _entry_offset(code: Any) -> Optional[int]:
    """The offset of the instruction after the first RESUME, or None without one."""
    instructions = iter(dis.get_instructions(code))
    for instruction in instructions:
        if instruction.opname == "RESUME":
            following = next(instructions, None)
            return instruction.offset if following is None else following.offset
    return None


def peek_exception(future: Any) -> Optional[BaseException]:
    """A finished, not cancelled asyncio future's error, without marking it retrieved."""
    return getattr(future, "_exception", None)
