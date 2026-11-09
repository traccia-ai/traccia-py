"""Reset per-process state in a child created by os.fork()."""

from __future__ import annotations

import os
import weakref
from typing import Callable


def reinit_after_fork(reinit: Callable[[], None]) -> None:
    """
    Call reinit() in the child after os.fork().

    Only the thread that forked runs in the child: other threads are gone and
    any lock they held stays locked. reinit() recreates locks and threads.
    A bound method is held by weak reference, so its object can still be
    garbage collected; a module-level function is registered once at import.
    """
    if not hasattr(os, "register_at_fork"):
        return
    if not hasattr(reinit, "__self__"):
        os.register_at_fork(after_in_child=reinit)
        return
    weak_method = weakref.WeakMethod(reinit)

    def _after_in_child() -> None:
        method = weak_method()
        if method is not None:
            method()

    os.register_at_fork(after_in_child=_after_in_child)
