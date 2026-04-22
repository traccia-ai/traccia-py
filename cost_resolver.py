"""Process-level singleton that holds the active pricing table.

Both the span processor (CostAnnotatingProcessor) and the OpenAI Agents
integration read from this resolver, so a pricing_override supplied via
start_tracing() is reflected identically in span attributes *and* the
gen_ai.client.operation.cost metric histogram.

Usage::

    from traccia.cost_resolver import get_resolver, CostResolver

    resolver = get_resolver()
    cost = resolver.compute("gpt-4o", prompt_tokens=100, completion_tokens=50)
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

from traccia.processors.cost_engine import compute_cost, match_pricing_model_key

_lock = threading.Lock()
_resolver: Optional["CostResolver"] = None


class CostResolver:
    """
    Holds one active pricing table and exposes cost computation.

    Thread-safe: ``update()`` and ``compute()`` are protected by an internal lock
    so a background pricing refresh never races with an active span end.
    """

    def __init__(
        self,
        pricing_table: Dict[str, Dict[str, Any]],
        source: str = "bundled",
        generated_at: str = "unknown",
    ) -> None:
        self._lock = threading.Lock()
        self._table = pricing_table
        self._source = source
        self._generated_at = generated_at

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def pricing_table(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self._table

    @property
    def source(self) -> str:
        with self._lock:
            return self._source

    @property
    def generated_at(self) -> str:
        with self._lock:
            return self._generated_at

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(
        self,
        pricing_table: Dict[str, Dict[str, Any]],
        source: Optional[str] = None,
        generated_at: Optional[str] = None,
    ) -> None:
        """Replace the active pricing table (e.g. after a background refresh)."""
        with self._lock:
            self._table = pricing_table
            if source is not None:
                self._source = source
            if generated_at is not None:
                self._generated_at = generated_at

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def compute(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Optional[float]:
        """
        Return the estimated cost in USD, or None if the model has no pricing entry.
        Thread-safe: takes a snapshot of the table under the lock before computing.
        """
        with self._lock:
            table = self._table
        return compute_cost(model, prompt_tokens, completion_tokens, pricing_table=table)

    def match_key(self, model: str) -> Optional[str]:
        """Return the pricing table key that matches *model*, or None."""
        with self._lock:
            table = self._table
        return match_pricing_model_key(model, table)

    def snapshot(self) -> Tuple[Dict[str, Dict[str, Any]], str, str]:
        """Return (pricing_table, source, generated_at) as an atomic snapshot."""
        with self._lock:
            return self._table, self._source, self._generated_at


# ---------------------------------------------------------------------------
# Module-level singleton — initialised lazily on first access
# ---------------------------------------------------------------------------

def _make_default() -> "CostResolver":
    from traccia.pricing_config import load_pricing_with_source
    table, source, generated_at = load_pricing_with_source()
    return CostResolver(table, source, generated_at)


def get_resolver() -> "CostResolver":
    """Return the process-level CostResolver, creating it on first call."""
    global _resolver
    if _resolver is None:
        with _lock:
            if _resolver is None:
                _resolver = _make_default()
    return _resolver


def set_resolver(resolver: "CostResolver") -> None:
    """Replace the process-level resolver (called by start_tracing)."""
    global _resolver
    with _lock:
        _resolver = resolver
