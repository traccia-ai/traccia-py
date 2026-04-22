"""Span processor that annotates spans with cost based on token usage and pricing."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from traccia.processors.cost_engine import compute_cost, match_pricing_model_key
from traccia.tracer.provider import SpanProcessor

logger = logging.getLogger(__name__)

# Staleness thresholds (days)
_WARN_AFTER_DAYS = 30
_INFO_AFTER_DAYS = 7

# Guard so the staleness log fires at most once per process.
_staleness_warned = threading.Event()


def _check_and_log_staleness(generated_at: str, age_days: Optional[float]) -> None:
    """Emit a one-time staleness warning if the pricing snapshot is old."""
    if _staleness_warned.is_set():
        return
    if age_days is None:
        return
    if age_days > _WARN_AFTER_DAYS:
        _staleness_warned.set()
        logger.warning(
            "Traccia SDK pricing snapshot is %.0f days old (from %s). "
            "Run 'traccia pricing refresh' to download the latest, or view costs "
            "on the Traccia platform for authoritative pricing.",
            age_days,
            generated_at,
        )
    elif age_days > _INFO_AFTER_DAYS:
        _staleness_warned.set()
        logger.info(
            "Traccia SDK pricing snapshot is %.0f days old (from %s). "
            "Run 'traccia pricing refresh' for the latest pricing.",
            age_days,
            generated_at,
        )


class CostAnnotatingProcessor(SpanProcessor):
    """
    Adds ``llm.cost.usd`` to spans when token usage and model info are available.

    Span attributes written:
      - llm.cost.usd                 — estimated cost in USD
      - llm.usage.source             — where token counts came from (was llm.cost.source)
      - llm.pricing.source           — which pricing layer was used
      - llm.pricing.model_key        — the pricing table key matched
      - llm.pricing.generated_at     — ISO timestamp of the pricing snapshot
      - llm.pricing.age_days         — integer age of the snapshot in days
      - llm.pricing.snapshot_version — source identifier of the snapshot
    """

    def __init__(
        self,
        pricing_table: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        pricing_source: str = "bundled",
        pricing_generated_at: str = "unknown",
    ) -> None:
        self.pricing_table = pricing_table or {}
        self.pricing_source = pricing_source
        self.pricing_generated_at = pricing_generated_at

        # Compute and cache age once at construction; refreshed on update_pricing_table.
        self._age_days: Optional[float] = self._compute_age()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_age(self) -> Optional[float]:
        from traccia.pricing_config import snapshot_age_days
        return snapshot_age_days(self.pricing_generated_at)

    def _age_days_int(self) -> Optional[int]:
        if self._age_days is None:
            return None
        return int(self._age_days)

    # ------------------------------------------------------------------
    # SpanProcessor interface
    # ------------------------------------------------------------------

    def on_end(self, span) -> None:
        if "llm.cost.usd" in (span.attributes or {}):
            return

        model = span.attributes.get("llm.model")
        prompt = span.attributes.get("llm.usage.prompt_tokens")
        completion = span.attributes.get("llm.usage.completion_tokens")
        # Anthropic-style aliases
        if prompt is None:
            prompt = span.attributes.get("llm.usage.input_tokens")
        if completion is None:
            completion = span.attributes.get("llm.usage.output_tokens")

        if not model or prompt is None or completion is None:
            return

        cost = compute_cost(
            model,
            int(prompt),
            int(completion),
            pricing_table=self.pricing_table,
        )
        if cost is None:
            return

        # Lazily check staleness on first successful cost computation.
        _check_and_log_staleness(self.pricing_generated_at, self._age_days)

        span.set_attribute("llm.cost.usd", cost)

        # llm.usage.source — where token counts originated (renamed from llm.cost.source).
        usage_source = span.attributes.get("llm.usage.source") or span.attributes.get("llm.cost.source", "unknown")
        span.set_attribute("llm.usage.source", usage_source)
        # Keep the old attribute name as an alias for one major version to avoid breakage.
        span.set_attribute("llm.cost.source", usage_source)

        span.set_attribute("llm.pricing.source", self.pricing_source)

        key = match_pricing_model_key(model, self.pricing_table)
        if key:
            span.set_attribute("llm.pricing.model_key", key)

        # Snapshot provenance — lets the platform know how old the client's rates were.
        span.set_attribute("llm.pricing.generated_at", self.pricing_generated_at)
        age_int = self._age_days_int()
        if age_int is not None:
            span.set_attribute("llm.pricing.age_days", age_int)
        # snapshot_version uses generated_at as the stable identifier (no commit SHA needed
        # when coming from the bundled snapshot; the platform normalises this on ingest).
        span.set_attribute("llm.pricing.snapshot_version", self.pricing_generated_at)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout: Optional[float] = None) -> None:
        return None

    def update_pricing_table(
        self,
        pricing_table: Dict[str, Dict[str, Any]],
        pricing_source: Optional[str] = None,
        pricing_generated_at: Optional[str] = None,
    ) -> None:
        self.pricing_table = pricing_table
        if pricing_source:
            self.pricing_source = pricing_source
        if pricing_generated_at:
            self.pricing_generated_at = pricing_generated_at
            self._age_days = self._compute_age()
