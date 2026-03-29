"""Helper APIs for explicit guardrail annotation in Traccia traces.

Provides a context-manager for wrapping guardrail checks so they emit
properly typed and attributed spans without requiring the user to remember
the exact attribute keys.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from traccia.guardrails.constants import (
    ATTR_GUARDRAIL_CATEGORY,
    ATTR_GUARDRAIL_ENFORCEMENT_MODE,
    ATTR_GUARDRAIL_EVIDENCE_TYPE,
    ATTR_GUARDRAIL_NAME,
    ATTR_GUARDRAIL_SOURCE_SDK,
    ATTR_GUARDRAIL_SUPPRESS_MISSING,
    ATTR_GUARDRAIL_TRIGGERED,
)

logger = logging.getLogger(__name__)

_GUARDRAIL_REQUIRED_ATTRS = {ATTR_GUARDRAIL_NAME, ATTR_GUARDRAIL_CATEGORY, ATTR_GUARDRAIL_TRIGGERED}


def _build_guardrail_attributes(
    name: str,
    category: str = "unknown",
    triggered: Optional[bool] = None,
    enforcement_mode: str = "unknown",
    policy_id: Optional[str] = None,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "span.type": "guardrail",
        ATTR_GUARDRAIL_NAME: name,
        ATTR_GUARDRAIL_CATEGORY: category,
        ATTR_GUARDRAIL_ENFORCEMENT_MODE: enforcement_mode,
        ATTR_GUARDRAIL_SOURCE_SDK: "manual_observe",
        ATTR_GUARDRAIL_EVIDENCE_TYPE: "span_attribute",
    }
    if triggered is not None:
        attrs[ATTR_GUARDRAIL_TRIGGERED] = triggered
    if policy_id is not None:
        from traccia.guardrails.constants import ATTR_GUARDRAIL_POLICY_ID
        attrs[ATTR_GUARDRAIL_POLICY_ID] = policy_id
    return attrs


@contextmanager
def guardrail_span(
    name: str,
    *,
    category: str = "unknown",
    enforcement_mode: str = "unknown",
    policy_id: Optional[str] = None,
    suppress_missing: Optional[list] = None,
):
    """Context manager that creates a guardrail-typed Traccia span.

    The caller should set ``triggered`` on the returned span before exiting::

        with guardrail_span("pii_check", category="pii") as span:
            result = run_pii_check(text)
            span.set_attribute("guardrail.triggered", result.found_pii)

    Parameters
    ----------
    suppress_missing:
        Optional list of ``GuardrailCategory`` string values to suppress from
        the missing-guardrail report for this run.  Useful for batch pipelines
        or internal-only agents that don't need certain guardrails::

            with guardrail_span(
                "batch_root",
                category="unknown",
                suppress_missing=["prompt_injection", "input_validation"],
            ) as span:
                ...
    """
    import traccia

    attrs = _build_guardrail_attributes(
        name=name,
        category=category,
        enforcement_mode=enforcement_mode,
        policy_id=policy_id,
    )
    if suppress_missing:
        attrs[ATTR_GUARDRAIL_SUPPRESS_MISSING] = list(suppress_missing)
    with traccia.span(f"guardrail.{name}", attributes=attrs) as span:
        yield span


def validate_guardrail_attributes(
    attrs: Dict[str, Any],
    *,
    require_triggered: bool = True,
) -> list[str]:
    """Return list of warnings for incomplete guardrail span attributes.

    Used by the observe decorator when ``as_type="guardrail"`` to surface
    missing recommended fields so the user gets actionable feedback.

    Parameters
    ----------
    require_triggered:
        If False, do not warn about missing ``guardrail.triggered``. Use for
        ``@observe(as_type="guardrail")`` when the function returns a ``bool``
        and the decorator sets ``guardrail.triggered`` automatically.
    """
    warnings: list[str] = []
    keys = _GUARDRAIL_REQUIRED_ATTRS
    if not require_triggered:
        keys = keys - {ATTR_GUARDRAIL_TRIGGERED}
    for key in keys:
        if key not in attrs:
            warnings.append(f"Missing recommended guardrail attribute: {key}")
    return warnings
