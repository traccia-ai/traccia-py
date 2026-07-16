"""Loaded prompt object and span attribute helpers."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Mapping, Optional, Union

from traccia.prompts.compile import CompileError, compile_body

logger = logging.getLogger("traccia.prompts")

ATTR_PROMPT_NAME = "traccia.prompt.name"
ATTR_PROMPT_VERSION = "traccia.prompt.version"
ATTR_PROMPT_VERSION_ID = "traccia.prompt.version_id"
ATTR_PROMPT_LABEL = "traccia.prompt.label"
ATTR_PROMPT_IS_FALLBACK = "traccia.prompt.is_fallback"


class LoadedPrompt:
    """Runtime prompt resolved from Traccia or an explicit fallback."""

    def __init__(
        self,
        *,
        name: str,
        prompt_type: str,
        body: Mapping[str, Any],
        version: Optional[int] = None,
        version_id: Optional[str] = None,
        label: Optional[str] = None,
        variables: Optional[Any] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        tools: Optional[Any] = None,
        is_fallback: bool = False,
        is_stale: bool = False,
    ) -> None:
        self.name = name
        self.type = prompt_type
        self.body = dict(body or {})
        self.version = version
        self.version_id = version_id
        self.label = label
        self.variables = variables or []
        self.config = dict(model_config or {})
        self.tools = tools
        self.is_fallback = is_fallback
        self.is_stale = is_stale

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        is_fallback: bool = False,
        is_stale: bool = False,
    ) -> "LoadedPrompt":
        return cls(
            name=str(payload.get("name") or ""),
            prompt_type=str(payload.get("type") or "text"),
            body=payload.get("body") or {},
            version=payload.get("version"),
            version_id=str(payload["version_id"]) if payload.get("version_id") is not None else None,
            label=payload.get("label"),
            variables=payload.get("variables"),
            model_config=payload.get("model_config") or payload.get("config"),
            tools=payload.get("tools"),
            is_fallback=is_fallback,
            is_stale=is_stale,
        )

    @classmethod
    def from_fallback(
        cls,
        name: str,
        fallback: Mapping[str, Any],
        *,
        label: Optional[str] = None,
    ) -> "LoadedPrompt":
        fb_type = str(fallback.get("type") or ("chat" if "messages" in fallback else "text"))
        if fb_type == "chat":
            body = {"messages": list(fallback.get("messages") or [])}
        else:
            body = {"text": str(fallback.get("text") or "")}
        return cls(
            name=name,
            prompt_type=fb_type,
            body=body,
            version=fallback.get("version"),
            version_id=fallback.get("version_id"),
            label=label or fallback.get("label"),
            model_config=fallback.get("model_config") or fallback.get("config"),
            tools=fallback.get("tools"),
            is_fallback=True,
            is_stale=False,
        )

    @property
    def text(self) -> Optional[str]:
        if self.type == "text":
            return str(self.body.get("text") or "")
        return None

    @property
    def messages(self) -> Optional[List[Dict[str, Any]]]:
        if self.type == "chat":
            return list(self.body.get("messages") or [])
        return None

    def span_attributes(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {ATTR_PROMPT_NAME: self.name}
        if self.version is not None:
            attrs[ATTR_PROMPT_VERSION] = str(self.version)
        if self.version_id:
            attrs[ATTR_PROMPT_VERSION_ID] = self.version_id
        if self.label:
            attrs[ATTR_PROMPT_LABEL] = self.label
        if self.is_fallback:
            attrs[ATTR_PROMPT_IS_FALLBACK] = True
        return attrs

    def apply_span_attributes(self, span: Any = None) -> None:
        """Attach traccia.prompt.* attrs to the current (or provided) span."""
        try:
            if span is None:
                from traccia.context import get_current_span

                span = get_current_span()
            if not span or not hasattr(span, "set_attribute"):
                return
            for key, value in self.span_attributes().items():
                span.set_attribute(key, value)
        except Exception as exc:
            logger.debug("Could not apply prompt span attributes: %s", exc)

    def compile(
        self,
        variables: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Substitute {{vars}}. Missing required → CompileError.
        Unknown extras → warning. Attaches traccia.prompt.* span attrs on success.
        """
        merged: Dict[str, Any] = {}
        if variables:
            merged.update(dict(variables))
        merged.update(kwargs)

        declared = None
        if isinstance(self.variables, list) and self.variables and isinstance(self.variables[0], dict):
            declared = self.variables

        compiled, extras = compile_body(
            self.type,
            self.body,
            merged,
            declared=declared,
        )
        if extras:
            warnings.warn(
                f"Unknown prompt variables ignored: {', '.join(extras)}",
                UserWarning,
                stacklevel=2,
            )

        self.apply_span_attributes()

        if self.type == "text":
            return str(compiled.get("text") or "")
        return list(compiled.get("messages") or [])


__all__ = [
    "LoadedPrompt",
    "CompileError",
    "ATTR_PROMPT_NAME",
    "ATTR_PROMPT_VERSION",
    "ATTR_PROMPT_VERSION_ID",
    "ATTR_PROMPT_LABEL",
    "ATTR_PROMPT_IS_FALLBACK",
]
