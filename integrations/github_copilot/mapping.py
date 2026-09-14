"""Pure mapping from GitHub Copilot hook events to Traccia span shape."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

# Events that open a span.
SESSION_START_EVENTS = frozenset({"sessionStart"})
TOOL_START_EVENTS = frozenset({"preToolUse"})
SUBAGENT_START_EVENTS = frozenset({"subagentStart"})

# Events that close a previously-opened span.
SESSION_END_EVENTS = frozenset({"sessionEnd"})
TOOL_END_EVENTS = frozenset({"postToolUse", "postToolUseFailure"})
SUBAGENT_END_EVENTS = frozenset({"subagentStop"})

SESSION_EVENT_ONLY = frozenset(
    {"preCompact", "userPromptSubmitted", "userPromptTransformed"}
)

IGNORED_EVENTS = frozenset({"agentStop", "notification", "permissionRequest"})

ALL_KNOWN_EVENTS = (
    SESSION_START_EVENTS
    | TOOL_START_EVENTS
    | SUBAGENT_START_EVENTS
    | SESSION_END_EVENTS
    | TOOL_END_EVENTS
    | SUBAGENT_END_EVENTS
    | SESSION_EVENT_ONLY
    | IGNORED_EVENTS
    | {"errorOccurred"}
)


_CONTENT_FIELDS_BY_EVENT: Dict[str, tuple] = {
    "sessionStart": ("initialPrompt",),
    "preToolUse": ("toolArgs",),
    "postToolUse": ("toolArgs",),
    "postToolUseFailure": ("toolArgs",),
    "subagentStop": ("response",),
    "userPromptSubmitted": ("prompt",),
    "userPromptTransformed": ("prompt", "transformedPrompt"),
    "subagentStart": ("agentDescription",),
    "preCompact": ("customInstructions",),
    "errorOccurred": ("errorContext",),
}

_SAFE_FIELDS_BY_EVENT: Dict[str, frozenset] = {
    "sessionStart": frozenset({"sessionId", "timestamp", "cwd", "source", "initialPrompt"}),
    "sessionEnd": frozenset({"sessionId", "timestamp", "cwd", "reason"}),
    "preToolUse": frozenset({"sessionId", "timestamp", "cwd", "toolName", "toolArgs"}),
    "postToolUse": frozenset({"sessionId", "timestamp", "cwd", "toolName", "toolArgs", "toolResult"}),
    "postToolUseFailure": frozenset({"sessionId", "timestamp", "cwd", "toolName", "toolArgs", "error"}),
    "subagentStart": frozenset({"sessionId", "timestamp", "cwd", "transcriptPath", "agentName", "agentDisplayName", "agentDescription"}),
    "subagentStop": frozenset({"sessionId", "timestamp", "cwd", "transcriptPath", "agentId", "agentType", "agentName", "agentDisplayName", "response", "stopReason"}),
    "errorOccurred": frozenset({"sessionId", "timestamp", "cwd", "error", "errorContext", "recoverable"}),
    "preCompact": frozenset({"sessionId", "timestamp", "cwd", "transcriptPath", "trigger", "customInstructions"}),
    "userPromptSubmitted": frozenset({"sessionId", "timestamp", "cwd", "prompt"}),
    "userPromptTransformed": frozenset({"sessionId", "timestamp", "cwd", "prompt", "transformedPrompt"}),
    "agentStop": frozenset({"sessionId", "timestamp", "cwd", "transcriptPath", "stopReason", "stop_hook_active"}),
    "notification": frozenset({"sessionId", "timestamp", "cwd", "title", "notification_type"}),
    "permissionRequest": frozenset({"sessionId", "timestamp", "cwd", "toolName"}),
}

_MAX_ERROR_CHARS = 200
_MAX_CONTENT_CHARS = 2000


def _length_of(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value))
    except (TypeError, ValueError):
        return len(str(value))


def _safe_text(value: Any) -> str:
    """Best-effort, size-capped string form of an arbitrary JSON value."""
    if isinstance(value, str):
        return value[:_MAX_CONTENT_CHARS]
    try:
        return json.dumps(value)[:_MAX_CONTENT_CHARS]
    except (TypeError, ValueError):
        return str(value)[:_MAX_CONTENT_CHARS]


def _capture_or_length(raw: Any, capture_content: bool) -> Any:
    """Reduce one content value for on-disk persistence."""
    if capture_content:
        from traccia.processors.redaction_processor import redact_string

        return redact_string(_safe_text(raw))
    return {"_stripped": True, "length": _length_of(raw)}


def _strip_result_object(value: Any, capture_content: bool) -> Any:
    """Sanitize Copilot's structured ``toolResult``."""
    if not isinstance(value, dict):
        return _capture_or_length(value, capture_content)
    out: Dict[str, Any] = {}
    result_type = value.get("resultType")
    if result_type is not None:
        out["resultType"] = str(result_type)[:_MAX_ERROR_CHARS]
    text = value.get("textResultForLlm")
    if text is not None:
        out["textResultForLlm"] = _capture_or_length(text, capture_content)
    return out


def strip_content_fields(
    event_name: str, payload: Dict[str, Any], capture_content: bool
) -> Dict[str, Any]:
    """Return a copy of `payload` with content-bearing fields stripped or capped."""
    
    payload = {
        key: value
        for key, value in dict(payload or {}).items()
        if key in _SAFE_FIELDS_BY_EVENT.get(event_name, frozenset({"sessionId", "timestamp", "cwd"}))
    }
    for field in _CONTENT_FIELDS_BY_EVENT.get(event_name, ()):
        if field not in payload or payload[field] is None:
            continue
        payload[field] = _capture_or_length(payload[field], capture_content)

    if event_name == "postToolUse" and payload.get("toolResult") is not None:
        payload["toolResult"] = _strip_result_object(payload["toolResult"], capture_content)

    error = payload.get("error")
    if error is not None:
        if isinstance(error, dict):
            message = error.get("message")
            err_type = error.get("type") or error.get("name")
        else:
            message = str(error)
            err_type = None
        from traccia.processors.redaction_processor import redact_string

        cleaned: Dict[str, Any] = {
            "message": redact_string((message or "")[:_MAX_ERROR_CHARS])
        }
        if err_type:
            cleaned["type"] = str(err_type)[:_MAX_ERROR_CHARS]
        payload["error"] = cleaned

    return payload


def span_name_for(event_name: str, payload: Dict[str, Any]) -> str:
    """Return the stable Traccia span name for one Copilot event."""
    if event_name in SESSION_START_EVENTS:
        return "github_copilot.session"
    if event_name in TOOL_START_EVENTS or event_name in TOOL_END_EVENTS:
        tool_name = payload.get("toolName") or "unknown"
        return f"github_copilot.tool.{tool_name}"
    if event_name in SUBAGENT_START_EVENTS or event_name in SUBAGENT_END_EVENTS:
        agent_name = payload.get("agentName") or payload.get("agentType") or "unknown"
        return f"github_copilot.subagent.{agent_name}"
    return f"github_copilot.{event_name}"


def _content_value(value: Any) -> Optional[str]:
    """Read back a field already processed by strip_content_fields. Returns
    None for a stripped (length-only) field."""
    if isinstance(value, dict) and value.get("_stripped"):
        return None
    if value is None:
        return None
    return value if isinstance(value, str) else _safe_text(value)


def _content_length(value: Any) -> Optional[int]:
    if isinstance(value, dict) and value.get("_stripped"):
        return value.get("length")
    return None


def start_attributes(event_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build attributes for a newly opened span.

    Args:
        event_name: Copilot hook event name.
        payload: Sanitized event payload.
    Returns:
        Attributes to apply to the span.
    """
    attrs: Dict[str, Any] = {
        "github_copilot.event": event_name,
        "agent.span.type": "github_copilot",
    }
    session_id = payload.get("sessionId")
    if session_id:
        attrs["session.id"] = session_id
        attrs["github_copilot.session.id"] = session_id
    cwd = payload.get("cwd")
    if cwd:
        attrs["github_copilot.cwd"] = cwd

    if event_name in SESSION_START_EVENTS:
        attrs["agent.type"] = "github_copilot"
        attrs["agent.name"] = "github-copilot"
        attrs["gen_ai.system"] = "github_copilot"
        source = payload.get("source")
        if source:
            attrs["github_copilot.session.source"] = source
        prompt = payload.get("initialPrompt")
        preview = _content_value(prompt)
        length = _content_length(prompt)
        if preview is not None:
            attrs["github_copilot.prompt.preview"] = preview
        elif length is not None:
            attrs["github_copilot.prompt.length"] = length

    elif event_name in TOOL_START_EVENTS:
        tool_name = payload.get("toolName")
        if tool_name:
            attrs["agent.tool.name"] = tool_name
        tool_args = payload.get("toolArgs")
        preview = _content_value(tool_args)
        length = _content_length(tool_args)
        if preview is not None:
            attrs["agent.tool.input"] = preview
        elif length is not None:
            attrs["agent.tool.input.length"] = length

    elif event_name in SUBAGENT_START_EVENTS:
        agent_name = payload.get("agentName") or payload.get("agentType")
        if agent_name:
            attrs["agent.name"] = agent_name
        display_name = payload.get("agentDisplayName")
        if display_name:
            attrs["agent.display_name"] = display_name
        description = payload.get("agentDescription")
        if description:
            preview = _content_value(description)
            length = _content_length(description)
            if preview is not None:
                attrs["agent.description"] = preview
            elif length is not None:
                attrs["agent.description.length"] = length
        attrs["agent.handoff.from"] = "github-copilot"

    return attrs


def end_attributes(event_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attributes/status to apply when a span is closed for this event.

    Args:
        event_name: Copilot hook event name.
        payload: Sanitized event payload.
    Returns:
        A dict containing attributes, error status, and an optional message.
    """
    attrs: Dict[str, Any] = {}
    is_error = False
    error_message: Optional[str] = None

    if event_name == "postToolUse":
        result = payload.get("toolResult")
        if isinstance(result, dict) and not result.get("_stripped"):
            result_type = result.get("resultType")
            if result_type:
                attrs["agent.tool.result_type"] = result_type
            text = result.get("textResultForLlm")
            preview = _content_value(text)
            length = _content_length(text)
            if preview is not None:
                attrs["agent.tool.output"] = _safe_text(preview)
            elif length is not None:
                attrs["agent.tool.output.length"] = length
        else:
            preview = _content_value(result)
            length = _content_length(result)
            if preview is not None:
                attrs["agent.tool.output"] = _safe_text(preview)
            elif length is not None:
                attrs["agent.tool.output.length"] = length

    elif event_name == "postToolUseFailure":
        is_error = True
        error = payload.get("error")
        if isinstance(error, dict):
            error_message = error.get("message")
            if error_message:
                attrs["error.message"] = error_message
            err_type = error.get("type")
            if err_type:
                attrs["error.type"] = err_type

    elif event_name == "subagentStop":
        response = payload.get("response")
        preview = _content_value(response)
        length = _content_length(response)
        if preview is not None:
            attrs["agent.response"] = preview
        elif length is not None:
            attrs["agent.response.length"] = length

    return {"attributes": attrs, "is_error": is_error, "error_message": error_message}
