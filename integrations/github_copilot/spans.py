"""Build Traccia spans from a GitHub Copilot session's buffered hook events."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from traccia.integrations.github_copilot import mapping
from traccia.processors.redaction_processor import redact_attributes, redact_string
from traccia.tracer.span import SpanStatus


def _vcs_attributes(cwd: Optional[str]) -> Dict[str, str]:
    """Best-effort repo/branch/commit for a session's working directory."""
    
    if not cwd:
        return {}
    import re
    import subprocess

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", cwd, *args],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        val = (out.stdout or "").strip()
        return val or None

    attrs: Dict[str, str] = {}
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        attrs["vcs.branch.name"] = branch
    sha = _git("rev-parse", "HEAD")
    if sha:
        attrs["vcs.commit.sha"] = sha
    remote = _git("config", "--get", "remote.origin.url")
    if remote:
        attrs["vcs.repository.url"] = re.sub(r"//[^/@]*@", "//", remote)
    return attrs


def _epoch_to_ns(value: Any) -> Optional[int]:
    """Best-effort conversion of an epoch timestamp of unknown unit to ns."""
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 1e17:
        return int(value)  # already nanoseconds
    if value > 1e14:
        return int(value * 1_000)  # microseconds
    if value > 1e11:
        return int(value * 1_000_000)  # milliseconds
    return int(value * 1_000_000_000)  # seconds


def _event_time_ns(payload: Dict[str, Any], received_at: float) -> int:
    ns = _epoch_to_ns(payload.get("timestamp"))
    if ns is not None:
        return ns
    return int(received_at * 1_000_000_000)


def _set_attrs(span: Any, attrs: Dict[str, Any]) -> None:
    for key, value in redact_attributes(attrs).items():
        span.set_attribute(key, value)


def _clamped_end(span: Any, end_ns: int) -> int:
    """Never let a span end before it started."""
    start = getattr(span, "start_time_ns", None)
    if isinstance(start, int) and end_ns < start:
        return start
    return end_ns


def _discard(stack: List[Any], span: Any) -> None:
    for i, existing in enumerate(stack):
        if existing is span:
            del stack[i]
            return


def build_trace(tracer: Any, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Replay one session's buffered events onto `tracer`.

    Args:
        tracer: Traccia tracer used to materialize the replayed spans.
        events: Buffered hook records in JSONL record shape.
    Returns:
        A summary dict with span/error counts, or None for no events.
    """
    if not events:
        return None

    events = sorted(events, key=lambda e: e.get("received_at", 0))

    session_span: Optional[Any] = None
    tool_queues: Dict[str, Deque[Any]] = defaultdict(deque)
    subagent_spans: Dict[str, Deque[Any]] = defaultdict(deque)
    open_spans: List[Any] = [] 
    summary = {"tool_spans": 0, "subagent_spans": 0, "errors": 0}
    last_event_ns: Optional[int] = None

    def pop_subagent(payload: Dict[str, Any]) -> Optional[Any]:
        """Pair a stop event by name, with a safe legacy single-span fallback."""
        key = payload.get("agentName") or payload.get("agentType")
        queue = subagent_spans.get(key) if key else None
        if queue:
            return queue.popleft()
        if payload.get("agentId"):
            candidates = [q for q in subagent_spans.values() if q]
            if len(candidates) == 1:
                return candidates[0].popleft()
        return None

    def ensure_session_span(payload: Dict[str, Any], start_ns: int) -> Any:
        nonlocal session_span
        if session_span is not None:
            return session_span
        
        attrs = mapping.start_attributes("sessionStart", payload)
        attrs["github_copilot.session.source"] = "recovered_missing_session_start"
        attrs.update(_vcs_attributes(payload.get("cwd")))
        session_span = tracer.start_span(
            mapping.span_name_for("sessionStart", payload),
            start_time=start_ns,
        )
        _set_attrs(session_span, attrs)
        open_spans.append(session_span)
        return session_span

    for record in events:
        event_name = record.get("event")
        payload = record.get("payload") or {}
        received_at = record.get("received_at", 0)
        start_ns = _event_time_ns(payload, received_at)
        last_event_ns = start_ns if last_event_ns is None else max(last_event_ns, start_ns)

        if event_name in mapping.SESSION_START_EVENTS:
            if session_span is not None:
                continue  # duplicate sessionStart in the log; ignore
            session_span = tracer.start_span(
                mapping.span_name_for(event_name, payload), start_time=start_ns
            )
            _set_attrs(session_span, mapping.start_attributes(event_name, payload))
            _set_attrs(session_span, _vcs_attributes(payload.get("cwd")))
            open_spans.append(session_span)

        elif event_name in mapping.TOOL_START_EVENTS:
            parent = ensure_session_span(payload, start_ns)
            tool_name = payload.get("toolName") or "unknown"
            span = tracer.start_span(
                mapping.span_name_for(event_name, payload),
                parent=parent,
                start_time=start_ns,
            )
            _set_attrs(span, mapping.start_attributes(event_name, payload))
            tool_queues[tool_name].append(span)
            open_spans.append(span)
            summary["tool_spans"] += 1

        elif event_name in mapping.TOOL_END_EVENTS:
            tool_name = payload.get("toolName") or "unknown"
            queue = tool_queues[tool_name]
            span = queue.popleft() if queue else None
            if span is None:
                parent = ensure_session_span(payload, start_ns)
                span = tracer.start_span(
                    mapping.span_name_for(event_name, payload),
                    parent=parent,
                    start_time=start_ns,
                )
                _set_attrs(span, mapping.start_attributes("preToolUse", payload))
            else:
                _discard(open_spans, span)
            result = mapping.end_attributes(event_name, payload)
            _set_attrs(span, result["attributes"])
            if result["is_error"]:
                span.set_status(SpanStatus.ERROR, result["error_message"])
                summary["errors"] += 1
            else:
                span.set_status(SpanStatus.OK)
            span.end(end_time=_clamped_end(span, start_ns))

        elif event_name in mapping.SUBAGENT_START_EVENTS:
            parent = ensure_session_span(payload, start_ns)
            span = tracer.start_span(
                mapping.span_name_for(event_name, payload),
                parent=parent,
                start_time=start_ns,
            )
            _set_attrs(span, mapping.start_attributes(event_name, payload))
            key = payload.get("agentName") or payload.get("agentType") or "unknown"
            subagent_spans[key].append(span)
            open_spans.append(span)
            summary["subagent_spans"] += 1

        elif event_name in mapping.SUBAGENT_END_EVENTS:
            span = pop_subagent(payload)
            if span is None:
                continue  # no matching subagentStart captured in this session's log
            _discard(open_spans, span)
            result = mapping.end_attributes(event_name, payload)
            _set_attrs(span, result["attributes"])
            span.set_status(SpanStatus.OK)
            span.end(end_time=_clamped_end(span, start_ns))

        elif event_name == "errorOccurred":
            summary["errors"] += 1
            target = open_spans[-1] if open_spans else session_span
            if target is not None:
                error = payload.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else str(error)
                err_type = error.get("type") if isinstance(error, dict) else None
                event_attrs = {"error.message": redact_string((message or "")[:200])}
                if err_type:
                    event_attrs["error.type"] = str(err_type)[:200]
                ctx = payload.get("errorContext")
                if isinstance(ctx, dict) and ctx.get("_stripped"):
                    event_attrs["error.context.length"] = ctx.get("length")
                elif isinstance(ctx, str) and ctx:
                    event_attrs["error.context"] = redact_string(ctx[:200])
                target.add_event(
                    "github_copilot.error",
                    event_attrs,
                    timestamp_ns=start_ns,
                )

        elif event_name in mapping.SESSION_EVENT_ONLY:
            if session_span is not None:
                session_span.add_event(f"github_copilot.{event_name}", {}, timestamp_ns=start_ns)

        elif event_name in mapping.SESSION_END_EVENTS:
            if session_span is None:
                continue
            reason = payload.get("reason")
            if reason:
                session_span.set_attribute("github_copilot.session.end_reason", reason)
            if reason == "error":
                session_span.set_status(SpanStatus.ERROR, "session ended with an error")
            else:
                session_span.set_status(SpanStatus.OK)
            session_span.end(end_time=_clamped_end(session_span, start_ns))
            _discard(open_spans, session_span)

    for span in reversed(open_spans):
        try:
            if last_event_ns is not None:
                span.end(end_time=_clamped_end(span, last_event_ns))
            else:
                span.end()
        except Exception:
            pass

    return summary
