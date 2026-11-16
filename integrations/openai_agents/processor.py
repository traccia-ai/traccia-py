"""Traccia processor for OpenAI Agents SDK tracing."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    try:
        from agents.tracing import Span, Trace
        from agents.tracing.span_data import (
            AgentSpanData,
            FunctionSpanData,
            GenerationSpanData,
            HandoffSpanData,
            GuardrailSpanData,
            ResponseSpanData,
            CustomSpanData,
        )
    except ImportError:
        pass


def _iso_to_ns(value: Any) -> Optional[int]:
    """Agents SDK ISO-8601 timestamp (``started_at`` / ``ended_at``) -> epoch ns, or None."""
    if not value or not isinstance(value, str):
        return None
    try:
        text = value.strip()
        if text.endswith(("Z", "z")):  # fromisoformat accepts "Z" only from Python 3.11
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = dt - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000
    except Exception:
        return None


class TracciaAgentsTracingProcessor:
    """
    Traccia processor for OpenAI Agents SDK.

    Implements the Agents SDK TracingProcessor interface to capture agent
    runs, tool calls, handoffs, and generations into Traccia spans.

    The callbacks carry the run's hierarchy (``trace_id``, ``parent_id``) and timing
    (``started_at``, ``ended_at``). Each Agents trace becomes one Traccia trace under a
    root span (nested in the caller's span when one is active), and each Agents span is
    parented explicitly on the Traccia span of its Agents parent, so the tree does not
    depend on which Traccia span happens to be current when a callback runs.
    """

    def __init__(self):
        """Initialize the processor."""
        # Agents trace_id -> {"span": root Traccia span, "last_end_ns": latest child end}
        self._trace_map: dict[str, dict[str, Any]] = {}
        # Agents span_id -> {"span": Traccia span, "trace_id": Agents trace_id}
        self._span_map: dict[str, dict[str, Any]] = {}
        self._tracer = None

    def _get_tracer(self):
        """Get or create the Traccia tracer."""
        if self._tracer is None:
            import traccia
            self._tracer = traccia.get_tracer("openai.agents")
        return self._tracer

    def _resolve_parent(self, span: Any) -> Any:
        """Traccia span for an Agents span's parent: its parent span, else its trace's root."""
        parent_entry = self._span_map.get(getattr(span, "parent_id", None))
        if parent_entry is not None:
            return parent_entry["span"]
        trace_entry = self._trace_map.get(getattr(span, "trace_id", None))
        if trace_entry is not None:
            return trace_entry["span"]
        # Unknown trace (processor registered mid-run): the caller's current span, if any.
        return None

    def on_trace_start(self, trace: Trace) -> None:
        """Called when an Agents trace starts."""
        try:
            trace_id = getattr(trace, "trace_id", None)
            attributes = {"agents.trace_id": trace_id, "agent.span.type": "trace"}
            group_id = getattr(trace, "group_id", None)
            if group_id:
                attributes["agents.group_id"] = str(group_id)
            # One Traccia root span per Agents trace, so the run's spans form one trace
            # (inside the caller's span when there is one).
            root = self._get_tracer().start_span(
                getattr(trace, "name", None) or "agents.trace",
                attributes=attributes,
                start_time_ns=_iso_to_ns(getattr(trace, "started_at", None)),
            )
            self._trace_map[trace_id] = {"span": root, "last_end_ns": None}
        except Exception:
            # Don't break agent execution on tracing errors
            pass

    def on_trace_end(self, trace: Trace) -> None:
        """Called when an Agents trace ends."""
        try:
            entry = self._trace_map.pop(getattr(trace, "trace_id", None), None)
            if entry and entry.get("span"):
                # Never end before a child whose framework end time is past our clock.
                end_ns = max(time.time_ns(), entry.get("last_end_ns") or 0)
                entry["span"].end(end_time_ns=end_ns)
        except Exception:
            pass

    def on_span_start(self, span: Span[Any]) -> None:
        """Called when an Agents span starts."""
        try:
            tracer = self._get_tracer()
            span_data = span.span_data
            span_name = self._get_span_name(span_data)
            attributes = self._extract_attributes(span_data)
            trace_id = getattr(span, "trace_id", None)
            if trace_id:
                attributes["agents.trace_id"] = trace_id
            attributes["agents.span_id"] = span.span_id
            parent_id = getattr(span, "parent_id", None)
            if parent_id:
                attributes["agents.parent_id"] = parent_id

            parent = self._resolve_parent(span)
            framework_start_ns = _iso_to_ns(getattr(span, "started_at", None))
            start_ns = framework_start_ns
            parent_start_ns = getattr(parent, "start_time_ns", None)
            if start_ns is not None and isinstance(parent_start_ns, int):
                # ISO timestamps are truncated to microseconds; keep children inside parents.
                start_ns = max(start_ns, parent_start_ns)

            traccia_span = tracer.start_span(
                span_name, attributes=attributes, parent=parent, start_time_ns=start_ns
            )
            self._span_map[span.span_id] = {
                "span": traccia_span,
                "trace_id": trace_id,
                "framework_start_ns": framework_start_ns,
            }
        except Exception:
            # Don't break agent execution
            pass

    def on_span_end(self, span: Span[Any]) -> None:
        """Called when an Agents span ends."""
        traccia_span = None
        end_ns = None
        try:
            entry = self._span_map.pop(span.span_id, None)
            if not entry:
                return
            traccia_span = entry["span"]

            # The framework's end time, never before the span's start.
            framework_end_ns = _iso_to_ns(getattr(span, "ended_at", None))
            end_ns = max(framework_end_ns or time.time_ns(), traccia_span.start_time_ns)
            trace_entry = self._trace_map.get(entry.get("trace_id"))
            if trace_entry is not None:
                trace_entry["last_end_ns"] = max(trace_entry.get("last_end_ns") or 0, end_ns)
            # Metrics report the framework's own duration when it gave both times.
            framework_start_ns = entry.get("framework_start_ns")
            if framework_start_ns is not None and framework_end_ns is not None:
                duration = max(framework_end_ns - framework_start_ns, 0) / 1e9
            else:
                duration = (end_ns - traccia_span.start_time_ns) / 1e9

            # Update attributes with final data
            span_data = span.span_data
            self._update_span_attributes(traccia_span, span_data)
            
            # Record error if present
            error = getattr(span, "error", None)
            if error:
                from traccia.tracer.span import SpanStatus
                error_msg = str(error.get("message", "Unknown error") if isinstance(error, dict) else error)
                traccia_span.set_status(SpanStatus.ERROR, error_msg)
            
            span_type = getattr(span_data, "type", None)
            
            # Record agent metrics if this is an agent span
            if span_type == "agent":
                agent_name = getattr(span_data, "name", None)
                self._record_agent_metrics(
                    agent_id=None,  # OpenAI Agents doesn't expose agent ID
                    agent_name=agent_name,
                    execution_time=duration,
                    is_run=True  # Agent spans represent runs
                )

            # Record token/cost metrics if this is a generation span
            if span_type == "generation":
                self._record_generation_metrics(span_data, duration)

            # End the span at the framework's end time
            traccia_span.end(end_time_ns=end_ns)
        except Exception:
            # Ensure span ends even if there's an error
            try:
                if traccia_span:
                    traccia_span.end(end_time_ns=end_ns)
            except Exception:
                pass
    
    def _record_agent_metrics(
        self,
        agent_id: Any,
        agent_name: Any,
        execution_time: float,
        is_run: bool
    ):
        """Record agent metrics if metrics are enabled."""
        try:
            from traccia.metrics.recorder import get_metrics_recorder
            recorder = get_metrics_recorder()
            if not recorder:
                return
            
            # Build attributes — prefer run-scoped identity from runtime_config so metrics
            # are attributed to the correct logical agent (not the sub-agent role name).
            attributes = {}
            try:
                from traccia import runtime_config as _rc
                _aid = _rc.get_agent_id()
                _aname = _rc.get_agent_name()
                _env = _rc.get_env()
                if _aid:
                    attributes["agent.id"] = _aid
                    attributes["agent_id"] = _aid
                if _aname:
                    attributes["agent.name"] = _aname
                if _env:
                    attributes["environment"] = _env
            except Exception:
                pass
            # Fall back to span-derived agent identity if runtime_config not set
            if not attributes.get("agent.id") and agent_id:
                attributes["gen_ai.agent.id"] = str(agent_id)
            if not attributes.get("agent.name") and agent_name:
                attributes["gen_ai.agent.name"] = str(agent_name)

            # Record agent run
            if is_run:
                recorder.record_agent_run(attributes=attributes)
            
            # Record execution time
            if execution_time is not None:
                recorder.record_agent_execution_time(execution_time, attributes=attributes)
        except Exception:
            # Silently fail if metrics recording fails
            pass

    def _record_generation_metrics(self, span_data: Any, duration: Optional[float]) -> None:
        """Record token usage, cost, and duration metrics for generation spans."""
        try:
            from traccia.metrics.recorder import get_metrics_recorder
            recorder = get_metrics_recorder()
            if not recorder:
                return
            
            usage = getattr(span_data, "usage", None)
            if not usage or not isinstance(usage, dict):
                return
            
            # Support both Response API (input_tokens/output_tokens) and Completions API (prompt_tokens/completion_tokens)
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            
            model = getattr(span_data, "model", None)
            if not model:
                return
            
            attributes = {"gen_ai.system": "openai.agents", "gen_ai.request.model": str(model)}
            try:
                from traccia import runtime_config as _rc
                _aid = _rc.get_agent_id()
                _aname = _rc.get_agent_name()
                _env = _rc.get_env()
                if _aid:
                    attributes["agent.id"] = _aid
                    attributes["agent_id"] = _aid
                if _aname:
                    attributes["agent.name"] = _aname
                if _env:
                    attributes["environment"] = _env
            except Exception:
                pass

            # Duration
            if duration is not None:
                recorder.record_duration(duration, attributes=attributes)
            
            # Tokens
            if prompt_tokens is not None or completion_tokens is not None:
                recorder.record_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attributes=attributes
                )
            
            # Cost — use the process-level resolver so pricing_override applied
            # via start_tracing() is reflected here too (trace/metrics parity).
            if prompt_tokens is not None and completion_tokens is not None:
                try:
                    from traccia.cost_resolver import get_resolver
                    cost = get_resolver().compute(str(model), prompt_tokens, completion_tokens)
                    if cost is not None and cost > 0:
                        recorder.record_cost(cost, attributes=attributes)
                except Exception:
                    pass
        except Exception:
            pass

    def _get_span_name(self, span_data: Any) -> str:
        """Determine Traccia span name from Agents span data."""
        span_type = getattr(span_data, "type", "unknown")
        
        if span_type == "agent":
            agent_name = getattr(span_data, "name", "unknown")
            return f"agent.{agent_name}"
        elif span_type == "generation":
            return "llm.agents.generation"
        elif span_type == "function":
            func_name = getattr(span_data, "name", "unknown")
            return f"agent.tool.{func_name}"
        elif span_type == "handoff":
            return "agent.handoff"
        elif span_type == "guardrail":
            guardrail_name = getattr(span_data, "name", "unknown")
            return f"agent.guardrail.{guardrail_name}"
        elif span_type == "response":
            return "agent.response"
        elif span_type == "custom":
            custom_name = getattr(span_data, "name", "unknown")
            return f"agent.custom.{custom_name}"
        else:
            return f"agent.{span_type}"

    def _extract_attributes(self, span_data: Any) -> dict[str, Any]:
        """Extract initial attributes from Agents span data."""
        attrs = {
            "agent.span.type": getattr(span_data, "type", "unknown"),
        }
        
        span_type = getattr(span_data, "type", None)
        
        if span_type == "agent":
            attrs["agent.name"] = getattr(span_data, "name", None)
            tools = getattr(span_data, "tools", None)
            if tools:
                attrs["agent.tools"] = json.dumps(tools)[:500]
            handoffs = getattr(span_data, "handoffs", None)
            if handoffs:
                attrs["agent.handoffs"] = json.dumps(handoffs)[:500]
            output_type = getattr(span_data, "output_type", None)
            if output_type:
                attrs["agent.output_type"] = str(output_type)
                
        elif span_type == "generation":
            model = getattr(span_data, "model", None)
            if model:
                attrs["llm.model"] = str(model)
            model_config = getattr(span_data, "model_config", None)
            if model_config:
                attrs["llm.model_config"] = json.dumps(model_config)[:500]
                
        elif span_type == "function":
            func_name = getattr(span_data, "name", None)
            if func_name:
                attrs["agent.tool.name"] = func_name
                
        elif span_type == "handoff":
            from_agent = getattr(span_data, "from_agent", None)
            to_agent = getattr(span_data, "to_agent", None)
            if from_agent:
                attrs["agent.handoff.from"] = from_agent
            if to_agent:
                attrs["agent.handoff.to"] = to_agent
                
        elif span_type == "guardrail":
            guardrail_name = getattr(span_data, "name", None)
            if guardrail_name:
                attrs["agent.guardrail.name"] = guardrail_name
                attrs["guardrail.name"] = guardrail_name
            attrs["guardrail.source_sdk"] = "openai_agents"
            attrs["guardrail.evidence_type"] = "span_attribute"
        
        return attrs

    def _update_span_attributes(self, traccia_span: Any, span_data: Any) -> None:
        """Update Traccia span with final attributes from completed Agents span."""
        try:
            span_type = getattr(span_data, "type", None)
            
            if span_type == "generation":
                # Add usage info
                usage = getattr(span_data, "usage", None)
                if usage and isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    if input_tokens is not None:
                        traccia_span.set_attribute("llm.usage.input_tokens", input_tokens)
                        traccia_span.set_attribute("llm.usage.prompt_tokens", input_tokens)
                    if output_tokens is not None:
                        traccia_span.set_attribute("llm.usage.output_tokens", output_tokens)
                        traccia_span.set_attribute("llm.usage.completion_tokens", output_tokens)
                    if input_tokens is not None and output_tokens is not None:
                        traccia_span.set_attribute("llm.usage.total_tokens", input_tokens + output_tokens)
                
                # Add truncated input/output for observability
                input_data = getattr(span_data, "input", None)
                if input_data:
                    try:
                        input_str = json.dumps(input_data)[:1000]
                        traccia_span.set_attribute("llm.input", input_str)
                    except:
                        traccia_span.set_attribute("llm.input", str(input_data)[:1000])
                
                output_data = getattr(span_data, "output", None)
                if output_data:
                    try:
                        output_str = json.dumps(output_data)[:1000]
                        traccia_span.set_attribute("llm.output", output_str)
                    except:
                        traccia_span.set_attribute("llm.output", str(output_data)[:1000])
            
            elif span_type == "function":
                # Add function input/output
                func_input = getattr(span_data, "input", None)
                if func_input:
                    traccia_span.set_attribute("agent.tool.input", str(func_input)[:500])
                
                func_output = getattr(span_data, "output", None)
                if func_output:
                    traccia_span.set_attribute("agent.tool.output", str(func_output)[:500])
                
                mcp_data = getattr(span_data, "mcp_data", None)
                if mcp_data:
                    traccia_span.set_attribute("agent.tool.mcp", json.dumps(mcp_data)[:500])
            
            elif span_type == "guardrail":
                triggered = getattr(span_data, "triggered", False)
                traccia_span.set_attribute("agent.guardrail.triggered", triggered)
                traccia_span.set_attribute("guardrail.triggered", triggered)
            
            elif span_type == "response":
                response = getattr(span_data, "response", None)
                if response:
                    response_id = getattr(response, "id", None)
                    if response_id:
                        traccia_span.set_attribute("agent.response.id", response_id)
        
        except Exception:
            # Don't break tracing on attribute errors
            pass

    def shutdown(self) -> None:
        """Shutdown the processor."""
        try:
            self._trace_map.clear()
            self._span_map.clear()
        except Exception:
            pass

    def force_flush(self) -> None:
        """Force flush any queued spans."""
        # Traccia handles flushing at the provider level
        pass
