"""Traccia processor for OpenAI Agents SDK tracing."""

from __future__ import annotations

import json
import time
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


class TracciaAgentsTracingProcessor:
    """
    Traccia processor for OpenAI Agents SDK.
    
    Implements the Agents SDK TracingProcessor interface to capture agent
    runs, tool calls, handoffs, and generations into Traccia spans.
    """

    def __init__(self):
        """Initialize the processor."""
        self._trace_map = {}  # Map Agents trace_id -> Traccia trace context
        self._span_map = {}   # Map Agents span_id -> Traccia span
        self._span_start_times = {}  # Map span_id -> start time
        self._tracer = None

    def _get_tracer(self):
        """Get or create the Traccia tracer."""
        if self._tracer is None:
            import traccia
            self._tracer = traccia.get_tracer("openai.agents")
        return self._tracer

    def on_trace_start(self, trace: Trace) -> None:
        """Called when an Agents trace starts."""
        try:
            # Store trace context for span correlation
            self._trace_map[trace.trace_id] = {
                "trace_id": trace.trace_id,
                "started_at": getattr(trace, "started_at", None),
            }
        except Exception:
            # Don't break agent execution on tracing errors
            pass

    def on_trace_end(self, trace: Trace) -> None:
        """Called when an Agents trace ends."""
        try:
            # Clean up trace mapping
            self._trace_map.pop(trace.trace_id, None)
        except Exception:
            pass

    def on_span_start(self, span: Span[Any]) -> None:
        """Called when an Agents span starts."""
        try:
            tracer = self._get_tracer()
            span_data = span.span_data
            
            # Determine span name based on span type
            span_name = self._get_span_name(span_data)
            
            # Start Traccia span
            attributes = self._extract_attributes(span_data)
            traccia_span = tracer.start_span(span_name, attributes=attributes)
            
            # Store mapping and start time
            self._span_map[span.span_id] = traccia_span
            self._span_start_times[span.span_id] = time.time()
        except Exception:
            # Don't break agent execution
            pass

    def on_span_end(self, span: Span[Any]) -> None:
        """Called when an Agents span ends."""
        try:
            traccia_span = self._span_map.pop(span.span_id, None)
            if not traccia_span:
                return
            
            start_time = self._span_start_times.pop(span.span_id, None)
            
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
            if span_type == "agent" and start_time is not None:
                execution_time = time.time() - start_time
                agent_name = getattr(span_data, "name", None)
                self._record_agent_metrics(
                    agent_id=None,  # OpenAI Agents doesn't expose agent ID
                    agent_name=agent_name,
                    execution_time=execution_time,
                    is_run=True  # Agent spans represent runs
                )
            
            # Record token/cost metrics if this is a generation span
            if span_type == "generation":
                self._record_generation_metrics(span_data, start_time)
            
            # End the span
            traccia_span.end()
        except Exception:
            # Ensure span ends even if there's an error
            try:
                if traccia_span:
                    traccia_span.end()
            except:
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
            
            # Build attributes
            attributes = {}
            if agent_id:
                attributes["gen_ai.agent.id"] = str(agent_id)
            if agent_name:
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

    def _record_generation_metrics(self, span_data: Any, start_time: Optional[float]) -> None:
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
            
            # Duration
            if start_time is not None:
                duration = time.time() - start_time
                recorder.record_duration(duration, attributes=attributes)
            
            # Tokens
            if prompt_tokens is not None or completion_tokens is not None:
                recorder.record_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    attributes=attributes
                )
            
            # Cost
            if prompt_tokens is not None and completion_tokens is not None:
                try:
                    from traccia.processors.cost_engine import compute_cost
                    from traccia.pricing_config import load_pricing
                    cost = compute_cost(str(model), prompt_tokens, completion_tokens, load_pricing())
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
