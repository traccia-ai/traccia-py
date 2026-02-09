"""Instrumentation for CrewAI using OpenTelemetry context."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_instrumented = False


def _record_agent_metrics(
    agent_id: Optional[str],
    agent_name: Optional[str],
    execution_time: Optional[float],
    is_run: bool = False
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
            attributes["gen_ai.agent.id"] = agent_id
        if agent_name:
            attributes["gen_ai.agent.name"] = agent_name
        
        # Record agent run (for crew kickoff)
        if is_run:
            recorder.record_agent_run(attributes=attributes)
        
        # Record execution time
        if execution_time is not None:
            recorder.record_agent_execution_time(execution_time, attributes=attributes)
    except Exception:
        # Silently fail if metrics recording fails
        pass


def instrument_crewai() -> bool:
    """
    Instrument CrewAI framework with Traccia tracing.
    
    This function monkey-patches key CrewAI methods to create spans:
    - Crew.kickoff / kickoff_async -> crew execution span
    - Task.execute_sync / execute_async -> task execution span
    - Agent.execute_task -> agent execution span
    
    LLM calls are captured by Traccia's existing OpenAI patching and will
    automatically nest under agent spans due to OpenTelemetry context propagation.
    
    Returns:
        True if instrumentation succeeded, False otherwise.
    """
    global _instrumented
    
    if _instrumented:
        logger.debug("CrewAI already instrumented")
        return True
    
    try:
        # Import CrewAI components
        from crewai import Agent, Crew, Task
        
        # Import Traccia tracer
        import traccia
        
        logger.debug("Starting CrewAI instrumentation")
        
        # Wrap Crew methods
        _wrap_method(Crew, "kickoff", _create_crew_wrapper, "crewai.crew.kickoff")
        if hasattr(Crew, "kickoff_async"):
            _wrap_method(Crew, "kickoff_async", _create_crew_wrapper, "crewai.crew.kickoff_async")
        
        # Wrap Task methods
        _wrap_method(Task, "execute_sync", _create_task_wrapper, "crewai.task.execute")
        if hasattr(Task, "execute_async"):
            _wrap_method(Task, "execute_async", _create_task_wrapper, "crewai.task.execute_async")
        
        # Wrap Agent methods
        _wrap_method(Agent, "execute_task", _create_agent_wrapper, "crewai.agent.execute")
        
        # Wrap LLM.call for explicit LLM spans under agent spans
        try:
            from crewai.llm import LLM
            if hasattr(LLM, "call"):
                _wrap_method(LLM, "call", _create_llm_wrapper, "crewai.llm.call")
                logger.debug("LLM.call instrumentation added")
        except (ImportError, AttributeError) as e:
            logger.debug(f"LLM.call not available for instrumentation: {e}")
        
        _instrumented = True
        logger.info("CrewAI instrumentation completed successfully")
        return True
        
    except ImportError as e:
        logger.debug(f"CrewAI not available for instrumentation: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to instrument CrewAI: {e}", exc_info=True)
        return False


def _wrap_method(
    cls: type,
    method_name: str,
    wrapper_factory: Callable,
    span_name: str,
) -> None:
    """
    Wrap a method on a class with tracing instrumentation.
    
    Args:
        cls: The class to wrap the method on
        method_name: Name of the method to wrap
        wrapper_factory: Factory function that creates the wrapper
        span_name: Base name for the span
    """
    if not hasattr(cls, method_name):
        logger.debug(f"{cls.__name__}.{method_name} not found, skipping")
        return
    
    original_method = getattr(cls, method_name)
    wrapped_method = wrapper_factory(original_method, span_name)
    setattr(cls, method_name, wrapped_method)
    logger.debug(f"Wrapped {cls.__name__}.{method_name}")


def _create_crew_wrapper(original_method: Callable, span_name: str) -> Callable:
    """Create a wrapper for Crew methods."""
    
    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        import traccia
        
        tracer = traccia.get_tracer("crewai")
        
        # Extract crew attributes
        attributes = _get_crew_attributes(self)
        
        # Start span with OpenTelemetry context propagation
        start_time = time.time()
        with tracer.start_as_current_span(
            span_name,
            attributes=attributes,
        ) as span:
            try:
                # Execute the original method
                result = original_method(self, *args, **kwargs)
                
                # Add result attributes
                _add_crew_result_attributes(span, result)
                
                # Record agent metrics
                execution_time = time.time() - start_time
                crew_id = getattr(self, "id", None)
                crew_name = getattr(self, "name", None) or "crew"
                _record_agent_metrics(
                    agent_id=str(crew_id) if crew_id else None,
                    agent_name=crew_name,
                    execution_time=execution_time,
                    is_run=True  # Crew kickoff is an agent run
                )
                
                return result
            except Exception as e:
                # Record exception
                if span and hasattr(span, 'record_exception'):
                    span.record_exception(e)
                    from traccia.tracer.span import SpanStatus
                    if hasattr(span, 'set_status'):
                        try:
                            span.set_status(SpanStatus.ERROR, str(e))
                        except:
                            pass
                raise
    
    return wrapper


def _create_task_wrapper(original_method: Callable, span_name: str) -> Callable:
    """Create a wrapper for Task methods."""
    
    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        import traccia
        
        tracer = traccia.get_tracer("crewai")
        
        # Extract task attributes
        attributes = _get_task_attributes(self)
        
        # Use task description or name for span name
        task_name = getattr(self, "name", None) or getattr(self, "description", "task")
        if task_name and len(task_name) > 50:
            task_name = task_name[:47] + "..."
        
        full_span_name = f"crewai.task.{task_name}"
        
        # Start span - this will automatically nest under crew span if one exists
        with tracer.start_as_current_span(
            full_span_name,
            attributes=attributes,
        ) as span:
            try:
                # Execute the original method
                result = original_method(self, *args, **kwargs)
                
                # Add result attributes
                if result and hasattr(result, "raw"):
                    _safe_set_attribute(span, "crewai.task.output", str(result.raw)[:1000])
                
                return result
            except Exception as e:
                # Record exception
                if span and hasattr(span, 'record_exception'):
                    span.record_exception(e)
                    from traccia.tracer.span import SpanStatus
                    if hasattr(span, 'set_status'):
                        try:
                            span.set_status(SpanStatus.ERROR, str(e))
                        except:
                            pass
                raise
    
    return wrapper


def _create_agent_wrapper(original_method: Callable, span_name: str) -> Callable:
    """Create a wrapper for Agent methods."""
    
    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        import traccia
        
        tracer = traccia.get_tracer("crewai")
        
        # Extract agent attributes
        attributes = _get_agent_attributes(self)
        
        # Use agent role for span name
        agent_role = getattr(self, "role", "agent")
        full_span_name = f"crewai.agent.{agent_role}"
        
        # Start span - this will automatically nest under task span if one exists
        start_time = time.time()
        with tracer.start_as_current_span(
            full_span_name,
            attributes=attributes,
        ) as span:
            try:
                # Execute the original method
                # LLM calls inside this will automatically nest due to OTel context
                result = original_method(self, *args, **kwargs)
                
                # Add result attributes
                _safe_set_attribute(span, "crewai.agent.result", str(result)[:1000])
                
                # Record agent turn metrics
                execution_time = time.time() - start_time
                agent_id = getattr(self, "id", None)
                _record_agent_metrics(
                    agent_id=str(agent_id) if agent_id else None,
                    agent_name=agent_role,
                    execution_time=execution_time,
                    is_run=False  # This is a turn, not a run
                )
                
                return result
            except Exception as e:
                # Record exception
                if span and hasattr(span, 'record_exception'):
                    span.record_exception(e)
                    from traccia.tracer.span import SpanStatus
                    if hasattr(span, 'set_status'):
                        try:
                            span.set_status(SpanStatus.ERROR, str(e))
                        except:
                            pass
                raise
    
    return wrapper


def _create_llm_wrapper(original_method: Callable, span_name: str) -> Callable:
    """Create a wrapper for LLM.call methods."""
    
    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        import traccia
        
        tracer = traccia.get_tracer("crewai")
        
        # Get model name for span name
        model_name = getattr(self, "model", "llm")
        full_span_name = f"crewai.llm.{model_name}"
        
        # Extract LLM attributes
        attributes = {
            "crewai.llm.model": str(model_name),
            "crewai.type": "llm",
        }
        
        # Add model provider if available
        if hasattr(self, "model_name"):
            attributes["crewai.llm.model_name"] = str(self.model_name)
        
        # Start span - this will automatically nest under agent span if one exists
        with tracer.start_as_current_span(
            full_span_name,
            attributes=attributes,
        ) as span:
            try:
                # Capture prompt from args if available (first arg is usually messages)
                if args and len(args) > 0:
                    messages = args[0]
                    if isinstance(messages, list) and messages:
                        # Truncate and store prompt
                        prompt_preview = str(messages)[:500]
                        _safe_set_attribute(span, "crewai.llm.prompt", prompt_preview)
                
                # Execute the original method
                result = original_method(self, *args, **kwargs)
                
                # Add result attributes (truncated)
                if result:
                    result_str = str(result)
                    _safe_set_attribute(span, "crewai.llm.completion", result_str[:1000])
                
                # Try to extract token usage from callbacks or result
                if "callbacks" in kwargs and kwargs["callbacks"]:
                    for callback in kwargs["callbacks"]:
                        if hasattr(callback, "token_cost_process"):
                            token_process = callback.token_cost_process
                            if hasattr(token_process, "total_tokens"):
                                _safe_set_attribute(span, "crewai.llm.token_usage.total", token_process.total_tokens)
                            if hasattr(token_process, "prompt_tokens"):
                                _safe_set_attribute(span, "crewai.llm.token_usage.prompt", token_process.prompt_tokens)
                            if hasattr(token_process, "completion_tokens"):
                                _safe_set_attribute(span, "crewai.llm.token_usage.completion", token_process.completion_tokens)
                            
                            # Record metrics for token usage and cost
                            prompt_tokens = getattr(token_process, "prompt_tokens", None)
                            completion_tokens = getattr(token_process, "completion_tokens", None)
                            model = getattr(self, "model", None)
                            if (prompt_tokens is not None or completion_tokens is not None) and model:
                                try:
                                    from traccia.metrics.recorder import get_metrics_recorder
                                    recorder = get_metrics_recorder()
                                    if recorder:
                                        attributes_metrics = {"gen_ai.system": "crewai", "gen_ai.request.model": str(model)}
                                        recorder.record_token_usage(
                                            prompt_tokens=prompt_tokens,
                                            completion_tokens=completion_tokens,
                                            attributes=attributes_metrics
                                        )
                                        if prompt_tokens is not None and completion_tokens is not None:
                                            from traccia.processors.cost_engine import compute_cost
                                            from traccia.pricing_config import load_pricing
                                            cost = compute_cost(str(model), prompt_tokens, completion_tokens, load_pricing())
                                            if cost is not None and cost > 0:
                                                recorder.record_cost(cost, attributes=attributes_metrics)
                                except Exception:
                                    pass
                            break  # Use first callback with token_cost_process
                
                return result
            except Exception as e:
                # Record exception
                if span and hasattr(span, 'record_exception'):
                    span.record_exception(e)
                    from traccia.tracer.span import SpanStatus
                    if hasattr(span, 'set_status'):
                        try:
                            span.set_status(SpanStatus.ERROR, str(e))
                        except:
                            pass
                raise
    
    return wrapper


def _get_crew_attributes(crew: Any) -> dict[str, Any]:
    """Extract attributes from a Crew instance."""
    attributes = {
        "crewai.crew.id": str(getattr(crew, "id", "")),
        "crewai.type": "crew",
    }
    
    # Add process type if available
    if hasattr(crew, "process"):
        attributes["crewai.crew.process"] = str(crew.process)
    
    # Add agent count
    if hasattr(crew, "agents") and crew.agents:
        attributes["crewai.crew.agent_count"] = len(crew.agents)
        
        # Add agent roles
        agent_roles = [getattr(agent, "role", "unknown") for agent in crew.agents if agent]
        if agent_roles:
            attributes["crewai.crew.agent_roles"] = ", ".join(agent_roles[:5])  # Limit to first 5
    
    # Add task count
    if hasattr(crew, "tasks") and crew.tasks:
        attributes["crewai.crew.task_count"] = len(crew.tasks)
    
    return attributes


def _get_task_attributes(task: Any) -> dict[str, Any]:
    """Extract attributes from a Task instance."""
    attributes = {
        "crewai.task.id": str(getattr(task, "id", "")),
        "crewai.type": "task",
    }
    
    # Add task name
    if hasattr(task, "name") and task.name:
        attributes["crewai.task.name"] = str(task.name)
    
    # Add task description (truncated)
    if hasattr(task, "description") and task.description:
        attributes["crewai.task.description"] = str(task.description)[:500]
    
    # Add expected output (truncated)
    if hasattr(task, "expected_output") and task.expected_output:
        attributes["crewai.task.expected_output"] = str(task.expected_output)[:500]
    
    # Add agent role if assigned
    if hasattr(task, "agent") and task.agent:
        agent_role = getattr(task.agent, "role", None)
        if agent_role:
            attributes["crewai.task.agent_role"] = str(agent_role)
    
    # Add async execution flag
    if hasattr(task, "async_execution"):
        attributes["crewai.task.async_execution"] = bool(task.async_execution)
    
    return attributes


def _get_agent_attributes(agent: Any) -> dict[str, Any]:
    """Extract attributes from an Agent instance."""
    attributes = {
        "crewai.agent.id": str(getattr(agent, "id", "")),
        "crewai.type": "agent",
    }
    
    # Add agent role
    if hasattr(agent, "role") and agent.role:
        attributes["crewai.agent.role"] = str(agent.role)
    
    # Add agent goal (truncated)
    if hasattr(agent, "goal") and agent.goal:
        attributes["crewai.agent.goal"] = str(agent.goal)[:500]
    
    # Add backstory (truncated)
    if hasattr(agent, "backstory") and agent.backstory:
        attributes["crewai.agent.backstory"] = str(agent.backstory)[:500]
    
    # Add LLM model if available
    if hasattr(agent, "llm") and agent.llm:
        if hasattr(agent.llm, "model"):
            attributes["crewai.agent.llm_model"] = str(agent.llm.model)
    
    # Add delegation flag
    if hasattr(agent, "allow_delegation"):
        attributes["crewai.agent.allow_delegation"] = bool(agent.allow_delegation)
    
    # Add verbose flag
    if hasattr(agent, "verbose"):
        attributes["crewai.agent.verbose"] = bool(agent.verbose)
    
    # Add tool count
    if hasattr(agent, "tools") and agent.tools:
        attributes["crewai.agent.tool_count"] = len(agent.tools)
        
        # Add tool names (first few)
        tool_names = []
        for tool in agent.tools[:5]:  # Limit to first 5
            if hasattr(tool, "name"):
                tool_names.append(str(tool.name))
        if tool_names:
            attributes["crewai.agent.tools"] = ", ".join(tool_names)
    
    return attributes


def _add_crew_result_attributes(span: Any, result: Any) -> None:
    """Add attributes from crew execution result."""
    try:
        # Add usage metrics if available
        if hasattr(result, "token_usage"):
            token_usage = result.token_usage
            if isinstance(token_usage, dict):
                if "total_tokens" in token_usage:
                    _safe_set_attribute(span, "crewai.crew.token_usage.total", token_usage["total_tokens"])
                if "prompt_tokens" in token_usage:
                    _safe_set_attribute(span, "crewai.crew.token_usage.prompt", token_usage["prompt_tokens"])
                if "completion_tokens" in token_usage:
                    _safe_set_attribute(span, "crewai.crew.token_usage.completion", token_usage["completion_tokens"])
        
        # Add task outputs if available
        if hasattr(result, "tasks_output") and result.tasks_output:
            _safe_set_attribute(span, "crewai.crew.tasks_completed", len(result.tasks_output))
        
        # Add final output (truncated)
        if hasattr(result, "raw"):
            _safe_set_attribute(span, "crewai.crew.output", str(result.raw)[:1000])
        elif result is not None:
            _safe_set_attribute(span, "crewai.crew.output", str(result)[:1000])
    except Exception as e:
        logger.debug(f"Failed to add crew result attributes: {e}")


def _safe_set_attribute(span: Any, key: str, value: Any) -> None:
    """Safely set an attribute on a span."""
    try:
        if span and hasattr(span, 'set_attribute'):
            span.set_attribute(key, value)
    except Exception as e:
        logger.debug(f"Failed to set attribute {key}: {e}")
