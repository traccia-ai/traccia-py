"""Core metrics utilities for Traccia instrumentation.

This module provides StandardMetrics for creating OTEL-compliant metrics
and MetricsRecorder for recording metric values across instrumentations.
"""

from typing import Dict, Any, Optional
from opentelemetry.metrics import Meter, Histogram, Counter


class StandardMetrics:
    """Factory for creating standard OTEL GenAI metrics."""

    @staticmethod
    def create_token_histogram(meter: Meter) -> Histogram:
        """Create histogram for token usage (OTEL GenAI semconv)."""
        return meter.create_histogram(
            name="gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used"
        )

    @staticmethod
    def create_duration_histogram(meter: Meter) -> Histogram:
        """Create histogram for operation duration (OTEL GenAI semconv)."""
        return meter.create_histogram(
            name="gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration"
        )

    @staticmethod
    def create_cost_histogram(meter: Meter) -> Histogram:
        """Create histogram for operation cost (OTEL-style extension)."""
        return meter.create_histogram(
            name="gen_ai.client.operation.cost",
            unit="usd",
            description="Cost per LLM operation in USD"
        )

    @staticmethod
    def create_exception_counter(meter: Meter) -> Counter:
        """Create counter for exceptions."""
        return meter.create_counter(
            name="gen_ai.client.completions.exceptions",
            unit="1",
            description="Number of exceptions during LLM operations"
        )

    @staticmethod
    def create_agent_runs_counter(meter: Meter) -> Counter:
        """Create counter for agent runs."""
        return meter.create_counter(
            name="gen_ai.agent.runs",
            unit="1",
            description="Number of agent runs"
        )

    @staticmethod
    def create_agent_turns_counter(meter: Meter) -> Counter:
        """Create counter for agent turns."""
        return meter.create_counter(
            name="gen_ai.agent.turns",
            unit="1",
            description="Number of agent turns"
        )

    @staticmethod
    def create_agent_execution_time_histogram(meter: Meter) -> Histogram:
        """Create histogram for agent execution time."""
        return meter.create_histogram(
            name="gen_ai.agent.execution_time",
            unit="s",
            description="Agent execution duration"
        )

    @staticmethod
    def create_standard_metrics(meter: Meter) -> Dict[str, Any]:
        """Create all standard metrics.

        Returns:
            Dictionary with metric names as keys and metric instances as values
        """
        return {
            "token_histogram": StandardMetrics.create_token_histogram(meter),
            "duration_histogram": StandardMetrics.create_duration_histogram(meter),
            "cost_histogram": StandardMetrics.create_cost_histogram(meter),
            "exception_counter": StandardMetrics.create_exception_counter(meter),
            "agent_runs_counter": StandardMetrics.create_agent_runs_counter(meter),
            "agent_turns_counter": StandardMetrics.create_agent_turns_counter(meter),
            "agent_execution_time_histogram": StandardMetrics.create_agent_execution_time_histogram(meter),
        }


class MetricsRecorder:
    """Utility class for recording metrics in a consistent way."""

    def __init__(self, metrics: Dict[str, Any], sample_rate: float = 1.0):
        """Initialize metrics recorder.
        
        Args:
            metrics: Dictionary of metric instances from StandardMetrics
            sample_rate: Sampling rate for metrics (0.0 to 1.0)
        """
        self.metrics = metrics
        self.sample_rate = sample_rate

    def should_record(self) -> bool:
        """Determine if this metric event should be recorded based on sample rate."""
        if self.sample_rate >= 1.0:
            return True
        import random
        return random.random() <= self.sample_rate

    def record_token_usage(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Record token usage metrics.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            attributes: Additional attributes (gen_ai.system, gen_ai.request.model, etc.)
        """
        if not self.should_record():
            return

        token_histogram = self.metrics.get("token_histogram")
        if not token_histogram:
            return

        attrs = attributes or {}

        if prompt_tokens is not None and prompt_tokens > 0:
            token_histogram.record(
                prompt_tokens,
                attributes={**attrs, "gen_ai.token.type": "input"}
            )

        if completion_tokens is not None and completion_tokens > 0:
            token_histogram.record(
                completion_tokens,
                attributes={**attrs, "gen_ai.token.type": "output"}
            )

    def record_duration(
        self,
        duration: float,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Record operation duration.
        
        Args:
            duration: Duration in seconds
            attributes: Additional attributes
        """
        if not self.should_record():
            return

        duration_histogram = self.metrics.get("duration_histogram")
        if duration_histogram and duration > 0:
            duration_histogram.record(duration, attributes=attributes or {})

    def record_cost(
        self,
        cost: float,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Record operation cost.
        
        Args:
            cost: Cost in USD
            attributes: Additional attributes
        """
        if not self.should_record():
            return

        cost_histogram = self.metrics.get("cost_histogram")
        if cost_histogram and cost > 0:
            cost_histogram.record(cost, attributes=attributes or {})

    def record_exception(self, attributes: Optional[Dict[str, Any]] = None):
        """Record an exception occurrence.
        
        Args:
            attributes: Additional attributes
        """
        if not self.should_record():
            return

        exception_counter = self.metrics.get("exception_counter")
        if exception_counter:
            exception_counter.add(1, attributes=attributes or {})

    def record_agent_run(self, attributes: Optional[Dict[str, Any]] = None):
        """Record an agent run.
        
        Args:
            attributes: Additional attributes (gen_ai.agent.id, gen_ai.agent.name, etc.)
        """
        if not self.should_record():
            return

        agent_runs_counter = self.metrics.get("agent_runs_counter")
        if agent_runs_counter:
            agent_runs_counter.add(1, attributes=attributes or {})

    def record_agent_turn(self, attributes: Optional[Dict[str, Any]] = None):
        """Record an agent turn.
        
        Args:
            attributes: Additional attributes
        """
        if not self.should_record():
            return

        agent_turns_counter = self.metrics.get("agent_turns_counter")
        if agent_turns_counter:
            agent_turns_counter.add(1, attributes=attributes or {})

    def record_agent_execution_time(
        self,
        duration: float,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Record agent execution time.
        
        Args:
            duration: Duration in seconds
            attributes: Additional attributes
        """
        if not self.should_record():
            return

        agent_exec_histogram = self.metrics.get("agent_execution_time_histogram")
        if agent_exec_histogram and duration > 0:
            agent_exec_histogram.record(duration, attributes=attributes or {})
