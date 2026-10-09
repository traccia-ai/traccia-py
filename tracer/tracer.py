"""Tracer using OpenTelemetry SDK with Traccia API compatibility."""

from __future__ import annotations

from typing import Any, Dict, Optional

from opentelemetry.trace import Tracer as OTelTracer
from opentelemetry.trace import set_span_in_context
from opentelemetry import context as context_api

from traccia import runtime_config


class Tracer:
    """
    Tracer wrapper that uses OpenTelemetry Tracer internally.
    
    Maintains Traccia API compatibility while using OTel underneath.
    """

    def __init__(self, provider: "TracerProvider", instrumentation_scope: str):
        """
        Initialize tracer with OpenTelemetry Tracer.
        
        Args:
            provider: Traccia TracerProvider instance
            instrumentation_scope: Instrumentation scope name
        """
        self._provider = provider
        self.instrumentation_scope = instrumentation_scope
        
        # Get OTel tracer from provider's OTel provider
        self._otel_tracer: OTelTracer = provider._otel_provider.get_tracer(instrumentation_scope)

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Any] = None,
        parent_context: Optional[Any] = None,
        start_time: Optional[int] = None,
    ) -> "Span":
        """
        Start a new span.

        Args:
            name: Span name
            attributes: Optional attributes dictionary
            parent: Optional parent span
            parent_context: Optional parent span context
            start_time: Optional explicit start time in nanoseconds since epoch.
                Defaults to now. Use when replaying externally-timestamped events
                (e.g. reconstructing spans from a log recorded earlier).

        Returns:
            Traccia Span instance (wraps OTel Span)
        """
        from opentelemetry.trace import get_current_span
        
        # Determine parent context
        otel_parent_context = None
        parent_span_id = None
        
        if parent:
            # Extract parent span ID for Traccia compatibility
            if hasattr(parent, 'context'):
                parent_span_id = parent.context.span_id
            
            # Get OTel span from parent
            if hasattr(parent, '_otel_span'):
                otel_parent_context = set_span_in_context(parent._otel_span)
            elif hasattr(parent, 'get_span_context'):
                # Direct OTel span
                otel_parent_context = set_span_in_context(parent)
                from traccia.utils.helpers import format_span_id
                parent_span_id = format_span_id(parent.get_span_context().span_id)
        
        elif parent_context:
            # Convert Traccia SpanContext to OTel context
            if hasattr(parent_context, 'trace_id'):
                from traccia.utils.helpers import parse_trace_id, parse_span_id
                from opentelemetry.trace import SpanContext as OTelSpanContext, TraceFlags, TraceState, NonRecordingSpan
                
                trace_id = parse_trace_id(parent_context.trace_id)
                span_id = parse_span_id(parent_context.span_id)
                trace_flags = TraceFlags(parent_context.trace_flags)
                
                # Parse trace_state
                trace_state = None
                if parent_context.trace_state:
                    from traccia.context.propagators import parse_tracestate
                    parsed = parse_tracestate(parent_context.trace_state)
                    if parsed:
                        items = [(k, v) for k, v in parsed.items()]
                        trace_state = TraceState(items)
                
                otel_span_context = OTelSpanContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    is_remote=False,
                    trace_flags=trace_flags,
                    trace_state=trace_state or TraceState(),
                )
                otel_parent_context = set_span_in_context(NonRecordingSpan(otel_span_context))
                parent_span_id = parent_context.span_id
        
        # If no parent specified, use current span
        if otel_parent_context is None:
            current_span = get_current_span()
            if current_span and current_span.get_span_context().is_valid:
                otel_parent_context = set_span_in_context(current_span)
                from traccia.utils.helpers import format_span_id
                parent_span_id = format_span_id(current_span.get_span_context().span_id)
        
        # Handle sampling
        sampler = getattr(self._provider, "sampler", None)
        if sampler and otel_parent_context is None:
            # New root trace - check sampler
            try:
                sampled = bool(sampler.should_sample().sampled)
                if not sampled:
                    # Create a non-recording span for unsampled traces
                    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
                    from opentelemetry.trace.id_generator import RandomIdGenerator
                    
                    id_generator = RandomIdGenerator()
                    trace_id = id_generator.generate_trace_id()
                    span_id = id_generator.generate_span_id()
                    
                    unsampled_context = SpanContext(
                        trace_id=trace_id,
                        span_id=span_id,
                        is_remote=False,
                        trace_flags=TraceFlags(0),  # Not sampled
                    )
                    unsampled_span = NonRecordingSpan(unsampled_context)
                    otel_parent_context = set_span_in_context(unsampled_span)
            except Exception:
                pass
        
        # Debug override: if enabled, force sampling for new traces
        if otel_parent_context is None and runtime_config.get_debug():
            pass  # OTel will handle this
        
        # Check for auto-trace conflict
        self._check_auto_trace_conflict(name, otel_parent_context)
        
        # Start OTel span
        start_span_kwargs: Dict[str, Any] = {
            "name": name,
            "attributes": attributes,
            "context": otel_parent_context,
        }
        if start_time is not None:
            start_span_kwargs["start_time"] = start_time
        otel_span = self._otel_tracer.start_span(**start_span_kwargs)

        # Wrap in Traccia Span
        from traccia.tracer.span import Span
        return Span(otel_span, self, parent_span_id, start_time_ns=start_time)

    def start_as_current_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Any] = None,
        parent_context: Optional[Any] = None,
        start_time: Optional[int] = None,
    ) -> "Span":
        """
        Start a span and set it as current (context manager).

        Args:
            name: Span name
            attributes: Optional attributes dictionary
            parent: Optional parent span
            parent_context: Optional parent span context
            start_time: Optional explicit start time in nanoseconds since epoch.

        Returns:
            Traccia Span instance (wraps OTel Span)
        """
        return self.start_span(
            name=name,
            attributes=attributes,
            parent=parent,
            parent_context=parent_context,
            start_time=start_time,
        )

    def get_current_span(self) -> Optional["Span"]:
        """Get the current span."""
        from opentelemetry.trace import get_current_span
        otel_span = get_current_span()
        if otel_span and otel_span.get_span_context().is_valid:
            from traccia.tracer.span import Span
            # Check if we already have a Traccia wrapper
            if hasattr(otel_span, '_traccia_tracer'):
                # Try to return existing wrapper (best effort)
                pass
            return Span(otel_span, self)
        return None
    
    def _check_auto_trace_conflict(self, span_name: str, parent_context: Optional[Any]) -> None:
        """
        Check if user is creating a span with 'root' in name while auto-trace is active.
        
        Logs an informational warning if a conflict is detected.
        
        Args:
            span_name: Name of the span being created
            parent_context: Parent context (if None, this might be a root span)
        """
        # Import here to avoid circular dependency
        from traccia import auto
        
        # Only warn if auto-trace is active
        if not auto._auto_trace_context:
            return
        
        # Only warn if span name is exactly "root" (case-insensitive) to avoid false positives
        # This helps users who might be migrating from manual root span creation
        if span_name.lower() != "root":
            return
        
        # Only warn if this would be a root span (no parent context)
        # Note: If parent_context exists, this is a child span and that's expected
        if parent_context is not None:
            return
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Auto-started trace '{auto._auto_trace_name}' is active. "
            f"Created span '{span_name}' will be a child of the auto-started trace. "
            f"Use traccia.end_auto_trace() if you want a separate trace."
        )

    def _run_enrichment_processors(self, span: "Span") -> None:
        """
        Run enrichment processors before span ends.
        
        Called by Span.end() before the OTel span is ended.
        
        Args:
            span: Traccia Span instance (still mutable)
        """
        for processor in self._provider._enrichment_processors:
            try:
                processor.on_end(span)
            except Exception:
                # Processors should not crash tracing
                pass
