"""Adapter layer wrapping OpenTelemetry components to match Traccia API."""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.resources import Resource as OTelResource
from opentelemetry.trace import Status as OTelStatus, StatusCode as OTelStatusCode
from opentelemetry.trace import SpanContext as OTelSpanContext
from opentelemetry.trace import Span as OTelSpan, NonRecordingSpan

from traccia.tracer.otel_utils import (
    otel_trace_id_to_traccia,
    otel_span_id_to_traccia,
    traccia_id_to_otel_trace_id,
    traccia_id_to_otel_span_id,
    otel_timestamp_to_ns,
)
from traccia.tracer.span_context import SpanContext as TracciaSpanContext
from traccia.tracer.span import SpanStatus as TracciaSpanStatus

if TYPE_CHECKING:
    from traccia.tracer.tracer import Tracer as TracciaTracer


class TracciaSpanContextAdapter:
    """Adapter wrapping OpenTelemetry SpanContext to match Traccia SpanContext API."""
    
    def __init__(self, otel_context: OTelSpanContext):
        """
        Initialize adapter with OpenTelemetry SpanContext.
        
        Args:
            otel_context: OpenTelemetry SpanContext instance
        """
        self._otel_context = otel_context
        # Convert int IDs to hex strings for Traccia compatibility
        self.trace_id = otel_trace_id_to_traccia(otel_context.trace_id)
        self.span_id = otel_span_id_to_traccia(otel_context.span_id)
        self.trace_flags = otel_context.trace_flags.sampled if hasattr(otel_context.trace_flags, 'sampled') else (1 if otel_context.is_valid else 0)
        # Convert TraceState to string
        trace_state = otel_context.trace_state
        if trace_state:
            # Convert OTel TraceState to W3C format string
            items = []
            for key, value in trace_state.items():
                items.append(f"{key}={value}")
            self.trace_state = ",".join(items) if items else None
        else:
            self.trace_state = None
    
    def is_valid(self) -> bool:
        """Check if the span context is valid."""
        return self._otel_context.is_valid
    
    @staticmethod
    def from_traccia(traccia_context: TracciaSpanContext) -> OTelSpanContext:
        """
        Convert Traccia SpanContext to OpenTelemetry SpanContext.
        
        Args:
            traccia_context: Traccia SpanContext instance
        
        Returns:
            OpenTelemetry SpanContext
        """
        from opentelemetry.trace import TraceFlags, TraceState
        
        trace_id = traccia_id_to_otel_trace_id(traccia_context.trace_id)
        span_id = traccia_id_to_otel_span_id(traccia_context.span_id)
        trace_flags = TraceFlags(traccia_context.trace_flags)
        
        # Parse trace_state string to OTel TraceState
        trace_state = None
        if traccia_context.trace_state:
            # Parse W3C format: key1=value1,key2=value2
            items = []
            for pair in traccia_context.trace_state.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    items.append((key.strip(), value.strip()))
            if items:
                trace_state = TraceState(items)
        
        return OTelSpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=trace_flags,
            trace_state=trace_state or TraceState(),
        )


class TracciaSpanAdapter:
    """Adapter wrapping OpenTelemetry Span to match Traccia Span API."""
    
    def __init__(
        self,
        otel_span: OTelSpan,
        tracer: "TracciaTracerAdapter",
        parent_span_id: Optional[str] = None,
    ):
        """
        Initialize adapter with OpenTelemetry Span.
        
        Args:
            otel_span: OpenTelemetry Span instance
            tracer: Traccia TracerAdapter that created this span
            parent_span_id: Parent span ID in hex string format
        """
        self._otel_span = otel_span
        self.tracer = tracer
        self.parent_span_id = parent_span_id
        
        # Get span context
        otel_context = otel_span.get_span_context()
        self.context = TracciaSpanContextAdapter(otel_context)
        
        # Get name
        if isinstance(otel_span, ReadableSpan):
            self.name = otel_span.name
        else:
            # For non-readable spans, we need to track name separately
            self.name = getattr(otel_span, '_traccia_name', 'unknown')
        
        # Initialize attributes dict - will be kept in sync with OTel span
        self.attributes: Dict[str, Any] = {}
        if isinstance(otel_span, ReadableSpan):
            if otel_span.attributes:
                self.attributes = dict(otel_span.attributes)
        
        # Initialize events list - will be kept in sync
        self.events: List[Dict[str, Any]] = []
        if isinstance(otel_span, ReadableSpan):
            if otel_span.events:
                for event in otel_span.events:
                    self.events.append({
                        "name": event.name,
                        "attributes": dict(event.attributes) if event.attributes else {},
                        "timestamp_ns": event.timestamp,
                    })
        
        # Convert status
        if isinstance(otel_span, ReadableSpan):
            self.status = self._convert_status(otel_span.status)
            self.status_description = otel_span.status.description if otel_span.status else None
        else:
            self.status = TracciaSpanStatus.UNSET
            self.status_description = None
        
        # Get timestamps
        if isinstance(otel_span, ReadableSpan):
            self.start_time_ns = otel_span.start_time if otel_span.start_time else time.time_ns()
            self.end_time_ns = otel_span.end_time if otel_span.end_time else None
        else:
            self.start_time_ns = time.time_ns()
            self.end_time_ns = None
        
        self._activation_tokens: Optional[Tuple] = None
        self._ended = False
        
        # Apply tracestate enrichment (matching Traccia behavior)
        self._enrich_tracestate()
    
    def _enrich_tracestate(self) -> None:
        """Enrich tracestate with runtime metadata (tenant, project, debug)."""
        try:
            from traccia.context.propagators import format_tracestate, parse_tracestate
            from traccia import runtime_config
            
            base = parse_tracestate(self.context.trace_state or "")
            if runtime_config.get_tenant_id():
                base.setdefault("tenant", runtime_config.get_tenant_id())
            if runtime_config.get_project_id():
                base.setdefault("project", runtime_config.get_project_id())
            if runtime_config.get_debug():
                base.setdefault("dbg", "1")
            
            ts = format_tracestate(base)
            if ts:
                # Update the context
                self.context.trace_state = ts
        except Exception:
            pass
    
    @staticmethod
    def _convert_status(otel_status: OTelStatus) -> TracciaSpanStatus:
        """Convert OpenTelemetry Status to Traccia SpanStatus."""
        if not otel_status:
            return TracciaSpanStatus.UNSET
        
        if otel_status.status_code == OTelStatusCode.OK:
            return TracciaSpanStatus.OK
        elif otel_status.status_code == OTelStatusCode.ERROR:
            return TracciaSpanStatus.ERROR
        else:
            return TracciaSpanStatus.UNSET
    
    @staticmethod
    def _convert_status_to_otel(traccia_status: TracciaSpanStatus, description: Optional[str] = None) -> OTelStatus:
        """Convert Traccia SpanStatus to OpenTelemetry Status."""
        if traccia_status == TracciaSpanStatus.OK:
            return OTelStatus(status_code=OTelStatusCode.OK, description=description)
        elif traccia_status == TracciaSpanStatus.ERROR:
            return OTelStatus(status_code=OTelStatusCode.ERROR, description=description)
        else:
            return OTelStatus(status_code=OTelStatusCode.UNSET, description=description)
    
    @property
    def duration_ns(self) -> Optional[int]:
        """Get span duration in nanoseconds."""
        if self.end_time_ns is None:
            return None
        return self.end_time_ns - self.start_time_ns
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        # Only set on OTel span if it's not ended
        # OTel spans don't allow setting attributes after end, but Traccia does
        # We'll allow it for Traccia compatibility but only update our local dict
        try:
            if not self._ended:
                self._otel_span.set_attribute(key, value)
        except Exception:
            # Span may be ended, just update local dict
            pass
        self.attributes[key] = value
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Add an event to the span."""
        event_dict = {
            "name": name,
            "attributes": dict(attributes) if attributes else {},
            "timestamp_ns": timestamp_ns or time.time_ns(),
        }
        self.events.append(event_dict)
        
        # Add to OTel span
        self._otel_span.add_event(
            name=name,
            attributes=attributes,
            timestamp=timestamp_ns,
        )
    
    def record_exception(self, error: BaseException) -> None:
        """Record an exception event on the span."""
        # Add to Traccia events list
        self.add_event(
            "exception",
            {
                "exception.type": error.__class__.__name__,
                "exception.message": str(error),
                "exception.stacktrace": "".join(
                    traceback.format_exception(error.__class__, error, error.__traceback__)
                ),
            },
        )
        
        # Record on OTel span
        self._otel_span.record_exception(error)
        
        # Set error status
        self.set_status(TracciaSpanStatus.ERROR, str(error))
    
    def set_status(self, status: TracciaSpanStatus, description: Optional[str] = None) -> None:
        """Set the span status."""
        self.status = status
        self.status_description = description
        
        # Convert and set on OTel span
        otel_status = self._convert_status_to_otel(status, description)
        self._otel_span.set_status(otel_status)
    
    def end(self) -> None:
        """End the span."""
        if self._ended:
            return
        
        self.end_time_ns = time.time_ns()
        if self.status == TracciaSpanStatus.UNSET:
            self.status = TracciaSpanStatus.OK
        
        self._ended = True
        
        # End OTel span
        self._otel_span.end(end_time=self.end_time_ns)
        
        # Notify tracer's provider
        self.tracer._on_span_end(self)
    
    def __enter__(self) -> "TracciaSpanAdapter":
        """Enter context manager."""
        self._activation_tokens = self.tracer._activate_span(self)
        return self
    
    def __exit__(self, exc_type, exc, tb) -> bool:
        """Exit context manager."""
        try:
            if exc:
                self.record_exception(exc)
            self.end()
        finally:
            if self._activation_tokens:
                self.tracer._deactivate_span(self._activation_tokens)
                self._activation_tokens = None
        return False
    
    async def __aenter__(self) -> "TracciaSpanAdapter":
        """Enter async context manager."""
        self._activation_tokens = self.tracer._activate_span(self)
        return self
    
    async def __aexit__(self, exc_type, exc, tb) -> bool:
        """Exit async context manager."""
        try:
            if exc:
                self.record_exception(exc)
            self.end()
        finally:
            if self._activation_tokens:
                self.tracer._deactivate_span(self._activation_tokens)
                self._activation_tokens = None
        return False


class TracciaTracerAdapter:
    """Adapter wrapping OpenTelemetry Tracer to match Traccia Tracer API."""
    
    def __init__(self, otel_tracer: otel_trace_api.Tracer, provider: "TracciaTracerProviderAdapter", instrumentation_scope: str):
        """
        Initialize adapter with OpenTelemetry Tracer.
        
        Args:
            otel_tracer: OpenTelemetry Tracer instance
            provider: Traccia TracerProviderAdapter that created this tracer
            instrumentation_scope: Instrumentation scope name
        """
        self._otel_tracer = otel_tracer
        self._provider = provider
        self.instrumentation_scope = instrumentation_scope
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional["TracciaSpanAdapter"] = None,
        parent_context: Optional[TracciaSpanContext] = None,
    ) -> TracciaSpanAdapter:
        """
        Start a new span.
        
        Args:
            name: Span name
            attributes: Optional attributes dictionary
            parent: Optional parent Traccia span
            parent_context: Optional parent Traccia span context
        
        Returns:
            TracciaSpanAdapter wrapping the new OTel span
        """
        from traccia.context import context as span_context
        
        # Determine parent
        parent_span = parent or span_context.get_current_span()
        
        # Convert parent context if provided
        otel_parent_context = None
        parent_span_id = None
        
        if parent_span:
            # Use parent span's context
            parent_span_id = parent_span.context.span_id
            if hasattr(parent_span, '_otel_span'):
                otel_parent_context = otel_trace_api.set_span_in_context(parent_span._otel_span)
            else:
                # Parent is native Traccia span, convert its context
                otel_parent_ctx = TracciaSpanContextAdapter.from_traccia(parent_span.context)
                otel_parent_context = otel_trace_api.set_span_in_context(
                    NonRecordingSpan(otel_parent_ctx)
                )
        elif parent_context and parent_context.is_valid():
            # Convert provided parent context
            parent_span_id = parent_context.span_id
            otel_parent_ctx = TracciaSpanContextAdapter.from_traccia(parent_context)
            otel_parent_context = otel_trace_api.set_span_in_context(
                NonRecordingSpan(otel_parent_ctx)
            )
        
        # Start OTel span
        otel_span = self._otel_tracer.start_span(
            name=name,
            attributes=attributes,
            context=otel_parent_context,
        )
        
        # Store name for non-readable spans
        if not isinstance(otel_span, ReadableSpan):
            otel_span._traccia_name = name
        
        # Wrap in adapter
        return TracciaSpanAdapter(otel_span, self, parent_span_id)
    
    def start_as_current_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional["TracciaSpanAdapter"] = None,
        parent_context: Optional[TracciaSpanContext] = None,
    ) -> TracciaSpanAdapter:
        """
        Start a span and set it as current.
        
        Args:
            name: Span name
            attributes: Optional attributes dictionary
            parent: Optional parent Traccia span
            parent_context: Optional parent Traccia span context
        
        Returns:
            TracciaSpanAdapter wrapping the new OTel span
        """
        return self.start_span(
            name=name,
            attributes=attributes,
            parent=parent,
            parent_context=parent_context,
        )
    
    def get_current_span(self) -> Optional["TracciaSpanAdapter"]:
        """Get the current span."""
        from traccia.context import context as span_context
        return span_context.get_current_span()
    
    def _activate_span(self, span: "TracciaSpanAdapter"):
        """Activate a span (set as current)."""
        from traccia.context import context as span_context
        return span_context.push_span(span)
    
    def _deactivate_span(self, tokens) -> None:
        """Deactivate a span (restore previous)."""
        from traccia.context import context as span_context
        span_context.pop_span(tokens)
    
    def _on_span_end(self, span: "TracciaSpanAdapter") -> None:
        """Called when a span ends."""
        self._provider._notify_span_end(span)


class TracciaTracerProviderAdapter:
    """Adapter wrapping OpenTelemetry TracerProvider to match Traccia TracerProvider API."""
    
    def __init__(self, resource: Optional[Dict[str, str]] = None):
        """
        Initialize adapter with OpenTelemetry TracerProvider.
        
        Args:
            resource: Optional resource attributes dictionary
        """
        # Convert resource dict to OTel Resource
        otel_resource = OTelResource.create(resource or {})
        self._otel_provider = OTelTracerProvider(resource=otel_resource)
        
        # Store resource as dict for Traccia compatibility
        self.resource = resource or {}
        
        # Tracers cache
        self._tracers: Dict[str, TracciaTracerAdapter] = {}
        self._span_processors: List[Any] = []
        self.sampler: Optional[Any] = None
    
    def get_tracer(self, name: str) -> TracciaTracerAdapter:
        """
        Get a tracer by name.
        
        Args:
            name: Instrumentation scope name
        
        Returns:
            TracciaTracerAdapter wrapping OTel Tracer
        """
        if name in self._tracers:
            return self._tracers[name]
        
        # Get OTel tracer
        otel_tracer = self._otel_provider.get_tracer(name)
        
        # Wrap in adapter
        tracer = TracciaTracerAdapter(otel_tracer, self, name)
        self._tracers[name] = tracer
        return tracer
    
    def add_span_processor(self, processor: Any) -> None:
        """
        Add a span processor.
        
        Args:
            processor: Traccia SpanProcessor instance
        """
        self._span_processors.append(processor)
        
        # If processor has OTel compatibility, add to OTel provider
        # For now, we'll handle Traccia processors via _notify_span_end
    
    def set_sampler(self, sampler: Any) -> None:
        """
        Set the sampler.
        
        Args:
            sampler: Traccia Sampler instance
        """
        self.sampler = sampler
        # Note: OTel sampler is set at provider creation time
        # For dynamic sampler changes, we'd need to recreate the provider
    
    def get_sampler(self) -> Optional[Any]:
        """Get the current sampler."""
        return self.sampler
    
    def _notify_span_end(self, span: TracciaSpanAdapter) -> None:
        """
        Notify processors that a span has ended.
        
        Args:
            span: The span that ended
        """
        for processor in list(self._span_processors):
            try:
                processor.on_end(span)
            except Exception:
                # Processors should not crash tracing
                continue
    
    def force_flush(self, timeout: Optional[float] = None) -> bool:
        """Force flush all processors.

        Returns True if the underlying OTel provider reports a successful flush
        of every span processor within the timeout, False otherwise.
        """
        # Flush OTel provider
        ok = self._otel_provider.force_flush(
            timeout_millis=int(timeout * 1000) if timeout else 30000
        )

        # Flush Traccia processors
        for processor in list(self._span_processors):
            try:
                processor.force_flush(timeout=timeout)
            except Exception:
                continue

        return bool(ok)
    
    def shutdown(self) -> None:
        """Shutdown the provider and all processors."""
        # Shutdown OTel provider
        self._otel_provider.shutdown()
        
        # Shutdown Traccia processors
        for processor in list(self._span_processors):
            try:
                processor.shutdown()
            except Exception:
                continue
    
    @staticmethod
    def generate_trace_id() -> str:
        """Generate a new trace ID in Traccia format (hex string)."""
        import secrets
        return secrets.token_hex(16)
    
    @staticmethod
    def generate_span_id() -> str:
        """Generate a new span ID in Traccia format (hex string)."""
        import secrets
        return secrets.token_hex(8)
