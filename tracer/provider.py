"""TracerProvider using OpenTelemetry SDK."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
from opentelemetry.sdk.trace import SpanProcessor as OTelSpanProcessor
from opentelemetry.sdk.resources import Resource as OTelResource
from opentelemetry.sdk.trace.sampling import Sampler as OTelSampler
from opentelemetry.sdk.trace.sampling import SamplingResult as OTelSamplingResult


class SpanProcessor:
    """
    Base span processor interface for Traccia enrichment processors.
    
    Enrichment processors run BEFORE span.end() (span is mutable).
    Export processors use OTel's SpanProcessor interface (run AFTER span.end()).
    """

    def on_end(self, span) -> None:
        """
        Called when a span ends.
        
        Note: This is called BEFORE the OTel span ends, so the span is still mutable.
        You can call span.set_attribute() here.
        
        Args:
            span: Traccia Span instance (mutable)
        """
        pass

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout: Optional[float] = None) -> None:
        """Force flush any pending spans."""
        pass


class _SwitchableSampler(OTelSampler):
    """
    OTel sampler that forwards to a replaceable delegate.

    OTel tracers capture the provider's sampler when they are created and the
    provider caches them per scope, so assigning a new sampler to the provider
    would not reach tracers that already exist. Installing this once and
    swapping its delegate changes sampling for every tracer immediately.
    """

    def __init__(self, delegate: OTelSampler) -> None:
        self.delegate = delegate

    def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, trace_state=None) -> OTelSamplingResult:
        return self.delegate.should_sample(
            parent_context, trace_id, name, kind=kind, attributes=attributes, links=links, trace_state=trace_state
        )

    def get_description(self) -> str:
        return self.delegate.get_description()


class TracerProvider:
    """
    TracerProvider using OpenTelemetry SDK.
    
    Separates enrichment processors (Traccia) from export processors (OTel).
    """

    def __init__(self, resource: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize TracerProvider with OpenTelemetry.
        
        Args:
            resource: Resource attributes dictionary (converted to OTel Resource)
        """
        # Convert resource dict to OTel Resource
        otel_resource = OTelResource.create(resource or {})
        self._otel_provider = OTelTracerProvider(resource=otel_resource)
        
        # Store resource as dict for backward compatibility
        self.resource = resource or {}
        
        # Separate enrichment vs export processors
        self._enrichment_processors: List[SpanProcessor] = []  # Traccia processors
        self._export_processors: List[OTelSpanProcessor] = []   # OTel processors
        
        # Tracers cache
        self._tracers: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Optional sampler used for head-based sampling at trace start
        self.sampler: Optional[Any] = None
        # OTel's own default (env-configured or parent-based always-on), restored by set_sampler(None).
        # Installed behind a switch before any tracer exists, so set_sampler reaches every tracer.
        self._default_otel_sampler: OTelSampler = self._otel_provider.sampler
        self._sampler_switch = _SwitchableSampler(self._default_otel_sampler)
        self._otel_provider.sampler = self._sampler_switch

    def get_tracer(self, name: str) -> "Tracer":
        """
        Get a tracer by name.
        
        Args:
            name: Instrumentation scope name
        
        Returns:
            Traccia Tracer instance (wraps OTel Tracer)
        """
        with self._lock:
            tracer = self._tracers.get(name)
            if tracer is None:
                from traccia.tracer.tracer import Tracer
                tracer = Tracer(self, name)
                self._tracers[name] = tracer
            return tracer

    def add_span_processor(self, processor: Any) -> None:
        """
        Add a span processor.
        
        Separates enrichment processors (Traccia) from export processors (OTel).
        
        Args:
            processor: SpanProcessor instance (OTel-compatible or Traccia-compatible)
        """
        # If processor is OTel-compatible, add to OTel provider
        if isinstance(processor, OTelSpanProcessor):
            self._otel_provider.add_span_processor(processor)
            self._export_processors.append(processor)
        else:
            # Traccia enrichment processor
            self._enrichment_processors.append(processor)

    def set_sampler(self, sampler: Any) -> None:
        """
        Set the sampler and apply it to the OTel provider.

        Takes effect for every span started after this call, including spans
        from tracers created earlier. Child spans follow their parent's decision.

        Args:
            sampler: Traccia Sampler, OTel Sampler, or None to restore the OTel default
        """
        self.sampler = sampler
        self._sampler_switch.delegate = self._resolve_otel_sampler(sampler)

    def _resolve_otel_sampler(self, sampler: Any) -> OTelSampler:
        if sampler is None:
            return self._default_otel_sampler
        if isinstance(sampler, OTelSampler):
            return sampler
        if hasattr(sampler, "to_otel_sampler"):
            return sampler.to_otel_sampler()
        raise TypeError(f"Unsupported sampler type: {type(sampler).__name__}")

    def get_sampler(self) -> Optional[Any]:
        """Get the current sampler."""
        return self.sampler

    def force_flush(self, timeout: Optional[float] = None) -> None:
        """Force flush all processors."""
        # Flush OTel processors
        self._otel_provider.force_flush(timeout_millis=int(timeout * 1000) if timeout else 30000)
        
        # Flush Traccia enrichment processors
        for processor in self._enrichment_processors:
            try:
                processor.force_flush(timeout=timeout)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Shutdown the provider and all processors."""
        # Shutdown OTel processors
        self._otel_provider.shutdown()
        
        # Shutdown Traccia enrichment processors
        for processor in self._enrichment_processors:
            try:
                processor.shutdown()
            except Exception:
                pass

    @property
    def _otel_tracer_provider(self) -> OTelTracerProvider:
        """Get the underlying OpenTelemetry TracerProvider."""
        return self._otel_provider
