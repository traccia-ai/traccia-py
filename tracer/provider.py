"""TracerProvider using OpenTelemetry SDK."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
from opentelemetry.sdk.trace import SpanProcessor as OTelSpanProcessor
from opentelemetry.sdk.resources import Resource as OTelResource


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
        Set the sampler.
        
        Note: OTel samplers are set at provider creation time.
        This method stores the sampler for reference but doesn't
        apply it dynamically. For dynamic sampling, recreate the provider.
        
        Args:
            sampler: Sampler instance
        """
        self.sampler = sampler
        # Note: OTel doesn't support dynamic sampler changes
        # The sampler would need to be set at provider creation

    def get_sampler(self) -> Optional[Any]:
        """Get the current sampler."""
        return self.sampler

    def force_flush(self, timeout: Optional[float] = None) -> bool:
        """Force flush all processors.

        Returns True if the underlying OTel provider reports that every span
        processor flushed within the timeout, False otherwise. Enrichment
        processor failures are swallowed (they don't export) and don't affect
        the return value.
        """
        # Flush OTel processors
        ok = self._otel_provider.force_flush(
            timeout_millis=int(timeout * 1000) if timeout else 30000
        )

        # Flush Traccia enrichment processors
        for processor in self._enrichment_processors:
            try:
                processor.force_flush(timeout=timeout)
            except Exception:
                pass

        return bool(ok)

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
