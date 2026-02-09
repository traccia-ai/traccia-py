"""Python SDK entrypoint for the agent tracing library."""

from traccia.auto import start_tracing, stop_tracing, init, trace, end_auto_trace
from traccia.tracer import TracerProvider
from traccia.instrumentation.decorator import observe
from traccia import metrics

# Version exposure
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("traccia")
    except PackageNotFoundError:
        __version__ = "0.1.0"  # Fallback version
except ImportError:
    __version__ = "0.1.0"  # Fallback for Python < 3.8

# Initialize global tracer provider (now uses OpenTelemetry directly)
_global_tracer_provider = TracerProvider()


def get_tracer(name: str = "default"):
    """Fetch a tracer from the global provider."""
    return _global_tracer_provider.get_tracer(name)


def span(name: str, attributes: dict = None):
    """
    Convenience function to create a span using the default tracer.
    
    This is a simpler alternative to:
        tracer = get_tracer("name")
        with tracer.start_as_current_span("span_name") as span:
    
    Usage:
        from traccia import span
        
        with span("my_operation"):
            # Your code here
            pass
        
        # With attributes
        with span("my_operation", {"key": "value"}) as s:
            s.set_attribute("another", "attr")
            # Your code here
    """
    tracer = get_tracer()
    return tracer.start_as_current_span(name, attributes=attributes)


def set_tracer_provider(provider: TracerProvider) -> None:
    """Override the global tracer provider (primarily for tests or customization)."""
    global _global_tracer_provider
    _global_tracer_provider = provider


def get_tracer_provider() -> TracerProvider:
    return _global_tracer_provider


__all__ = [
    "__version__",
    "get_tracer",
    "get_tracer_provider",
    "set_tracer_provider",
    "start_tracing",
    "stop_tracing",
    "init",
    "trace",
    "end_auto_trace",
    "span",
    "observe",
    "metrics",
]

