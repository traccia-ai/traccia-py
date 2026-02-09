"""Global metrics recorder for custom metrics API."""

from typing import Dict, Any, Optional

# Global recorder instance (initialized by auto.py)
_global_recorder: Optional[Any] = None


def set_global_recorder(recorder):
    """Set the global metrics recorder (called by auto.py during initialization)."""
    global _global_recorder
    _global_recorder = recorder


def get_metrics_recorder():
    """Get the global metrics recorder instance.
    
    Returns:
        MetricsRecorder instance or None if metrics are not enabled
    """
    return _global_recorder


def record_counter(name: str, value: int = 1, attributes: Optional[Dict[str, Any]] = None):
    """Record a custom counter metric.
    
    Args:
        name: Metric name
        value: Counter value (default: 1)
        attributes: Metric attributes/dimensions
        
    Example:
        >>> from traccia.metrics import record_counter
        >>> record_counter("my_custom_events", 1, {"event_type": "user_action"})
    """
    if _global_recorder is None:
        return
    
    # Get the meter from the recorder's metrics dict (if available)
    # For custom metrics, we need access to the meter
    # We'll add a custom_metrics dict to the recorder that holds custom metric instances
    if not hasattr(_global_recorder, '_custom_counters'):
        _global_recorder._custom_counters = {}
    
    if name not in _global_recorder._custom_counters:
        # Create counter on-demand using the meter
        if hasattr(_global_recorder, '_meter'):
            counter = _global_recorder._meter.create_counter(
                name=name,
                unit="1",
                description=f"Custom counter: {name}"
            )
            _global_recorder._custom_counters[name] = counter
        else:
            return
    
    counter = _global_recorder._custom_counters.get(name)
    if counter:
        counter.add(value, attributes=attributes or {})


def record_histogram(name: str, value: float, attributes: Optional[Dict[str, Any]] = None, unit: str = "1"):
    """Record a custom histogram metric.
    
    Args:
        name: Metric name
        value: Histogram value
        attributes: Metric attributes/dimensions
        unit: Unit of measurement (default: "1")
        
    Example:
        >>> from traccia.metrics import record_histogram
        >>> record_histogram("my_custom_latency", 0.123, {"service": "api"}, unit="s")
    """
    if _global_recorder is None:
        return
    
    if not hasattr(_global_recorder, '_custom_histograms'):
        _global_recorder._custom_histograms = {}
    
    if name not in _global_recorder._custom_histograms:
        # Create histogram on-demand
        if hasattr(_global_recorder, '_meter'):
            histogram = _global_recorder._meter.create_histogram(
                name=name,
                unit=unit,
                description=f"Custom histogram: {name}"
            )
            _global_recorder._custom_histograms[name] = histogram
        else:
            return
    
    histogram = _global_recorder._custom_histograms.get(name)
    if histogram:
        histogram.record(value, attributes=attributes or {})
