"""Initialization helpers for wiring tracer provider, processors, and patches."""

from __future__ import annotations

import atexit
import inspect
import os
import sys
from pathlib import Path
from typing import Optional, Any

from traccia.exporter import ConsoleExporter, FileExporter, OTLPExporter
from traccia.config import DEFAULT_OTLP_TRACE_ENDPOINT
from traccia.instrumentation import patch_anthropic, patch_openai, patch_requests
from traccia.processors import (
    BatchSpanProcessor,
    Sampler,
    TokenCountingProcessor,
    CostAnnotatingProcessor,
    LoggingSpanProcessor,
    AgentEnrichmentProcessor,
)
from traccia import pricing_config
import threading
import time
from traccia.tracer.provider import TracerProvider
from traccia import config as sdk_config
from traccia import runtime_config
from traccia import auto_instrumentation

# Metrics imports
from traccia.metrics import StandardMetrics, MetricsRecorder
from traccia.metrics.recorder import set_global_recorder

_started = False
_registered_shutdown = False
_active_processor: Optional[BatchSpanProcessor] = None
_init_method: Optional[str] = None  # Track how SDK was initialized: "init" or "start_tracing"
_auto_trace_context: Optional[Any] = None  # Context for auto-started trace
_auto_trace_name: str = "root"  # Default name for auto-started trace


def init(
    api_key: Optional[str] = None,
    *,
    auto_start_trace: bool = True,
    auto_trace_name: str = "root",
    config_file: Optional[str] = None,
    **kwargs
) -> TracerProvider:
    """
    Simplified initialization for Traccia SDK with config file support.
    
    Configuration priority (highest to lowest):
    1. Explicit parameters (kwargs)
    2. Environment variables
    3. Config file (./traccia.toml or ~/.traccia/config.toml)
    
    Args:
        api_key: Optional API key (required for SaaS, optional for open-source)
        auto_start_trace: If True, automatically start a root trace (default: True)
        auto_trace_name: Name for auto-started trace (default: "root")
        config_file: Optional explicit path to config file
        **kwargs: All parameters from start_tracing() can be passed here
        
    Returns:
        TracerProvider instance
        
    Example:
        >>> import traccia
        >>> traccia.init(api_key="...")
        >>> # All spans created after this are children of auto-started trace
    """
    global _started, _init_method, _auto_trace_context, _auto_trace_name
    
    # Check if already initialized
    if _started:
        if _init_method == "start_tracing":
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "SDK was initialized with start_tracing(). "
                "Calling init() will not re-initialize. "
                "Use stop_tracing() first if you need to re-initialize."
            )
        return _get_provider()
    
    # Load config file if exists (lowest priority)
    merged_config = {}
    if config_file or sdk_config.find_config_file():
        file_config = sdk_config.load_config_with_priority(config_file=config_file)
        merged_config.update(file_config)
    
    # Override with explicit parameters (highest priority)
    if api_key is not None:
        merged_config['api_key'] = api_key
    for key, value in kwargs.items():
        if value is not None:
            merged_config[key] = value
    
    # Handle auto_start_trace and auto_trace_name - these are init() specific, not start_tracing()
    # Get auto_start_trace from merged config or use default
    final_auto_start = merged_config.pop('auto_start_trace', auto_start_trace)
    if isinstance(final_auto_start, str):
        # Convert string to bool if needed
        final_auto_start = final_auto_start.lower() in ('true', '1', 'yes')
    
    # Store auto-trace config before calling start_tracing
    _auto_trace_name = merged_config.pop('auto_trace_name', auto_trace_name)
    
    # Map config file keys to start_tracing() parameter names
    # Config file uses shorter names, start_tracing() uses full names
    key_mapping = {
        'enable_console': 'enable_console_exporter',
        'enable_file': 'enable_file_exporter',
    }
    for old_key, new_key in key_mapping.items():
        if old_key in merged_config:
            merged_config[new_key] = merged_config.pop(old_key)
    
    # Extract rate limiting config to pass separately to start_tracing
    rate_limit_config = {
        'max_spans_per_second': merged_config.pop('max_spans_per_second', None),
        'max_block_ms': merged_config.pop('max_block_ms', 100),
    }
    
    # Add rate limiting config back into merged_config for start_tracing
    merged_config.update(rate_limit_config)
    
    # Initialize via start_tracing with full config
    provider = start_tracing(**merged_config)
    _init_method = "init"
    
    # Auto-start trace if requested
    if final_auto_start:
        _auto_trace_context = _start_auto_trace(provider, _auto_trace_name)
        if not _registered_shutdown:
            atexit.register(_cleanup_auto_trace)
    
    return provider


def _start_auto_trace(provider: TracerProvider, name: str = "root") -> Any:
    """
    Start an auto-managed root trace.
    
    Args:
        provider: TracerProvider instance
        name: Name for the root trace span
        
    Returns:
        Span context for cleanup
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        tracer = provider.get_tracer("traccia.auto")
        
        # Create root span and make it current
        span = tracer.start_span(
            name=name,
            attributes={"traccia.auto_started": True}
        )
        
        # Make this span the current span in the context
        from opentelemetry import context
        from opentelemetry.trace import set_span_in_context
        
        token = context.attach(set_span_in_context(span))
        
        logger.debug(f"Auto-started trace '{name}' created")
        
        return {"span": span, "token": token}
    
    except Exception as e:
        logger.error(f"Failed to start auto-trace: {e}")
        return None


def _cleanup_auto_trace() -> None:
    """Cleanup auto-started trace on program exit."""
    global _auto_trace_context
    
    if _auto_trace_context and _auto_trace_context.get("span"):
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            span = _auto_trace_context["span"]
            if hasattr(span, "is_recording") and span.is_recording():
                span.end()
                logger.debug("Auto-started trace ended")
            
            # Detach context
            if _auto_trace_context.get("token"):
                from opentelemetry import context
                context.detach(_auto_trace_context["token"])
        
        except Exception as e:
            logger.error(f"Error cleaning up auto-trace: {e}")
        
        finally:
            _auto_trace_context = None


def end_auto_trace() -> None:
    """
    Explicitly end the auto-started trace.
    
    This allows users to end the auto-trace and create their own root traces.
    """
    global _auto_trace_context
    
    if _auto_trace_context:
        _cleanup_auto_trace()


class trace:
    """
    Context manager for explicit trace management.
    
    Ends auto-trace if active and starts a new explicit trace.
    
    Example:
        >>> import traccia
        >>> traccia.init()
        >>> with traccia.trace("custom-trace"):
        ...     # Your code here
        ...     pass
    """
    
    def __init__(self, name: str = "trace", **kwargs):
        """
        Initialize trace context manager.
        
        Args:
            name: Name for the trace span
            **kwargs: Additional span attributes
        """
        self.name = name
        self.kwargs = kwargs
        self.span = None
        self.token = None
    
    def __enter__(self):
        """Start the explicit trace."""
        import logging
        logger = logging.getLogger(__name__)
        
        # End auto-trace if active
        if _auto_trace_context:
            logger.debug("Ending auto-trace to start explicit trace")
            end_auto_trace()
        
        # Start new explicit trace
        try:
            provider = _get_provider()
            tracer = provider.get_tracer("traccia.explicit")
            
            self.span = tracer.start_span(
                name=self.name,
                attributes=self.kwargs
            )
            
            # Make this span the current span
            from opentelemetry import context
            from opentelemetry.trace import set_span_in_context
            
            self.token = context.attach(set_span_in_context(self.span._otel_span))
            
            return self.span
        
        except Exception as e:
            logger.error(f"Failed to start explicit trace: {e}")
            return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the explicit trace."""
        if self.span:
            try:
                if exc_type:
                    # Record exception if one occurred
                    self.span.record_exception(exc_val)
                    from traccia.tracer.span import SpanStatus
                    self.span.set_status(SpanStatus.ERROR, str(exc_val))
                
                self.span.end()
            except Exception:
                pass
        
        if self.token:
            try:
                from opentelemetry import context
                context.detach(self.token)
            except Exception:
                pass
        
        return False  # Don't suppress exceptions


def start_tracing(
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
    max_queue_size: int = 5000,
    max_export_batch_size: int = 512,
    schedule_delay_millis: int = 5000,
    exporter: Optional[Any] = None,
    use_otlp: bool = True,  # Use OTLP exporter by default
    transport=None,
    enable_patching: bool = True,
    enable_token_counting: bool = True,
    enable_costs: bool = True,
    pricing_override=None,
    pricing_refresh_seconds: Optional[int] = None,
    enable_console_exporter: bool = False,
    enable_file_exporter: bool = False,
    file_exporter_path: str = "traces.jsonl",
    reset_trace_file: bool = False,
    load_env: bool = True,
    enable_span_logging: bool = False,
    auto_instrument_tools: bool = False,
    tool_include: Optional[list] = None,
    max_tool_spans: int = 100,
    max_span_depth: int = 10,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    project_id: Optional[str] = None,
    project: Optional[str] = None,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    env: Optional[str] = None,
    debug: bool = False,
    attr_truncation_limit: Optional[int] = None,
    service_name: Optional[str] = None,
    max_spans_per_second: Optional[float] = None,  # Rate limiting
    max_block_ms: int = 100,  # Rate limiting block time
    openai_agents: Optional[bool] = None,  # Auto-install OpenAI Agents integration
    crewai: Optional[bool] = None,  # Auto-install CrewAI integration
    enable_metrics: bool = True,  # Enable metrics
    metrics_endpoint: Optional[str] = None,  # Metrics endpoint
    metrics_sample_rate: float = 1.0,  # Metrics sampling rate
) -> TracerProvider:
    """
    Initialize global tracing:
    - Builds OTLP exporter (or uses provided one)
    - Attaches BatchSpanProcessor with sampling and bounded queue
    - Registers monkey patches (OpenAI, Anthropic, requests)
    - Registers atexit shutdown hook
    """
    global _started, _active_processor, _init_method
    if _started:
        if _init_method == "init":
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "SDK was initialized with init(). "
                "Calling start_tracing() will not re-initialize. "
                "Use stop_tracing() first if you need to re-initialize."
            )
        return _get_provider()

    if load_env:
        sdk_config.load_dotenv()
    
    # Load config from environment (backward compatible)
    env_cfg = sdk_config.load_config_from_env()
    
    # Apply any explicit overrides
    if api_key:
        env_cfg['api_key'] = api_key
    if endpoint:
        env_cfg['endpoint'] = endpoint

    # Resolve agent configuration path automatically if not provided by env.
    agent_cfg_path = _resolve_agent_config_path()
    if agent_cfg_path:
        os.environ.setdefault("AGENT_DASHBOARD_AGENT_CONFIG", agent_cfg_path)

    provider = _get_provider()
    key = env_cfg.get("api_key") or api_key
    endpoint = env_cfg.get("endpoint") or endpoint
    try:
        sample_rate = float(env_cfg.get("sample_rate", sample_rate))
    except Exception:
        sample_rate = sample_rate

    # Validate exporter configuration when OTLP is disabled
    if not use_otlp and not (enable_console_exporter or enable_file_exporter):
        from traccia.errors import ConfigError
        raise ConfigError(
            "When use_otlp is false, you must enable either console or file exporter.",
            details={
                "use_otlp": use_otlp,
                "enable_console_exporter": enable_console_exporter,
                "enable_file_exporter": enable_file_exporter,
            },
        )

    # Set runtime config for auto-instrumentation
    runtime_config.set_auto_instrument_tools(auto_instrument_tools)
    runtime_config.set_tool_include(tool_include or [])
    runtime_config.set_max_tool_spans(max_tool_spans)
    runtime_config.set_max_span_depth(max_span_depth)
    runtime_config.set_session_id(session_id)
    runtime_config.set_user_id(user_id)
    runtime_config.set_tenant_id(_resolve_tenant_id(tenant_id))
    runtime_config.set_project_id(_resolve_project_id(project_id or project))
    # Resolve agent identity: explicit params > TRACCIA_* env vars
    _agent_id = agent_id or sdk_config.get_env_value("agent_id")
    _agent_name = agent_name or sdk_config.get_env_value("agent_name")
    _env = env or sdk_config.get_env_value("env")
    runtime_config.set_agent_id(_agent_id)
    runtime_config.set_agent_name(_agent_name)
    runtime_config.set_env(_env)
    runtime_config.set_debug(_resolve_debug(debug))
    # Log once which source provided agent identity (for debugging "unknown-agent" issues)
    import logging
    _log = logging.getLogger(__name__)
    if _agent_id or _agent_name or _env:
        _src = "params" if (agent_id or agent_name or env) else "TRACCIA_* env"
        _log.info("Traccia agent identity: id=%s name=%s env=%s (from %s)", _agent_id or "(none)", _agent_name or "(none)", _env or "(none)", _src)
    elif key or endpoint:
        _log.info("Traccia agent identity not set; traces may appear as unknown-agent in the UI. Set agent_id (and optionally env) via init() or TRACCIA_AGENT_ID / TRACCIA_ENV.")
    runtime_config.set_attr_truncation_limit(attr_truncation_limit)

    # Build resource attributes from runtime config
    # This ensures tenant.id, project.id, etc. are included in OTLP exports
    resource_attrs = {}
    
    # Set service.name - required for proper service identification in Tempo/Grafana
    # This prevents "unknown_service" from appearing
    from opentelemetry.semconv.resource import ResourceAttributes
    service_name_value = _resolve_service_name(service_name)
    resource_attrs[ResourceAttributes.SERVICE_NAME] = service_name_value
    
    if runtime_config.get_tenant_id():
        resource_attrs["tenant.id"] = runtime_config.get_tenant_id()
    if runtime_config.get_project_id():
        resource_attrs["project.id"] = runtime_config.get_project_id()
    if runtime_config.get_session_id():
        resource_attrs["session.id"] = runtime_config.get_session_id()
    if runtime_config.get_user_id():
        resource_attrs["user.id"] = runtime_config.get_user_id()
    if runtime_config.get_agent_id():
        resource_attrs["agent.id"] = runtime_config.get_agent_id()
    if runtime_config.get_agent_name():
        resource_attrs["agent.name"] = runtime_config.get_agent_name()
    if runtime_config.get_env():
        resource_attrs["environment"] = runtime_config.get_env()
        resource_attrs["env"] = runtime_config.get_env()
    if runtime_config.get_debug():
        resource_attrs["trace.debug"] = True
    
    # Update provider resource dict for exporter compatibility
    if resource_attrs:
        provider.resource.update(resource_attrs)
    
    # For OTLP, we need to recreate the provider with updated resource
    # since OTel Resource is immutable
    if resource_attrs and use_otlp:
        from opentelemetry.sdk.resources import Resource as OTelResource
        from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
        # Merge with existing resource attributes
        existing_resource = provider._otel_provider.resource
        existing_attrs = dict(existing_resource.attributes) if existing_resource.attributes else {}
        existing_attrs.update(resource_attrs)
        # Create new resource with merged attributes
        new_resource = OTelResource.create(existing_attrs)
        # Recreate OTel provider with updated resource
        provider._otel_provider = OTelTracerProvider(resource=new_resource)
        # Re-add any existing export processors to the new provider
        for proc in provider._export_processors:
            provider._otel_provider.add_span_processor(proc)

    # Use OTLP exporter by default for network export
    if exporter:
        network_exporter = exporter
    elif use_otlp:
        # Use OTLP exporter (OpenTelemetry standard)
        network_exporter = OTLPExporter(
            endpoint=endpoint or DEFAULT_OTLP_TRACE_ENDPOINT,
            api_key=key,
        )
    else:
        # When use_otlp is False, rely on console/file exporters only
        network_exporter = None

    if enable_console_exporter:
        network_exporter = _combine_exporters(network_exporter, ConsoleExporter())

    if enable_file_exporter:
        # If reset_trace_file is True, clear the file when start_tracing is called
        if reset_trace_file:
            try:
                with open(file_exporter_path, "w", encoding="utf-8") as f:
                    pass  # Truncate file to empty
            except Exception:
                pass  # Silently fail if file cannot be cleared
        network_exporter = _combine_exporters(
            network_exporter,
            FileExporter(file_path=file_exporter_path, reset_on_start=False)
        )

    sampler = Sampler(sample_rate)
    # Use the sampler at trace start (head sampling) and also to make the
    # batch processor respect trace_flags.
    try:
        provider.set_sampler(sampler)
    except Exception:
        pass

    # Ordering matters: enrich spans before batching/export.
    if enable_token_counting:
        provider.add_span_processor(TokenCountingProcessor())
    cost_processor = None
    if enable_costs:
        pricing_table, pricing_source = pricing_config.load_pricing_with_source(pricing_override)
        cost_processor = CostAnnotatingProcessor(
            pricing_table=pricing_table, pricing_source=pricing_source
        )
        provider.add_span_processor(cost_processor)
    if enable_span_logging:
        provider.add_span_processor(LoggingSpanProcessor())
    # Agent enrichment: use init-time identity as default so span-level overrides win over resource/defaults.
    provider.add_span_processor(
        AgentEnrichmentProcessor(
            agent_config_path=os.getenv("AGENT_DASHBOARD_AGENT_CONFIG"),
            default_agent_id=runtime_config.get_agent_id(),
            default_agent_name=runtime_config.get_agent_name(),
            default_env=runtime_config.get_env(),
        )
    )

    # For OTLP exporter, use OTel's BatchSpanProcessor directly
    # For non-OTLP exporters (console/file), use our custom BatchSpanProcessor
    if use_otlp and isinstance(network_exporter, OTLPExporter) and hasattr(network_exporter, '_otel_exporter'):
        # Use OTel's BatchSpanProcessor for OTLP export
        from opentelemetry.sdk.trace.export import BatchSpanProcessor as OTelBatchSpanProcessor
        otel_processor = OTelBatchSpanProcessor(
            network_exporter._otel_exporter,
            max_queue_size=max_queue_size,
            max_export_batch_size=max_export_batch_size,
            schedule_delay_millis=schedule_delay_millis,
        )
        
        # Wrap with rate limiting if configured
        if max_spans_per_second is not None and max_spans_per_second > 0:
            from traccia.processors.rate_limiter import RateLimitingSpanProcessor
            rate_limited_processor = RateLimitingSpanProcessor(
                next_processor=otel_processor,
                max_spans_per_second=max_spans_per_second,
                max_block_ms=max_block_ms,
            )
            provider._otel_provider.add_span_processor(rate_limited_processor)
        else:
            provider._otel_provider.add_span_processor(otel_processor)
        _active_processor = None  # OTel handles this
    else:
        # Use our custom BatchSpanProcessor for non-OTLP exporters
        processor = BatchSpanProcessor(
            exporter=network_exporter,
            sampler=sampler,
            max_queue_size=max_queue_size,
            max_export_batch_size=max_export_batch_size,
            schedule_delay_millis=schedule_delay_millis,
        )
        
        # Wrap with rate limiting if configured
        if max_spans_per_second is not None and max_spans_per_second > 0:
            from traccia.processors.rate_limiter import RateLimitingSpanProcessor
            rate_limited_processor = RateLimitingSpanProcessor(
                next_processor=processor,
                max_spans_per_second=max_spans_per_second,
                max_block_ms=max_block_ms,
            )
            provider.add_span_processor(rate_limited_processor)
            _active_processor = rate_limited_processor
        else:
            provider.add_span_processor(processor)
            _active_processor = processor

    if _active_processor:
        _register_shutdown(provider, _active_processor)
    _start_pricing_refresh(cost_processor, pricing_override, pricing_refresh_seconds)

    # Initialize metrics if enabled
    if enable_metrics:
        _initialize_metrics(
            endpoint=endpoint,
            api_key=key,
            metrics_endpoint=metrics_endpoint,
            metrics_sample_rate=metrics_sample_rate,
            service_name=_resolve_service_name(service_name),
            agent_id=runtime_config.get_agent_id(),
            agent_name=runtime_config.get_agent_name(),
            env=runtime_config.get_env(),
        )

    # Auto-instrument in-repo functions/tools if enabled
    if auto_instrument_tools and tool_include:
        try:
            auto_instrumentation.instrument_functions(tool_include or [])
        except Exception:
            pass

    if enable_patching:
        try:
            patch_openai()
        except Exception:
            pass
        try:
            patch_anthropic()
        except Exception:
            pass
        try:
            patch_requests()
        except Exception:
            pass

    _started = True
    if _init_method is None:
        _init_method = "start_tracing"

    # Auto-install framework integrations (OpenAI Agents, CrewAI, etc.)
    _install_integrations(openai_agents_flag=openai_agents, crewai_flag=crewai)

    return provider


def stop_tracing(flush_timeout: Optional[float] = None) -> None:
    """Force flush and shutdown registered processors and provider."""
    global _started, _init_method, _auto_trace_context
    
    # End auto-trace if active
    if _auto_trace_context:
        _cleanup_auto_trace()
    
    _stop_pricing_refresh()
    provider = _get_provider()
    if _active_processor:
        try:
            _active_processor.force_flush(timeout=flush_timeout)
        finally:
            _active_processor.shutdown()
    provider.shutdown()
    _started = False
    _init_method = None


def _register_shutdown(provider: TracerProvider, processor: Optional[BatchSpanProcessor]) -> None:
    global _registered_shutdown
    if _registered_shutdown:
        return

    def _cleanup():
        try:
            if processor:
                processor.force_flush()
                processor.shutdown()
        finally:
            provider.shutdown()

    atexit.register(_cleanup)
    _registered_shutdown = True


def _resolve_service_name(service_name: Optional[str]) -> str:
    """Resolve service.name using override, env, or inferred entrypoint."""
    if service_name:
        return service_name
    env_name = os.getenv("OTEL_SERVICE_NAME") or os.getenv("SERVICE_NAME")
    if env_name:
        return env_name
    # Use current working directory name
    cwd_name = Path.cwd().name
    if cwd_name:
        return cwd_name
    # Infer from entry script if available (e.g., "app.py" -> "app")
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 not in ("-c", "-m"):
        script_name = Path(argv0).name
        if script_name:
            return Path(script_name).stem or script_name
    return "traccia_app"


def _get_provider() -> TracerProvider:
    import traccia

    return traccia.get_tracer_provider()


def _resolve_agent_config_path() -> Optional[str]:
    """
    Locate agent_config.json for users automatically:
      1) Respect AGENT_DASHBOARD_AGENT_CONFIG if set and file exists
      2) Use ./agent_config.json from current working directory if present
      3) Try to find agent_config.json adjacent to the first non-sdk caller
    """
    env_path = os.getenv("AGENT_DASHBOARD_AGENT_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path.resolve())

    cwd_path = Path.cwd() / "agent_config.json"
    if cwd_path.exists():
        return str(cwd_path.resolve())

    try:
        for frame in inspect.stack():
            frame_path = Path(frame.filename)
            # Skip SDK internal files
            if "traccia" in frame_path.parts:
                continue
            candidate = frame_path.parent / "agent_config.json"
            if candidate.exists():
                return str(candidate.resolve())
    except Exception:
        return None
    return None


def _resolve_debug(cli_value: bool) -> bool:
    raw = os.getenv("AGENT_DASHBOARD_DEBUG")
    if raw is None:
        return bool(cli_value)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_tenant_id(cli_value: Optional[str]) -> str:
    return (
        cli_value
        or os.getenv("AGENT_DASHBOARD_TENANT_ID")
        or "default-tenant"
    )


def _resolve_project_id(cli_value: Optional[str]) -> str:
    return cli_value or os.getenv("AGENT_DASHBOARD_PROJECT_ID") or "default-project"


def _combine_exporters(primary, secondary):
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    class _Multi:
        def export(self, spans):
            ok1 = primary.export(spans)
            ok2 = secondary.export(spans)
            return ok1 and ok2

        def shutdown(self):
            for exp in (primary, secondary):
                if hasattr(exp, "shutdown"):
                    exp.shutdown()

    return _Multi()


_pricing_refresh_stop: Optional[threading.Event] = None
_pricing_refresh_thread: Optional[threading.Thread] = None


def _start_pricing_refresh(cost_processor: Optional[CostAnnotatingProcessor], override, interval: Optional[int]) -> None:
    global _pricing_refresh_stop, _pricing_refresh_thread
    if not cost_processor or not interval or interval <= 0:
        return
    _pricing_refresh_stop = threading.Event()

    def _loop():
        while not _pricing_refresh_stop.is_set():
            time.sleep(interval)
            if _pricing_refresh_stop.is_set():
                break
            try:
                table, source = pricing_config.load_pricing_with_source(override)
                cost_processor.update_pricing_table(table, pricing_source=source)
            except Exception:
                continue

    _pricing_refresh_thread = threading.Thread(target=_loop, daemon=True)
    _pricing_refresh_thread.start()


def _stop_pricing_refresh() -> None:
    if _pricing_refresh_stop:
        _pricing_refresh_stop.set()
    if _pricing_refresh_thread:
        _pricing_refresh_thread.join(timeout=1)


def _initialize_metrics(
    endpoint: Optional[str],
    api_key: Optional[str],
    metrics_endpoint: Optional[str],
    metrics_sample_rate: float,
    service_name: str,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    env: Optional[str] = None,
) -> None:
    """Initialize OpenTelemetry MeterProvider and metrics."""
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "OpenTelemetry metrics SDK not installed. Metrics disabled. "
            "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )
        return

    # Determine metrics endpoint: default to {traces_base}/v2/metrics
    if not metrics_endpoint:
        if endpoint:
            # Replace /v1/traces or /v2/traces with /v2/metrics
            if endpoint.endswith("/v1/traces"):
                metrics_endpoint = endpoint.replace("/v1/traces", "/v2/metrics")
            elif endpoint.endswith("/v2/traces"):
                metrics_endpoint = endpoint.replace("/v2/traces", "/v2/metrics")
            else:
                # Append /v2/metrics to base
                metrics_endpoint = endpoint.rstrip("/") + "/v2/metrics"
        else:
            # Use default endpoint
            metrics_endpoint = DEFAULT_OTLP_TRACE_ENDPOINT.replace("/v2/traces", "/v2/metrics")

    # Create resource with service.name and agent identity (align with traces)
    resource_attrs = {"service.name": service_name}
    if agent_id:
        resource_attrs["agent.id"] = agent_id
    if agent_name:
        resource_attrs["agent.name"] = agent_name
    if env:
        resource_attrs["environment"] = env
        resource_attrs["env"] = env
    resource = Resource(attributes=resource_attrs)

    # Create OTLP metric exporter with API key
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    metric_exporter = OTLPMetricExporter(
        endpoint=metrics_endpoint,
        headers=headers
    )

    # Create metric reader with periodic export
    metric_reader = PeriodicExportingMetricReader(
        exporter=metric_exporter,
        export_interval_millis=5000  # Export every 5 seconds
    )

    # Create MeterProvider
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )
    metrics.set_meter_provider(meter_provider)

    # Create meter and standard metrics
    meter = metrics.get_meter("traccia", "1.0.0")
    standard_metrics = StandardMetrics.create_standard_metrics(meter)

    # Create metrics recorder
    recorder = MetricsRecorder(standard_metrics, sample_rate=metrics_sample_rate)
    recorder._meter = meter  # Store meter for custom metrics
    
    # Set global recorder
    set_global_recorder(recorder)

    import logging
    logging.getLogger(__name__).info(f"Metrics initialized: endpoint={metrics_endpoint}, sample_rate={metrics_sample_rate}")


def _install_integrations(
    openai_agents_flag: Optional[bool],
    crewai_flag: Optional[bool],
) -> None:
    """
    Auto-install framework integrations so that init() and start_tracing()
    behave equivalently with respect to supported frameworks.
    """
    # OpenAI Agents SDK integration
    try:
        from traccia import runtime_config as _rc
    except Exception:
        _rc = None

    # Determine OpenAI Agents enablement: explicit flag > runtime config > default True
    openai_enabled = True
    if _rc is not None:
        openai_enabled = _rc.get_config_value("openai_agents", True)
    if openai_agents_flag is not None:
        openai_enabled = bool(openai_agents_flag)

    if openai_enabled:
        try:
            from traccia.integrations.openai_agents import install as install_openai_agents
            install_openai_agents(enabled=True)
        except Exception:
            # Agents SDK not installed or error during install, skip silently
            pass

    # Determine CrewAI enablement: explicit flag > runtime config > default True
    crewai_enabled = True
    if _rc is not None:
        crewai_enabled = _rc.get_config_value("crewai", True)
    if crewai_flag is not None:
        crewai_enabled = bool(crewai_flag)

    if crewai_enabled:
        try:
            from traccia.integrations.crewai import install as install_crewai
            install_crewai(enabled=True)
        except Exception:
            # CrewAI not installed or error during install, skip silently
            pass

