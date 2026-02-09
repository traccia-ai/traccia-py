"""Configuration management with Pydantic models and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl

from traccia.errors import ConfigError, ValidationError


# Environment variable mapping
ENV_VAR_MAPPING = {
    # Tracing config
    "api_key": ["TRACCIA_API_KEY", "AGENT_DASHBOARD_API_KEY"],
    "endpoint": ["TRACCIA_ENDPOINT", "AGENT_DASHBOARD_ENDPOINT"],
    "sample_rate": ["TRACCIA_SAMPLE_RATE", "AGENT_DASHBOARD_SAMPLE_RATE"],
    "auto_start_trace": ["TRACCIA_AUTO_START_TRACE", "AGENT_DASHBOARD_AUTO_START_TRACE"],
    "auto_trace_name": ["TRACCIA_AUTO_TRACE_NAME"],
    "use_otlp": ["TRACCIA_USE_OTLP"],
    "service_name": ["TRACCIA_SERVICE_NAME"],
    
    # Exporter config
    "enable_console": ["TRACCIA_ENABLE_CONSOLE", "AGENT_DASHBOARD_ENABLE_CONSOLE_EXPORTER"],
    "enable_file": ["TRACCIA_ENABLE_FILE", "AGENT_DASHBOARD_ENABLE_FILE_EXPORTER"],
    "file_exporter_path": ["TRACCIA_FILE_PATH"],
    "reset_trace_file": ["TRACCIA_RESET_TRACE_FILE"],
    
    # Instrumentation config
    "enable_patching": ["TRACCIA_ENABLE_PATCHING", "AGENT_DASHBOARD_ENABLE_PATCHING"],
    "enable_token_counting": ["TRACCIA_ENABLE_TOKEN_COUNTING", "AGENT_DASHBOARD_ENABLE_TOKEN_COUNTING"],
    "enable_costs": ["TRACCIA_ENABLE_COSTS", "AGENT_DASHBOARD_ENABLE_COSTS"],
    "auto_instrument_tools": ["TRACCIA_AUTO_INSTRUMENT_TOOLS"],
    "max_tool_spans": ["TRACCIA_MAX_TOOL_SPANS"],
    "max_span_depth": ["TRACCIA_MAX_SPAN_DEPTH"],
    "openai_agents": ["TRACCIA_OPENAI_AGENTS"],
    "crewai": ["TRACCIA_CREWAI"],
    
    # Rate limiting & Batching
    "max_spans_per_second": ["TRACCIA_MAX_SPANS_PER_SECOND"],
    "max_queue_size": ["TRACCIA_MAX_QUEUE_SIZE"],
    "max_block_ms": ["TRACCIA_MAX_BLOCK_MS"],
    "max_export_batch_size": ["TRACCIA_MAX_EXPORT_BATCH_SIZE"],
    "schedule_delay_millis": ["TRACCIA_SCHEDULE_DELAY_MILLIS"],
    
    # Runtime metadata
    "session_id": ["TRACCIA_SESSION_ID"],
    "user_id": ["TRACCIA_USER_ID"],
    "tenant_id": ["TRACCIA_TENANT_ID"],
    "project_id": ["TRACCIA_PROJECT_ID"],
    "agent_id": ["TRACCIA_AGENT_ID", "AGENT_DASHBOARD_AGENT_ID"],
    
    # Logging
    "debug": ["TRACCIA_DEBUG"],
    "enable_span_logging": ["TRACCIA_ENABLE_SPAN_LOGGING"],
    
    # Metrics
    "enable_metrics": ["TRACCIA_ENABLE_METRICS"],
    "metrics_endpoint": ["TRACCIA_METRICS_ENDPOINT"],
    "metrics_sample_rate": ["TRACCIA_METRICS_SAMPLE_RATE"],
    
    # Advanced
    "attr_truncation_limit": ["TRACCIA_ATTR_TRUNCATION_LIMIT"],
}


class TracingConfig(BaseModel):
    """Tracing configuration section."""
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (required for SaaS, optional for open-source)"
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint URL for trace ingestion"
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling rate (0.0 to 1.0)"
    )
    auto_start_trace: bool = Field(
        default=True,
        description="Automatically start a root trace on init"
    )
    auto_trace_name: str = Field(
        default="root",
        description="Name for the auto-started root trace"
    )
    use_otlp: bool = Field(
        default=True,
        description="Use OTLP exporter (set to false for console/file exporters)"
    )
    service_name: Optional[str] = Field(
        default=None,
        description="Service name for the application"
    )


class ExporterConfig(BaseModel):
    """Exporter configuration section."""
    
    enable_console: bool = Field(
        default=False,
        description="Enable console exporter for debugging"
    )
    enable_file: bool = Field(
        default=False,
        description="Enable file exporter to write traces to local file"
    )
    file_exporter_path: str = Field(
        default="traces.jsonl",
        description="File path for file exporter"
    )
    reset_trace_file: bool = Field(
        default=False,
        description="Reset/clear trace file on initialization"
    )
    
    @model_validator(mode='after')
    def check_single_exporter(self) -> 'ExporterConfig':
        """Ensure only one exporter is enabled at a time."""
        enabled_count = sum([
            self.enable_console,
            self.enable_file,
        ])
        if enabled_count > 1:
            raise ValidationError(
                "Only one exporter can be enabled at a time. "
                "Choose either console or file exporter (OTLP is controlled by use_otlp in tracing section).",
                details={
                    "enable_console": self.enable_console,
                    "enable_file": self.enable_file,
                }
            )
        return self


class InstrumentationConfig(BaseModel):
    """Instrumentation configuration section."""
    
    enable_patching: bool = Field(
        default=True,
        description="Auto-patch popular libraries (OpenAI, Anthropic, requests)"
    )
    enable_token_counting: bool = Field(
        default=True,
        description="Count tokens for LLM calls"
    )
    enable_costs: bool = Field(
        default=True,
        description="Calculate costs for LLM calls"
    )
    auto_instrument_tools: bool = Field(
        default=False,
        description="Automatically instrument tool calls"
    )
    max_tool_spans: int = Field(
        default=100,
        gt=0,
        description="Maximum number of tool spans to create"
    )
    max_span_depth: int = Field(
        default=10,
        gt=0,
        description="Maximum depth of nested spans"
    )
    openai_agents: bool = Field(
        default=True,
        description="Auto-install OpenAI Agents SDK integration when available"
    )
    crewai: bool = Field(
        default=True,
        description="Auto-install CrewAI integration when available"
    )


class RateLimitConfig(BaseModel):
    """Rate limiting and batching configuration section."""
    
    max_spans_per_second: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum spans per second (None = unlimited)"
    )
    max_queue_size: int = Field(
        default=5000,
        gt=0,
        description="Maximum queue size for buffered spans"
    )
    max_block_ms: int = Field(
        default=100,
        ge=0,
        description="Maximum milliseconds to block before dropping spans"
    )
    max_export_batch_size: int = Field(
        default=512,
        gt=0,
        description="Maximum number of spans in a single export batch"
    )
    schedule_delay_millis: int = Field(
        default=5000,
        gt=0,
        description="Delay in milliseconds between export batches"
    )


class LoggingConfig(BaseModel):
    """Logging configuration section."""
    
    debug: bool = Field(
        default=False,
        description="Enable debug logging"
    )
    enable_span_logging: bool = Field(
        default=False,
        description="Enable span-level logging for debugging"
    )


class MetricsConfig(BaseModel):
    """Metrics configuration section."""
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable OpenTelemetry metrics emission"
    )
    metrics_endpoint: Optional[str] = Field(
        default=None,
        description="Metrics endpoint URL (defaults to {traces_base}/v2/metrics)"
    )
    metrics_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Metrics sampling rate (0.0 to 1.0, default: 1.0 = 100%)"
    )


class RuntimeConfig(BaseModel):
    """Runtime metadata configuration section."""
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for grouping traces"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for the current session"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant/organization identifier"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Project identifier"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent identifier for the current session"
    )


class AdvancedConfig(BaseModel):
    """Advanced configuration options."""
    
    attr_truncation_limit: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum length for attribute values (None = no limit)"
    )


class TracciaConfig(BaseModel):
    """
    Complete Traccia SDK configuration.
    
    This model validates and merges configuration from multiple sources:
    1. Config file (traccia.toml)
    2. Environment variables
    3. Explicit parameters
    """
    
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    exporters: ExporterConfig = Field(default_factory=ExporterConfig)
    instrumentation: InstrumentationConfig = Field(default_factory=InstrumentationConfig)
    rate_limiting: RateLimitConfig = Field(default_factory=RateLimitConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    
    @model_validator(mode='after')
    def validate_complete_config(self) -> 'TracciaConfig':
        """Validate the complete configuration for conflicts."""
        # If OTLP is disabled, at least one other exporter must be enabled
        if not self.tracing.use_otlp:
            if not (self.exporters.enable_console or self.exporters.enable_file):
                raise ConfigError(
                    "When use_otlp is false, you must enable either console or file exporter.",
                    details={
                        "use_otlp": self.tracing.use_otlp,
                        "enable_console": self.exporters.enable_console,
                        "enable_file": self.exporters.enable_file,
                    }
                )
        
        return self
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for backward compatibility."""
        return {
            # Tracing
            "api_key": self.tracing.api_key,
            "endpoint": self.tracing.endpoint,
            "sample_rate": self.tracing.sample_rate,
            "auto_start_trace": self.tracing.auto_start_trace,
            "auto_trace_name": self.tracing.auto_trace_name,
            "use_otlp": self.tracing.use_otlp,
            "service_name": self.tracing.service_name,
            # Exporters
            "enable_console": self.exporters.enable_console,
            "enable_file": self.exporters.enable_file,
            "file_exporter_path": self.exporters.file_exporter_path,
            "reset_trace_file": self.exporters.reset_trace_file,
            # Instrumentation
            "enable_patching": self.instrumentation.enable_patching,
            "enable_token_counting": self.instrumentation.enable_token_counting,
            "enable_costs": self.instrumentation.enable_costs,
            "auto_instrument_tools": self.instrumentation.auto_instrument_tools,
            "max_tool_spans": self.instrumentation.max_tool_spans,
            "max_span_depth": self.instrumentation.max_span_depth,
            "openai_agents": self.instrumentation.openai_agents,
            "crewai": self.instrumentation.crewai,
            # Rate limiting & Batching
            "max_spans_per_second": self.rate_limiting.max_spans_per_second,
            "max_queue_size": self.rate_limiting.max_queue_size,
            "max_block_ms": self.rate_limiting.max_block_ms,
            "max_export_batch_size": self.rate_limiting.max_export_batch_size,
            "schedule_delay_millis": self.rate_limiting.schedule_delay_millis,
            # Metrics
            "enable_metrics": self.metrics.enable_metrics,
            "metrics_endpoint": self.metrics.metrics_endpoint,
            "metrics_sample_rate": self.metrics.metrics_sample_rate,
            # Runtime
            "session_id": self.runtime.session_id,
            "user_id": self.runtime.user_id,
            "tenant_id": self.runtime.tenant_id,
            "project_id": self.runtime.project_id,
            "agent_id": self.runtime.agent_id,
            # Logging
            "debug": self.logging.debug,
            "enable_span_logging": self.logging.enable_span_logging,
            # Advanced
            "attr_truncation_limit": self.advanced.attr_truncation_limit,
        }


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dependency)."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Fail silently; this loader is best-effort.
        return


def find_config_file() -> Optional[str]:
    """
    Find traccia.toml config file in standard locations.
    
    Lookup order:
    1. ./traccia.toml (current directory)
    2. ~/.traccia/config.toml (user home)
    
    Returns:
        Path to config file if found, None otherwise
    """
    # Check current directory
    cwd_config = Path.cwd() / "traccia.toml"
    if cwd_config.exists():
        return str(cwd_config)
    
    # Check user home directory
    home_config = Path.home() / ".traccia" / "config.toml"
    if home_config.exists():
        return str(home_config)
    
    return None


def load_toml_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.
    
    Args:
        path: Path to the TOML config file
        
    Returns:
        Dictionary with nested config structure
    """
    if not os.path.exists(path):
        return {}
    
    try:
        # Try to import tomli for Python 3.11+, fall back to toml
        try:
            import tomli as toml_lib
            with open(path, "rb") as f:
                data = toml_lib.load(f)
        except ImportError:
            try:
                import toml as toml_lib  # type: ignore
                with open(path, "r", encoding="utf-8") as f:
                    data = toml_lib.load(f)
            except ImportError:
                raise ConfigError(
                    "No TOML library available. Install tomli or toml: pip install tomli"
                )
        
        return data
    
    except Exception as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(f"Failed to load config file: {e}")


def get_env_value(config_key: str) -> Optional[str]:
    """
    Get environment variable value for a config key.
    
    Tries multiple environment variable names in order of preference.
    """
    env_vars = ENV_VAR_MAPPING.get(config_key, [])
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value is not None:
            return value
    return None


def load_config_from_env(flat: bool = False) -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Args:
        flat: If True, return flat dictionary for backward compatibility
    
    Returns:
        Dictionary of config values from environment (nested structure by default)
    """
    env_config = {
        "tracing": {},
        "exporters": {},
        "instrumentation": {},
        "rate_limiting": {},
        "metrics": {},
        "runtime": {},
        "logging": {},
        "advanced": {},
    }
    
    # Tracing section
    for key in ["api_key", "endpoint", "sample_rate", "auto_start_trace", "auto_trace_name", "use_otlp", "service_name"]:
        value = get_env_value(key)
        if value is not None:
            # Convert string to appropriate type
            if key in ["auto_start_trace", "use_otlp"]:
                env_config["tracing"][key] = value.lower() in ("true", "1", "yes")
            elif key == "sample_rate":
                try:
                    env_config["tracing"][key] = float(value)
                except ValueError:
                    raise ConfigError(f"Invalid sample_rate value: {value}. Must be a float between 0.0 and 1.0.")
            else:
                env_config["tracing"][key] = value
    
    # Exporters section
    for key in ["enable_console", "enable_file", "file_exporter_path", "reset_trace_file"]:
        value = get_env_value(key)
        if value is not None:
            if key in ["enable_console", "enable_file", "reset_trace_file"]:
                env_config["exporters"][key] = value.lower() in ("true", "1", "yes")
            else:
                env_config["exporters"][key] = value
    
    # Instrumentation section
    for key in ["enable_patching", "enable_token_counting", "enable_costs", "auto_instrument_tools", "openai_agents", "crewai"]:
        value = get_env_value(key)
        if value is not None:
            env_config["instrumentation"][key] = value.lower() in ("true", "1", "yes")
    
    for key in ["max_tool_spans", "max_span_depth"]:
        value = get_env_value(key)
        if value is not None:
            try:
                env_config["instrumentation"][key] = int(value)
            except ValueError:
                raise ConfigError(f"Invalid {key} value: {value}. Must be an integer.")
    
    # Rate limiting section
    for key in ["max_spans_per_second"]:
        value = get_env_value(key)
        if value is not None:
            try:
                env_config["rate_limiting"][key] = float(value) if value else None
            except ValueError:
                raise ConfigError(f"Invalid {key} value: {value}. Must be a number.")
    
    for key in ["max_queue_size", "max_block_ms", "max_export_batch_size", "schedule_delay_millis"]:
        value = get_env_value(key)
        if value is not None:
            try:
                env_config["rate_limiting"][key] = int(value)
            except ValueError:
                raise ConfigError(f"Invalid {key} value: {value}. Must be a number.")
    
    # Metrics section
    for key in ["enable_metrics"]:
        value = get_env_value(key)
        if value is not None:
            env_config["metrics"][key] = value.lower() in ("true", "1", "yes")
    
    value = get_env_value("metrics_endpoint")
    if value is not None:
        env_config["metrics"]["metrics_endpoint"] = value
    
    value = get_env_value("metrics_sample_rate")
    if value is not None:
        try:
            env_config["metrics"]["metrics_sample_rate"] = float(value)
        except ValueError:
            raise ConfigError(f"Invalid metrics_sample_rate value: {value}. Must be a float between 0.0 and 1.0.")
    
    # Runtime section
    for key in ["session_id", "user_id", "tenant_id", "project_id", "agent_id"]:
        value = get_env_value(key)
        if value is not None:
            env_config["runtime"][key] = value
    
    # Logging section
    for key in ["debug", "enable_span_logging"]:
        value = get_env_value(key)
        if value is not None:
            env_config["logging"][key] = value.lower() in ("true", "1", "yes")
    
    # Advanced section
    value = get_env_value("attr_truncation_limit")
    if value is not None:
        try:
            env_config["advanced"]["attr_truncation_limit"] = int(value)
        except ValueError:
            raise ConfigError(f"Invalid attr_truncation_limit value: {value}. Must be an integer.")
    
    # Remove empty sections
    nested_result = {k: v for k, v in env_config.items() if v}
    
    # Flatten if requested (for backward compatibility)
    if flat:
        flat_result = {}
        for section, values in nested_result.items():
            flat_result.update(values)
        return flat_result
    
    return nested_result


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two config dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> TracciaConfig:
    """
    Load and validate Traccia configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Explicit overrides (passed as parameters)
    2. Environment variables
    3. Config file (./traccia.toml or ~/.traccia/config.toml)
    4. Defaults
    
    Args:
        config_file: Optional explicit path to config file
        overrides: Optional dict of explicit parameter overrides
        
    Returns:
        Validated TracciaConfig instance
        
    Raises:
        ConfigError: If configuration is invalid or conflicting
    """
    merged_config: Dict[str, Any] = {}
    
    # 1. Load from config file (lowest priority)
    if config_file:
        file_config = load_toml_config(config_file)
        merged_config = merge_configs(merged_config, file_config)
    else:
        # Try to find config file automatically
        found_config = find_config_file()
        if found_config:
            file_config = load_toml_config(found_config)
            merged_config = merge_configs(merged_config, file_config)
    
    # 2. Override with environment variables (medium priority)
    env_config = load_config_from_env()
    merged_config = merge_configs(merged_config, env_config)
    
    # 3. Override with explicit parameters (highest priority)
    if overrides:
        merged_config = merge_configs(merged_config, overrides)
    
    # 4. Create and validate Pydantic model
    try:
        return TracciaConfig(**merged_config)
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}")


def validate_config(
    config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> tuple[bool, str, Optional[TracciaConfig]]:
    """
    Validate configuration without loading it.
    
    Used by the `traccia doctor` CLI command.
    
    Args:
        config_file: Optional explicit path to config file
        overrides: Optional dict of explicit parameter overrides
        
    Returns:
        Tuple of (is_valid, message, config_or_none)
    """
    try:
        config = load_config(config_file=config_file, overrides=overrides)
        return True, "Configuration is valid", config
    except ConfigError as e:
        return False, f"Configuration error: {e}", None
    except Exception as e:
        return False, f"Unexpected error: {e}", None


# Backward compatibility functions
def load_config_with_priority(
    config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    
    Returns flattened config dictionary instead of Pydantic model.
    Accepts flat overrides and converts them to nested format.
    """
    # Convert flat overrides to nested format
    nested_overrides = None
    if overrides:
        nested_overrides = {
            "tracing": {}, 
            "exporters": {}, 
            "instrumentation": {}, 
            "rate_limiting": {},
            "runtime": {},
            "logging": {},
            "advanced": {}
        }
        
        # Map flat keys to nested structure
        flat_to_nested = {
            # Tracing
            "api_key": ("tracing", "api_key"),
            "endpoint": ("tracing", "endpoint"),
            "sample_rate": ("tracing", "sample_rate"),
            "auto_start_trace": ("tracing", "auto_start_trace"),
            "auto_trace_name": ("tracing", "auto_trace_name"),
            "use_otlp": ("tracing", "use_otlp"),
            "service_name": ("tracing", "service_name"),
            # Exporters
            "enable_console": ("exporters", "enable_console"),
            "enable_file": ("exporters", "enable_file"),
            "file_exporter_path": ("exporters", "file_exporter_path"),
            "reset_trace_file": ("exporters", "reset_trace_file"),
            # Instrumentation
            "enable_patching": ("instrumentation", "enable_patching"),
            "enable_token_counting": ("instrumentation", "enable_token_counting"),
            "enable_costs": ("instrumentation", "enable_costs"),
            "auto_instrument_tools": ("instrumentation", "auto_instrument_tools"),
            "max_tool_spans": ("instrumentation", "max_tool_spans"),
            "max_span_depth": ("instrumentation", "max_span_depth"),
            # Rate limiting & Batching
            "max_spans_per_second": ("rate_limiting", "max_spans_per_second"),
            "max_queue_size": ("rate_limiting", "max_queue_size"),
            "max_block_ms": ("rate_limiting", "max_block_ms"),
            "max_export_batch_size": ("rate_limiting", "max_export_batch_size"),
            "schedule_delay_millis": ("rate_limiting", "schedule_delay_millis"),
            # Metrics
            "enable_metrics": ("metrics", "enable_metrics"),
            "metrics_endpoint": ("metrics", "metrics_endpoint"),
            "metrics_sample_rate": ("metrics", "metrics_sample_rate"),
            # Runtime
            "session_id": ("runtime", "session_id"),
            "user_id": ("runtime", "user_id"),
            "tenant_id": ("runtime", "tenant_id"),
            "project_id": ("runtime", "project_id"),
            "agent_id": ("runtime", "agent_id"),
            # Logging
            "debug": ("logging", "debug"),
            "enable_span_logging": ("logging", "enable_span_logging"),
            # Advanced
            "attr_truncation_limit": ("advanced", "attr_truncation_limit"),
        }
        
        for flat_key, value in overrides.items():
            if flat_key in flat_to_nested:
                section, nested_key = flat_to_nested[flat_key]
                nested_overrides[section][nested_key] = value
    
    config = load_config(config_file=config_file, overrides=nested_overrides)
    return config.to_flat_dict()
