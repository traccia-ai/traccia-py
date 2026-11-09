"""Python SDK entrypoint for the agent tracing library."""

from traccia.auto import start_tracing, stop_tracing, init, trace, end_auto_trace, force_flush
from traccia.tracer import TracerProvider
from traccia.instrumentation.decorator import observe
from traccia.governance import govern, AgentBlockedError, ApprovalPending, GovernanceConfigError, IdempotencyRepeat, pending_tool_result, check_policy
from traccia.governance.idempotency import DefiniteFailure, Unkeyable, idempotency_key
from traccia.context import (
    get_current_span,
    run_with_span,
    run_with_span_async,
    SpanScope,
    span_scope,
)
from traccia import metrics
from traccia.identity import AgentIdentity
from traccia import runtime_config
from traccia.prompts import (
    load_prompt,
    prefetch_prompts,
    LoadedPrompt,
    CompileError,
    PromptFetchError,
)
from traccia.eval import evaluate, EvaluateResult, EvaluateError

# Version exposure
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("traccia")
    except PackageNotFoundError:
        __version__ = "0.1.0"  # Fallback version
except ImportError:
    __version__ = "0.1.0"  # Fallback for Python < 3.8

# Global tracer provider (OpenTelemetry).
_global_tracer_provider = TracerProvider()


def get_tracer(name: str = "default"):
    """Fetch a tracer from the global provider."""
    return _global_tracer_provider.get_tracer(name)


def span(name: str, attributes: dict = None):
    """Start a span on the default tracer, as a context manager.

    Usage::

        with span("my_operation", {"key": "value"}) as s:
            s.set_attribute("another", "attr")
    """
    tracer = get_tracer()
    return tracer.start_as_current_span(name, attributes=attributes)


def set_tracer_provider(provider: TracerProvider) -> None:
    """Override the global tracer provider (primarily for tests or customization)."""
    global _global_tracer_provider
    _global_tracer_provider = provider


def get_tracer_provider() -> TracerProvider:
    return _global_tracer_provider


def get_agent_identity() -> AgentIdentity:
    """Return the current agent identity resolved by the SDK (init params or TRACCIA_* env)."""
    return AgentIdentity(
        id=runtime_config.get_agent_id(),
        name=runtime_config.get_agent_name(),
        env=runtime_config.get_env(),
        project=runtime_config.get_project_id(),
    )


__all__ = [
    "__version__",
    "get_tracer",
    "get_tracer_provider",
    "set_tracer_provider",
    "get_agent_identity",
    "AgentIdentity",
    "start_tracing",
    "stop_tracing",
    "init",
    "trace",
    "end_auto_trace",
    "force_flush",
    "span",
    "span_scope",
    "SpanScope",
    "get_current_span",
    "run_with_span",
    "run_with_span_async",
    "observe",
    "govern",
    "AgentBlockedError",
    "IdempotencyRepeat",
    "GovernanceConfigError",
    "ApprovalPending",
    "pending_tool_result",
    "check_policy",
    "idempotency_key",
    "DefiniteFailure",
    "Unkeyable",
    "metrics",
    "load_prompt",
    "prefetch_prompts",
    "LoadedPrompt",
    "CompileError",
    "PromptFetchError",
    "evaluate",
    "EvaluateResult",
    "EvaluateError",
]

