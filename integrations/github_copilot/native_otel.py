"""Point GitHub Copilot's built-in OpenTelemetry exporter at Traccia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

# VS Code stores Copilot's OTLP config under this settings namespace.
VSCODE_PREFIX = "github.copilot.chat.otel"

# The path Copilot's exporter appends to its base endpoint for trace export.
STANDARD_TRACES_PATH = "/v1/traces"

# Where the rendered config tells Copilot to send spans when a Collector bridge is needed
LOCAL_COLLECTOR_ENDPOINT = "http://localhost:4318"


@dataclass(frozen=True)
class NativeOtelConfig:
    """Resolved inputs for rendering Copilot's native OTLP settings."""

    endpoint: str
    """The Traccia traces endpoint spans must ultimately reach."""

    base: str
    """``endpoint`` with a trailing ``/v1/traces`` removed, else ``endpoint``."""

    path_compatible: bool
    """True when ``base`` + ``/v1/traces`` reproduces ``endpoint`` -- i.e.
    Copilot can be pointed straight at it with no Collector."""

    api_key: Optional[str]
    capture_content: bool
    max_attribute_size_chars: Optional[int] = None
    service_name: Optional[str] = None
    resource_attributes: Optional[Dict[str, str]] = None

    @property
    def needs_collector(self) -> bool:
        return not self.path_compatible

    @property
    def copilot_endpoint(self) -> str:
        """The base endpoint Copilot itself should target."""
        return self.base if self.path_compatible else LOCAL_COLLECTOR_ENDPOINT


def resolve(
    endpoint: str,
    api_key: Optional[str],
    capture_content: bool,
    *,
    max_attribute_size_chars: Optional[int] = None,
    service_name: Optional[str] = None,
    resource_attributes: Optional[Dict[str, str]] = None,
) -> NativeOtelConfig:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith(STANDARD_TRACES_PATH):
        base = endpoint[: -len(STANDARD_TRACES_PATH)]
        compatible = True
    else:
        base = endpoint
        compatible = False
    return NativeOtelConfig(
        endpoint=endpoint,
        base=base,
        path_compatible=compatible,
        api_key=api_key or None,
        capture_content=bool(capture_content),
        max_attribute_size_chars=max_attribute_size_chars,
        service_name=service_name or None,
        resource_attributes=resource_attributes or None,
    )


def vscode_settings(cfg: NativeOtelConfig) -> Dict[str, object]:
    """The ``github.copilot.chat.otel.*`` keys for ``.vscode/settings.json``.

    ``otlpEndpoint`` is Copilot's own target: the Traccia endpoint when it is
    ``/v1/traces``-shaped, otherwise the local Collector that forwards there.
    Authentication headers are configured through
    ``OTEL_EXPORTER_OTLP_HEADERS`` rather than workspace settings, so secrets
    are not written to a repository file.
    """
    settings: Dict[str, object] = {
        f"{VSCODE_PREFIX}.enabled": True,
        f"{VSCODE_PREFIX}.exporterType": "otlp-http",
        f"{VSCODE_PREFIX}.protocol": "http/protobuf",
        f"{VSCODE_PREFIX}.otlpEndpoint": cfg.copilot_endpoint,
        f"{VSCODE_PREFIX}.captureContent": cfg.capture_content,
    }
    if cfg.max_attribute_size_chars is not None:
        settings[f"{VSCODE_PREFIX}.maxAttributeSizeChars"] = cfg.max_attribute_size_chars
    return settings


def file_exporter_vscode_settings(
    outfile: str,
    capture_content: bool,
    max_attribute_size_chars: Optional[int] = None,
) -> Dict[str, object]:
    """``github.copilot.chat.otel.*`` keys for the local file-exporter fallback.

    For environments where a Collector endpoint isn't reachable, Copilot can
    write spans to a local JSONL file instead of exporting over the network.
    Nothing forwards this file to Traccia automatically -- it's a manual or
    scripted pickup, e.g. the "Chat: Export Agent Traces DB" command mirrors
    it in the CLI via `dbSpanExporter`.
    """
    settings: Dict[str, object] = {
        f"{VSCODE_PREFIX}.enabled": True,
        f"{VSCODE_PREFIX}.exporterType": "file",
        f"{VSCODE_PREFIX}.outfile": outfile,
        f"{VSCODE_PREFIX}.captureContent": bool(capture_content),
    }
    if max_attribute_size_chars is not None:
        settings[f"{VSCODE_PREFIX}.maxAttributeSizeChars"] = max_attribute_size_chars
    return settings


def env_vars(cfg: NativeOtelConfig) -> Dict[str, str]:
    """Environment variables for the Copilot CLI agent host.

    Uses the base ``OTEL_EXPORTER_OTLP_ENDPOINT`` (Copilot appends
    ``/v1/traces``); there is no supported per-signal override. Content capture
    uses the OpenTelemetry GenAI standard variable.
    """
    env: Dict[str, str] = {
        "COPILOT_OTEL_ENABLED": "true",
        "OTEL_EXPORTER_OTLP_ENDPOINT": cfg.copilot_endpoint,
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    }
    if cfg.path_compatible and cfg.api_key:
        env["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Bearer {cfg.api_key}"
    if cfg.capture_content:
        env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    if cfg.service_name:
        env["OTEL_SERVICE_NAME"] = cfg.service_name
    if cfg.resource_attributes:
        env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(
            f"{k}={v}" for k, v in cfg.resource_attributes.items()
        )
    return env


def collector_config(cfg: NativeOtelConfig) -> str:
    """A minimal OpenTelemetry Collector config bridging Copilot to Traccia.

    Empty string when no bridge is needed.
    """
    if cfg.path_compatible:
        return ""
    auth = (
        '      Authorization: "Bearer ${env:TRACCIA_API_KEY}"'
        if not cfg.api_key
        else f'      Authorization: "Bearer {cfg.api_key}"'
    )
    return "\n".join(
        [
            "receivers:",
            "  otlp:",
            "    protocols:",
            "      http:",
            "        endpoint: 0.0.0.0:4318",
            "exporters:",
            "  otlphttp/traccia:",
            f"    traces_endpoint: {cfg.endpoint}",
            "    headers:",
            auth,
            "service:",
            "  pipelines:",
            "    traces:",
            "      receivers: [otlp]",
            "      exporters: [otlphttp/traccia]",
        ]
    )


def merge_settings(
    existing: Dict[str, object], wanted: Dict[str, object], *, overwrite: bool
) -> Tuple[Dict[str, object], List[str]]:
    """Fold `wanted` key/value pairs into `existing`.

    Without ``overwrite`` a key already present with a different value is
    left alone. Returns the new dict and the list of keys that changed. Used
    for both `.vscode/settings.json` and a managed-settings.json target --
    they share the same key/value shape.
    """
    out = dict(existing)
    changed: List[str] = []
    for key, value in wanted.items():
        if key in out and out[key] == value:
            continue
        if key in out and not overwrite:
            continue
        out[key] = value
        changed.append(key)
    return out, changed


def merge_into_vscode_settings(
    existing: Dict[str, object], cfg: NativeOtelConfig, *, overwrite: bool
) -> Tuple[Dict[str, object], List[str]]:
    """Fold the OTLP keys into an existing settings dict.

    Only keys under :data:`VSCODE_PREFIX` are touched. Without ``overwrite`` a
    key already present with a different value is left alone. Returns the new
    dict and the list of keys that changed.
    """
    return merge_settings(existing, vscode_settings(cfg), overwrite=overwrite)


def warnings(cfg: NativeOtelConfig) -> List[str]:
    """Caveats a user should see before applying the rendered config."""
    notes: List[str] = []
    if cfg.needs_collector:
        path = urlsplit(cfg.endpoint).path or "/"
        notes.append(
            f"Copilot appends '/v1/traces' to its base endpoint on both VS Code "
            f"and the CLI, but Traccia ingests at '{path}'. The rendered config "
            f"points Copilot at a local Collector ({LOCAL_COLLECTOR_ENDPOINT}); "
            "run one with the printed config, or add a '/v1/traces' route to "
            "your ingest."
        )
    if not cfg.api_key:
        notes.append(
            "No API key resolved. Set tracing.api_key (or TRACCIA_API_KEY), or "
            "pass --api-key. The Collector config falls back to "
            "${env:TRACCIA_API_KEY} at run time."
        )
    if cfg.capture_content:
        notes.append(
            "Content capture is ON. Copilot does not redact captured prompts, "
            "responses or tool arguments (unlike Traccia's hook path)."
        )
    return notes
