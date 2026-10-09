"""Point GitHub Copilot's built-in OpenTelemetry exporter at Traccia."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

# VS Code stores Copilot's OTLP config under this settings namespace.
VSCODE_PREFIX = "github.copilot.chat.otel"

# The path Copilot's exporter appends to its base endpoint for trace export.
STANDARD_TRACES_PATH = "/v1/traces"

@dataclass(frozen=True)
class NativeOtelConfig:
    """Resolved inputs for rendering Copilot's native OTLP settings."""

    endpoint: str
    """The normalized OTLP traces endpoint Copilot exports to."""

    base: str
    """``endpoint`` with its trailing ``/v1/traces`` removed."""

    api_key_present: bool
    capture_content: bool
    max_attribute_size_chars: Optional[int] = None
    service_name: Optional[str] = None
    resource_attributes: Optional[Dict[str, str]] = None

def resolve(
    endpoint: str,
    api_key: Optional[str],
    capture_content: bool,
    *,
    max_attribute_size_chars: Optional[int] = None,
    service_name: Optional[str] = None,
    resource_attributes: Optional[Dict[str, str]] = None,
) -> NativeOtelConfig:
    endpoint = _normalize_endpoint(endpoint)
    _validate_endpoint(endpoint)
    return NativeOtelConfig(
        endpoint=endpoint,
        base=endpoint[: -len(STANDARD_TRACES_PATH)],
        api_key_present=bool(api_key),
        capture_content=bool(capture_content),
        max_attribute_size_chars=max_attribute_size_chars,
        service_name=service_name or None,
        resource_attributes=resource_attributes or None,
    )


def _normalize_endpoint(endpoint: str) -> str:
    """Convert Traccia's legacy traces path to Copilot's OTLP path."""
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v2/traces"):
        endpoint = endpoint[: -len("/v2/traces")] + STANDARD_TRACES_PATH
    if not endpoint.endswith(STANDARD_TRACES_PATH):
        raise ValueError(
            "Copilot native OpenTelemetry requires an OTLP HTTP traces endpoint "
            "ending in '/v1/traces'."
        )
    return endpoint


def _validate_endpoint(endpoint: str) -> None:
    """Reject malformed and insecure remote endpoints."""
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Endpoint must be an absolute HTTP or HTTPS URL.")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("Remote endpoints must use HTTPS. HTTP is allowed only for loopback hosts.")


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def vscode_settings(cfg: NativeOtelConfig) -> Dict[str, object]:
    """The ``github.copilot.chat.otel.*`` keys for ``.vscode/settings.json``.

    ``otlpEndpoint`` is Copilot's own target: the base URL of the normalized
    ``/v1/traces`` endpoint.
    Authentication headers are configured through
    ``OTEL_EXPORTER_OTLP_HEADERS`` rather than workspace settings, so secrets
    are not written to a repository file.
    """
    settings: Dict[str, object] = {
        f"{VSCODE_PREFIX}.enabled": True,
        f"{VSCODE_PREFIX}.exporterType": "otlp-http",
        f"{VSCODE_PREFIX}.protocol": "http/protobuf",
        f"{VSCODE_PREFIX}.otlpEndpoint": cfg.base,
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

    Copilot can write spans to a local JSONL file instead of exporting over the
    network.
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
        "OTEL_EXPORTER_OTLP_ENDPOINT": cfg.base,
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    }
    if cfg.api_key_present:
        env["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Bearer ${TRACCIA_API_KEY}"
    if cfg.capture_content:
        env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    if cfg.service_name:
        env["OTEL_SERVICE_NAME"] = cfg.service_name
    if cfg.resource_attributes:
        env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(
            f"{k}={v}" for k, v in cfg.resource_attributes.items()
        )
    return env


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
    if not cfg.api_key_present:
        notes.append(
            "No API key resolved. Set TRACCIA_API_KEY before starting Copilot, "
            "or pass --api-key to verify that authentication is configured."
        )
    else:
        notes.append(
            "Set TRACCIA_API_KEY securely before starting Copilot. The rendered "
            "header deliberately references that environment variable."
        )
    if cfg.capture_content:
        notes.append(
            "Content capture is ON. Copilot does not redact captured prompts, "
            "responses or tool arguments (unlike Traccia's hook path)."
        )
    return notes
