"""Infrastructure and vendor instrumentation.

This module exposes helpers for HTTP client/server tracing (including FastAPI),
vendor SDK patching (OpenAI, Anthropic, requests), and decorators used for
auto-instrumentation.
"""

from traccia.instrumentation.decorator import observe
from traccia.instrumentation.openai import patch_openai, patch_openai_responses
from traccia.instrumentation.anthropic import patch_anthropic
from traccia.instrumentation.gemini import patch_gemini
from traccia.instrumentation.requests import patch_requests
from traccia.instrumentation.groq import patch_groq
from traccia.instrumentation.http_client import inject_headers as inject_http_headers
from traccia.instrumentation.http_server import extract_parent_context, start_server_span
from traccia.instrumentation.fastapi import install_http_middleware

__all__ = [
    "observe",
    "patch_openai",
    "patch_openai_responses",
    "patch_anthropic",
    "patch_gemini",
    "patch_requests",
    "patch_groq",
    "inject_http_headers",
    "extract_parent_context",
    "start_server_span",
    "install_http_middleware",
]
