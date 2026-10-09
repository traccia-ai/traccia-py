"""
Tests for instrumentation/gemini.py's patch_gemini() and span population.

sys.modules level for both SDK layouts patch_gemini() knows about:

  * primary candidate (google-genai >= 2.x): google.genai._gaos.google_genai
    .GeminiNextGenInteractions / .AsyncGeminiNextGenInteractions
  * legacy fallback: google.genai.resources.interactions
    .Interactions / .AsyncInteractions
"""

import asyncio
import contextlib
import sys
import types

import pytest

import instrumentation.gemini as gemini_mod
from instrumentation.gemini import patch_gemini


# Fakes: a tracer/span pair that just records attributes
class FakeSpan:
    def __init__(self):
        self.attributes = {}
        self.exception = None
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exception = exc

    def set_status(self, status, message=None):
        self.status = (status, message)


class FakeTracer:
    def __init__(self):
        self.spans = []

    @contextlib.contextmanager
    def start_as_current_span(self, name, attributes=None):
        span = FakeSpan()
        span.attributes.update(attributes or {})
        self.spans.append(span)
        yield span


@pytest.fixture(autouse=True)
def reset_patch_state(monkeypatch):
    """patch_gemini() short-circuits via a module-level _patched flag; each
    test patches a fresh fake class, so the flag must not leak between tests."""
    gemini_mod._patched = False
    yield
    gemini_mod._patched = False


@pytest.fixture
def fake_tracer(monkeypatch):
    tracer = FakeTracer()
    monkeypatch.setattr(gemini_mod, "_get_tracer", lambda name: tracer)
    return tracer


def _install_primary_sdk(monkeypatch, sync_cls, async_cls):
    """Register the real primary-candidate module path patch_gemini() checks first."""
    fake_genai_pkg = types.ModuleType("google.genai")
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_pkg)

    fake_gaos_pkg = types.ModuleType("google.genai._gaos")
    monkeypatch.setitem(sys.modules, "google.genai._gaos", fake_gaos_pkg)

    fake_google_genai_mod = types.ModuleType("google.genai._gaos.google_genai")
    fake_google_genai_mod.GeminiNextGenInteractions = sync_cls
    fake_google_genai_mod.AsyncGeminiNextGenInteractions = async_cls
    monkeypatch.setitem(
        sys.modules, "google.genai._gaos.google_genai", fake_google_genai_mod
    )
    return fake_genai_pkg


def _install_legacy_sdk_only(monkeypatch, sync_cls, async_cls):
    """Register only the legacy fallback path — primary candidate stays absent."""
    fake_genai_pkg = types.ModuleType("google.genai")
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_pkg)

    fake_resources_pkg = types.ModuleType("google.genai.resources")
    monkeypatch.setitem(sys.modules, "google.genai.resources", fake_resources_pkg)

    fake_interactions_mod = types.ModuleType("google.genai.resources.interactions")
    fake_interactions_mod.Interactions = sync_cls
    fake_interactions_mod.AsyncInteractions = async_cls
    monkeypatch.setitem(
        sys.modules, "google.genai.resources.interactions", fake_interactions_mod
    )
    return fake_genai_pkg


def _make_fake_response(**overrides):
    usage = types.SimpleNamespace(
        total_input_tokens=14,
        total_output_tokens=7,
        total_thought_tokens=3,
        total_cached_tokens=0,
        total_tool_use_tokens=0,
        total_tokens=24,  # deliberately != input+output
    )
    defaults = dict(
        id="test-001",
        model="models/gemini-flash-latest",
        status="completed",
        output_text="The capital of France is Paris.",
        usage=usage,
        agent=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_fake_interaction_classes():
    """Build a fresh pair of fake SDK classes for a single test.

    patch_gemini() mutates `create` in place and marks it
    `_agent_trace_patched` to guard against double-patching. If tests shared
    one module-level class, patching it in test A would make test B's
    (unrelated) patch_gemini() call see an already-patched `create` and
    silently no-op — exactly the kind of cross-test state leak that makes a
    suite pass in isolation but fail in combination.
    """

    class FakeInteractions:
        """Sync fake matching the real SDK's `create(self, **kwargs)` shape."""

        def create(self, **kwargs):
            if kwargs.get("stream"):
                return types.SimpleNamespace(_is_stream=True)
            return _make_fake_response()

    class AsyncFakeInteractions:
        async def create(self, **kwargs):
            if kwargs.get("stream"):
                return types.SimpleNamespace(_is_stream=True)
            return _make_fake_response()

    return FakeInteractions, AsyncFakeInteractions


def test_extract_usage_returns_all_none_when_usage_missing():
    resp = types.SimpleNamespace(usage=None)
    result = gemini_mod._extract_usage(resp)
    assert result == (None, None, None, None, None, None)


def test_extract_usage_prefers_provider_total_over_synthesis():
    resp = _make_fake_response()
    input_tok, output_tok, thought_tok, cached_tok, tool_use_tok, total_tok = (
        gemini_mod._extract_usage(resp)
    )
    assert input_tok == 14
    assert output_tok == 7
    assert thought_tok == 3
    assert total_tok == 24  # provider's total, not 14 + 7


def test_extract_usage_treats_zero_as_present_not_missing():
    usage = types.SimpleNamespace(total_input_tokens=0, total_output_tokens=5)
    resp = types.SimpleNamespace(usage=usage)
    input_tok, output_tok, *_ = gemini_mod._extract_usage(resp)
    assert input_tok == 0
    assert output_tok == 5


# patch_gemini() — candidate resolution

def test_patch_gemini_patches_primary_candidate(monkeypatch):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_primary_sdk(monkeypatch, FakeInteractions, AsyncFakeInteractions)

    result = patch_gemini()

    assert result is True
    assert getattr(FakeInteractions.create, "_agent_trace_patched", False) is True
    assert getattr(AsyncFakeInteractions.create, "_agent_trace_patched", False) is True


def test_patch_gemini_falls_back_when_primary_absent(monkeypatch):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_legacy_sdk_only(monkeypatch, FakeInteractions, AsyncFakeInteractions)

    result = patch_gemini()

    assert result is True
    assert getattr(FakeInteractions.create, "_agent_trace_patched", False) is True


def test_patch_gemini_returns_false_when_google_genai_missing(monkeypatch):
    # sys.modules[name] = None makes `import google.genai` raise ImportError.
    monkeypatch.setitem(sys.modules, "google.genai", None)

    result = patch_gemini()

    assert result is False


# Span population via the real wrapper, through the primary candidate

def test_sync_call_populates_span_with_provider_usage(monkeypatch, fake_tracer):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_primary_sdk(monkeypatch, FakeInteractions, AsyncFakeInteractions)
    assert patch_gemini() is True

    client = FakeInteractions()
    resp = client.create(model="models/gemini-flash-latest", input="What is the capital of France?")

    assert resp.output_text == "The capital of France is Paris."
    span = fake_tracer.spans[-1]
    assert span.attributes["llm.model"] == "models/gemini-flash-latest"
    assert span.attributes["llm.prompt"] == "What is the capital of France?"
    assert span.attributes["llm.usage.total_tokens"] == 24
    assert span.attributes["llm.usage.source"] == "provider_usage"
    assert span.attributes["llm.completion"] == "The capital of France is Paris."
    assert span.attributes["llm.interaction_id"] == "test-001"
    assert "llm.streaming" not in span.attributes


def test_model_falls_back_to_response_model_when_kwarg_absent(monkeypatch, fake_tracer):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_primary_sdk(monkeypatch, FakeInteractions, AsyncFakeInteractions)
    assert patch_gemini() is True

    client = FakeInteractions()
    client.create(input="no model kwarg passed")

    span = fake_tracer.spans[-1]
    assert span.attributes["llm.model"] == "models/gemini-flash-latest"


def test_previous_interaction_id_captured(monkeypatch, fake_tracer):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_primary_sdk(monkeypatch, FakeInteractions, AsyncFakeInteractions)
    assert patch_gemini() is True

    client = FakeInteractions()
    client.create(
        model="models/gemini-flash-latest",
        input="follow-up question",
        previous_interaction_id="test-001",
    )

    span = fake_tracer.spans[-1]
    assert span.attributes["llm.previous_interaction_id"] == "test-001"


# Streaming — must no-op instead of populating span from a Stream object

def test_streaming_call_skips_usage_population(monkeypatch, fake_tracer):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_primary_sdk(monkeypatch, FakeInteractions, AsyncFakeInteractions)
    assert patch_gemini() is True

    client = FakeInteractions()
    resp = client.create(model="models/gemini-flash-latest", input="stream this", stream=True)

    assert getattr(resp, "_is_stream", False) is True
    span = fake_tracer.spans[-1]
    assert span.attributes["llm.streaming"] is True
    # _populate_span must NOT have run against the stream object
    assert "llm.usage.total_tokens" not in span.attributes
    assert "llm.completion" not in span.attributes


def test_async_streaming_call_skips_usage_population(monkeypatch, fake_tracer):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_primary_sdk(monkeypatch, FakeInteractions, AsyncFakeInteractions)
    assert patch_gemini() is True

    client = AsyncFakeInteractions()

    async def run():
        return await client.create(model="models/gemini-flash-latest", input="stream this", stream=True)

    resp = asyncio.run(run())

    assert getattr(resp, "_is_stream", False) is True
    span = fake_tracer.spans[-1]
    assert span.attributes["llm.streaming"] is True
    assert "llm.usage.total_tokens" not in span.attributes


def test_async_call_populates_span(monkeypatch, fake_tracer):
    FakeInteractions, AsyncFakeInteractions = _make_fake_interaction_classes()
    _install_primary_sdk(monkeypatch, FakeInteractions, AsyncFakeInteractions)
    assert patch_gemini() is True

    client = AsyncFakeInteractions()

    async def run():
        return await client.create(model="models/gemini-flash-latest", input="hello")

    resp = asyncio.run(run())

    assert resp.output_text == "The capital of France is Paris."
    span = fake_tracer.spans[-1]
    assert span.attributes["llm.usage.total_tokens"] == 24


def test_generate_content_sync_call_populates_span(monkeypatch, fake_tracer):
    pytest.importorskip("google.genai")
    from google.genai.models import Models

    fake_response = types.SimpleNamespace(
        text="Generated content response",
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            thoughts_token_count=5,
            total_token_count=35,
        ),
        model_version="gemini-2.5-flash",
    )

    def fake_generate_content(self, model, contents, **kwargs):
        return fake_response

    monkeypatch.setattr(Models, "generate_content", fake_generate_content)

    assert patch_gemini() is True

    models = object.__new__(Models)
    resp = models.generate_content(model="gemini-2.5-flash", contents="Tell me a joke")

    assert resp.text == "Generated content response"
    span = fake_tracer.spans[-1]
    assert span.attributes["llm.vendor"] == "google_gemini"
    assert span.attributes["llm.model"] == "gemini-2.5-flash"
    assert span.attributes["llm.prompt"] == "Tell me a joke"
    assert span.attributes["llm.completion"] == "Generated content response"
    assert span.attributes["llm.usage.prompt_tokens"] == 10
    assert span.attributes["llm.usage.completion_tokens"] == 20
    assert span.attributes["llm.usage.thought_tokens"] == 5
    assert span.attributes["llm.usage.total_tokens"] == 35
    assert span.attributes["llm.usage.prompt_source"] == "provider_usage"
    assert span.attributes["llm.usage.completion_source"] == "provider_usage"
    assert span.attributes["llm.usage.source"] == "provider_usage"
