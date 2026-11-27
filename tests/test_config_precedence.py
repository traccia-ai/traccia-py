"""init() and start_tracing() resolve each option as explicit > environment > file > default (TRC-296).

False and 0 are real values at every level, aliases (enable_file / enable_file_exporter)
must not overwrite each other, environment toggles apply with or without a config file,
and unknown options are rejected. auto_start_trace precedence lives in test_auto_start.py.
"""

from __future__ import annotations

import pytest

import traccia
from traccia import auto, get_tracer_provider, set_tracer_provider
from traccia import config as sdk_config
from traccia.errors import ConfigError
from traccia.processors.cost_processor import CostAnnotatingProcessor
from traccia.processors.logging_processor import LoggingSpanProcessor
from traccia.processors.rate_limiter import RateLimitingSpanProcessor
from traccia.processors.redaction_processor import RedactionSpanProcessor
from traccia.processors.token_counter import TokenCountingProcessor
from traccia.tracer.provider import TracerProvider


class CollectingExporter:
    def __init__(self) -> None:
        self.spans = []

    def export(self, spans) -> bool:
        self.spans.extend(spans)
        return True

    def shutdown(self) -> None:
        pass


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No TRACCIA_* variable, .env or config file from this machine; a fresh provider."""
    names = {name for names in sdk_config.ENV_VAR_MAPPING.values() for name in names}
    names |= {"TRACCIA_REDACT_PII", "TRACCIA_CONFIG_FILE"}
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert sdk_config.find_config_file() is None

    traccia.stop_tracing()
    previous = get_tracer_provider()
    provider = TracerProvider()
    set_tracer_provider(provider)
    yield tmp_path
    traccia.stop_tracing()
    set_tracer_provider(previous)


def _toml(tmp_path, body: str) -> str:
    path = tmp_path / "traccia.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _init(**kwargs):
    """init() with the side effects these tests do not look at turned off."""
    kwargs.setdefault("exporter", CollectingExporter())
    kwargs.setdefault("enable_metrics", False)
    kwargs.setdefault("enable_patching", False)
    kwargs.setdefault("load_env", False)
    kwargs.setdefault("auto_start_trace", False)
    return traccia.init(**kwargs)


def _has(processor_type) -> bool:
    return any(isinstance(p, processor_type) for p in get_tracer_provider()._enrichment_processors)


def _sample_rate() -> float:
    return get_tracer_provider().get_sampler().sample_rate


# --- sample_rate -------------------------------------------------------------


def test_explicit_sample_rate_wins_over_environment(clean_env, monkeypatch):
    """Ticket repro: explicit 0.1 became the environment's 0.9."""
    monkeypatch.setenv("TRACCIA_SAMPLE_RATE", "0.9")
    _init(sample_rate=0.1)
    assert _sample_rate() == 0.1


def test_explicit_sample_rate_zero_wins_over_environment(clean_env, monkeypatch):
    monkeypatch.setenv("TRACCIA_SAMPLE_RATE", "0.9")
    _init(sample_rate=0.0)
    assert _sample_rate() == 0.0


def test_start_tracing_explicit_sample_rate_wins_over_environment(clean_env, monkeypatch):
    monkeypatch.setenv("TRACCIA_SAMPLE_RATE", "0.9")
    auto.start_tracing(
        exporter=CollectingExporter(), sample_rate=0.1, enable_metrics=False, enable_patching=False, load_env=False
    )
    assert _sample_rate() == 0.1


def test_environment_sample_rate_applies_when_not_passed(clean_env, monkeypatch):
    monkeypatch.setenv("TRACCIA_SAMPLE_RATE", "0.9")
    _init()
    assert _sample_rate() == 0.9


def test_environment_sample_rate_wins_over_file(clean_env, monkeypatch):
    monkeypatch.setenv("TRACCIA_SAMPLE_RATE", "0.9")
    _init(config_file=_toml(clean_env, "[tracing]\nsample_rate = 0.2\n"))
    assert _sample_rate() == 0.9


def test_file_sample_rate_zero_is_kept(clean_env):
    _init(config_file=_toml(clean_env, "[tracing]\nsample_rate = 0.0\n"))
    assert _sample_rate() == 0.0


# --- exporter aliases ----------------------------------------------------------


def test_explicit_enable_file_exporter_survives_config_file_default(clean_env):
    """Ticket repro: the file's enable_file default (false) overwrote the explicit alias."""
    trace_file = clean_env / "traces.jsonl"
    config = _toml(clean_env, "[tracing]\nsample_rate = 1.0\n")
    _init(config_file=config, enable_file_exporter=True, file_exporter_path=str(trace_file))
    traccia.get_tracer("t").start_span("span").end()
    auto.force_flush()
    assert trace_file.exists() and trace_file.read_text(encoding="utf-8").strip()


def test_explicit_enable_file_exporter_false_wins_over_file_true(clean_env):
    trace_file = clean_env / "traces.jsonl"
    config = _toml(clean_env, f"[exporters]\nenable_file = true\nfile_exporter_path = '{trace_file.as_posix()}'\n")
    _init(config_file=config, enable_file_exporter=False)
    traccia.get_tracer("t").start_span("span").end()
    auto.force_flush()
    assert not trace_file.exists() or not trace_file.read_text(encoding="utf-8").strip()


def test_file_enable_file_applies_when_not_passed(clean_env):
    trace_file = clean_env / "traces.jsonl"
    config = _toml(clean_env, f"[exporters]\nenable_file = true\nfile_exporter_path = '{trace_file.as_posix()}'\n")
    _init(config_file=config)
    traccia.get_tracer("t").start_span("span").end()
    auto.force_flush()
    assert trace_file.read_text(encoding="utf-8").strip()


# --- feature toggles from the environment ---------------------------------------

TOGGLES = [
    # (option, env var, env value, processor, present when env applies)
    ("enable_costs", "TRACCIA_ENABLE_COSTS", "false", CostAnnotatingProcessor, False),
    ("enable_token_counting", "TRACCIA_ENABLE_TOKEN_COUNTING", "false", TokenCountingProcessor, False),
    ("enable_span_logging", "TRACCIA_ENABLE_SPAN_LOGGING", "true", LoggingSpanProcessor, True),
    ("redact_pii", "TRACCIA_REDACT_PII", "true", RedactionSpanProcessor, True),
]
TOGGLE_IDS = [t[0] for t in TOGGLES]


@pytest.mark.parametrize("option,env,value,processor,present", TOGGLES, ids=TOGGLE_IDS)
def test_environment_toggle_applies_without_config_file(clean_env, monkeypatch, option, env, value, processor, present):
    """Ticket repro: cost-disable and redaction toggles were ignored in the no-file path."""
    monkeypatch.setenv(env, value)
    _init()
    assert _has(processor) is present


@pytest.mark.parametrize("option,env,value,processor,present", TOGGLES, ids=TOGGLE_IDS)
def test_environment_toggle_applies_with_config_file(clean_env, monkeypatch, option, env, value, processor, present):
    monkeypatch.setenv(env, value)
    _init(config_file=_toml(clean_env, "[tracing]\nsample_rate = 1.0\n"))
    assert _has(processor) is present


@pytest.mark.parametrize("option,env,value,processor,present", TOGGLES, ids=TOGGLE_IDS)
def test_explicit_toggle_wins_over_environment(clean_env, monkeypatch, option, env, value, processor, present):
    monkeypatch.setenv(env, value)
    _init(**{option: not present})
    assert _has(processor) is (not present)


def test_explicit_false_wins_over_file_true(clean_env):
    _init(config_file=_toml(clean_env, "[instrumentation]\nenable_costs = true\n"), enable_costs=False)
    assert not _has(CostAnnotatingProcessor)


def test_environment_wins_over_file(clean_env, monkeypatch):
    monkeypatch.setenv("TRACCIA_ENABLE_COSTS", "true")
    _init(config_file=_toml(clean_env, "[instrumentation]\nenable_costs = false\n"))
    assert _has(CostAnnotatingProcessor)


# --- rate limiter ---------------------------------------------------------------


def test_environment_rate_limit_applies_without_config_file(clean_env, monkeypatch):
    monkeypatch.setenv("TRACCIA_MAX_SPANS_PER_SECOND", "50")
    _init()
    assert isinstance(auto._active_processor, RateLimitingSpanProcessor)


def test_explicit_rate_limit_applies(clean_env):
    _init(max_spans_per_second=50)
    assert isinstance(auto._active_processor, RateLimitingSpanProcessor)


# --- unsupported options --------------------------------------------------------


def test_unknown_init_option_is_rejected(clean_env):
    with pytest.raises(TypeError):
        _init(sampel_rate=0.5)


def test_unknown_config_file_option_is_rejected(clean_env):
    with pytest.raises(ConfigError):
        _init(config_file=_toml(clean_env, "[tracing]\nsampel_rate = 0.5\n"))
