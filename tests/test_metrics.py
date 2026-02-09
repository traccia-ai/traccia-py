"""Metrics module tests for Traccia SDK."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from traccia.metrics import StandardMetrics, MetricsRecorder
from traccia.metrics.recorder import get_metrics_recorder, set_global_recorder, record_counter, record_histogram


class TestStandardMetrics(unittest.TestCase):
    """Test StandardMetrics factory."""

    def test_create_token_histogram(self):
        """Token histogram has correct name and unit."""
        meter = MagicMock()
        hist = StandardMetrics.create_token_histogram(meter)
        meter.create_histogram.assert_called_once()
        call_kw = meter.create_histogram.call_args[1]
        assert call_kw["name"] == "gen_ai.client.token.usage"
        assert call_kw["unit"] == "{token}"

    def test_create_duration_histogram(self):
        """Duration histogram has correct name and unit."""
        meter = MagicMock()
        hist = StandardMetrics.create_duration_histogram(meter)
        call_kw = meter.create_histogram.call_args[1]
        assert call_kw["name"] == "gen_ai.client.operation.duration"
        assert call_kw["unit"] == "s"

    def test_create_cost_histogram(self):
        """Cost histogram has correct name and unit."""
        meter = MagicMock()
        hist = StandardMetrics.create_cost_histogram(meter)
        call_kw = meter.create_histogram.call_args[1]
        assert call_kw["name"] == "gen_ai.client.operation.cost"
        assert call_kw["unit"] == "usd"

    def test_create_exception_counter(self):
        """Exception counter has correct name."""
        meter = MagicMock()
        StandardMetrics.create_exception_counter(meter)
        call_kw = meter.create_counter.call_args[1]
        assert call_kw["name"] == "gen_ai.client.completions.exceptions"

    def test_create_standard_metrics(self):
        """create_standard_metrics returns all metrics."""
        meter = MagicMock()
        metrics = StandardMetrics.create_standard_metrics(meter)
        assert "token_histogram" in metrics
        assert "duration_histogram" in metrics
        assert "cost_histogram" in metrics
        assert "exception_counter" in metrics
        assert "agent_runs_counter" in metrics
        assert "agent_turns_counter" in metrics
        assert "agent_execution_time_histogram" in metrics


class TestMetricsRecorder(unittest.TestCase):
    """Test MetricsRecorder."""

    def setUp(self):
        self.token_hist = MagicMock()
        self.duration_hist = MagicMock()
        self.cost_hist = MagicMock()
        self.exception_counter = MagicMock()
        self.agent_runs_counter = MagicMock()
        self.agent_turns_counter = MagicMock()
        self.agent_exec_hist = MagicMock()
        self.metrics = {
            "token_histogram": self.token_hist,
            "duration_histogram": self.duration_hist,
            "cost_histogram": self.cost_hist,
            "exception_counter": self.exception_counter,
            "agent_runs_counter": self.agent_runs_counter,
            "agent_turns_counter": self.agent_turns_counter,
            "agent_execution_time_histogram": self.agent_exec_hist,
        }

    def test_record_token_usage(self):
        """Record token usage calls histogram with correct attributes."""
        recorder = MetricsRecorder(self.metrics, sample_rate=1.0)
        recorder.record_token_usage(
            prompt_tokens=10,
            completion_tokens=5,
            attributes={"gen_ai.system": "openai"},
        )
        assert self.token_hist.record.call_count == 2  # input + output
        calls = self.token_hist.record.call_args_list
        assert calls[0][0][0] == 10
        assert calls[0][1]["attributes"]["gen_ai.token.type"] == "input"
        assert calls[1][0][0] == 5
        assert calls[1][1]["attributes"]["gen_ai.token.type"] == "output"

    def test_record_duration(self):
        """Record duration calls histogram."""
        recorder = MetricsRecorder(self.metrics, sample_rate=1.0)
        recorder.record_duration(0.5, attributes={"gen_ai.system": "openai"})
        self.duration_hist.record.assert_called_once_with(0.5, attributes={"gen_ai.system": "openai"})

    def test_record_cost(self):
        """Record cost calls histogram."""
        recorder = MetricsRecorder(self.metrics, sample_rate=1.0)
        recorder.record_cost(0.001, attributes={"gen_ai.request.model": "gpt-4"})
        self.cost_hist.record.assert_called_once_with(0.001, attributes={"gen_ai.request.model": "gpt-4"})

    def test_record_exception(self):
        """Record exception calls counter."""
        recorder = MetricsRecorder(self.metrics, sample_rate=1.0)
        recorder.record_exception(attributes={"gen_ai.system": "openai"})
        self.exception_counter.add.assert_called_once_with(1, attributes={"gen_ai.system": "openai"})

    def test_record_agent_run(self):
        """Record agent run calls counter."""
        recorder = MetricsRecorder(self.metrics, sample_rate=1.0)
        recorder.record_agent_run(attributes={"gen_ai.agent.name": "test_agent"})
        self.agent_runs_counter.add.assert_called_once_with(1, attributes={"gen_ai.agent.name": "test_agent"})

    def test_record_agent_turn(self):
        """Record agent turn calls counter."""
        recorder = MetricsRecorder(self.metrics, sample_rate=1.0)
        recorder.record_agent_turn()
        self.agent_turns_counter.add.assert_called_once_with(1, attributes={})

    def test_record_agent_execution_time(self):
        """Record agent execution time calls histogram."""
        recorder = MetricsRecorder(self.metrics, sample_rate=1.0)
        recorder.record_agent_execution_time(2.5, attributes={"gen_ai.agent.id": "1"})
        self.agent_exec_hist.record.assert_called_once_with(2.5, attributes={"gen_ai.agent.id": "1"})

    def test_sample_rate_skips_recording(self):
        """When sample_rate=0, no metrics are recorded."""
        recorder = MetricsRecorder(self.metrics, sample_rate=0.0)
        recorder.record_token_usage(prompt_tokens=10, completion_tokens=5)
        recorder.record_cost(0.001)
        self.token_hist.record.assert_not_called()
        self.cost_hist.record.assert_not_called()


class TestGlobalRecorder(unittest.TestCase):
    """Test global recorder and custom metrics API."""

    def tearDown(self):
        set_global_recorder(None)

    def test_get_metrics_recorder_none_when_not_set(self):
        """get_metrics_recorder returns None when not set."""
        set_global_recorder(None)
        assert get_metrics_recorder() is None

    def test_set_and_get_global_recorder(self):
        """set_global_recorder and get_metrics_recorder work."""
        recorder = MagicMock()
        set_global_recorder(recorder)
        assert get_metrics_recorder() is recorder

    def test_record_counter_no_recorder(self):
        """record_counter does nothing when recorder is None."""
        set_global_recorder(None)
        record_counter("test_counter", 1)  # Should not raise

    def test_record_histogram_no_recorder(self):
        """record_histogram does nothing when recorder is None."""
        set_global_recorder(None)
        record_histogram("test_hist", 1.0)  # Should not raise


class TestMetricsConfig(unittest.TestCase):
    """Test metrics configuration integration."""

    def test_config_has_metrics_section(self):
        """TracciaConfig includes metrics section."""
        from traccia.config import TracciaConfig
        config = TracciaConfig()
        assert hasattr(config, "metrics")
        assert config.metrics.enable_metrics is True
        assert config.metrics.metrics_sample_rate == 1.0


if __name__ == "__main__":
    unittest.main()
