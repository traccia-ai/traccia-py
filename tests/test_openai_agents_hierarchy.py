"""OpenAI Agents callbacks keep the framework's hierarchy and timing (TRC-294)."""

import os
import tempfile
import time
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from traccia import get_tracer, stop_tracing
from traccia.auto import start_tracing
from traccia.integrations.openai_agents.processor import TracciaAgentsTracingProcessor, _iso_to_ns

try:
    import agents  # noqa: F401

    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False


class _Capture:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return True

    def shutdown(self):
        pass


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _ns(dt: datetime) -> int:
    return _iso_to_ns(dt.isoformat())


def _trace(trace_id, name="Agent workflow", **kw):
    return SimpleNamespace(trace_id=trace_id, name=name, **kw)


def _span(span_id, trace_id, parent_id, span_type, name=None, started_at=None, ended_at=None, error=None, **data):
    return SimpleNamespace(
        span_id=span_id, trace_id=trace_id, parent_id=parent_id, started_at=started_at, ended_at=ended_at,
        error=error, span_data=SimpleNamespace(type=span_type, name=name, **data),
    )


class _TracingTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("TRACCIA_", "AGENT_DASHBOARD_"))}
        patcher = unittest.mock.patch.dict(os.environ, dict(env, HOME=self._dir.name), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.capture = _Capture()
        start_tracing(exporter=self.capture, enable_patching=False, enable_metrics=False, load_env=False)
        self.processor = TracciaAgentsTracingProcessor()

    def tearDown(self):
        stop_tracing()

    def exported(self):
        stop_tracing()  # exports what is queued
        return {s.name: s for s in self.capture.spans}


class TestIsoToNs(unittest.TestCase):
    def test_parses_agents_sdk_formats(self):
        expected = 1_700_000_000_123_456_000
        self.assertEqual(_iso_to_ns("2023-11-14T22:13:20.123456+00:00"), expected)  # Python SDK
        self.assertEqual(_iso_to_ns("2023-11-14T22:13:20.123456Z"), expected)
        self.assertEqual(_iso_to_ns("2023-11-14T22:13:20.123Z"), 1_700_000_000_123_000_000)  # JS SDK
        self.assertEqual(_iso_to_ns("2023-11-14T22:13:20.123456"), expected)  # naive = UTC
        self.assertEqual(_iso_to_ns("2023-11-15T03:43:20.123456+05:30"), expected)

    def test_rejects_missing_or_bad_values(self):
        for value in (None, "", "not a date", 123, object()):
            self.assertIsNone(_iso_to_ns(value), value)


class TestTracerExplicitTimes(_TracingTest):
    def test_start_and_end_times_are_used(self):
        span = get_tracer("t").start_span("work", start_time_ns=1_700_000_000_000_000_000)
        span.end(end_time_ns=1_700_000_000_250_000_000)
        out = self.exported()["work"]
        self.assertEqual(out.start_time_ns, 1_700_000_000_000_000_000)
        self.assertEqual(out.end_time_ns, 1_700_000_000_250_000_000)

    def test_end_is_never_before_start(self):
        span = get_tracer("t").start_span("work", start_time_ns=1_700_000_000_000_000_000)
        span.end(end_time_ns=1_600_000_000_000_000_000)
        out = self.exported()["work"]
        self.assertEqual(out.end_time_ns, out.start_time_ns)

    def test_defaults_to_now(self):
        before = time.time_ns()
        span = get_tracer("t").start_span("work")
        span.end()
        after = time.time_ns()
        out = self.exported()["work"]
        self.assertTrue(before <= out.start_time_ns <= out.end_time_ns <= after)


class TestAgentsHierarchy(_TracingTest):
    def test_ticket_repro_root_and_child_share_trace_and_link(self):
        """TRC-294: no active Traccia span; the callbacks' trace/parent IDs are authoritative."""
        p = self.processor
        p.on_trace_start(_trace("trace_a"))
        p.on_span_start(_span("span_root", "trace_a", None, "agent", "triage"))
        p.on_span_start(_span("span_child", "trace_a", "span_root", "function", "lookup"))
        p.on_span_start(_span("span_grandchild", "trace_a", "span_child", "custom", "inner"))
        p.on_span_end(_span("span_grandchild", "trace_a", "span_child", "custom", "inner"))
        p.on_span_end(_span("span_child", "trace_a", "span_root", "function", "lookup"))
        p.on_span_end(_span("span_root", "trace_a", None, "agent", "triage"))
        p.on_trace_end(_trace("trace_a"))

        spans = self.exported()
        root, agent = spans["Agent workflow"], spans["agent.triage"]
        tool, inner = spans["agent.tool.lookup"], spans["agent.custom.inner"]
        self.assertIsNone(root.parent_span_id)
        self.assertEqual({s.context.trace_id for s in spans.values()}, {root.context.trace_id})
        self.assertEqual(agent.parent_span_id, root.context.span_id)
        self.assertEqual(tool.parent_span_id, agent.context.span_id)
        self.assertEqual(inner.parent_span_id, tool.context.span_id)
        self.assertEqual(tool.attributes["agents.span_id"], "span_child")
        self.assertEqual(tool.attributes["agents.parent_id"], "span_root")
        self.assertEqual(tool.attributes["agents.trace_id"], "trace_a")
        self.assertEqual(root.attributes["agents.trace_id"], "trace_a")

    def test_group_id_recorded_on_root(self):
        self.processor.on_trace_start(_trace("trace_g", group_id="conv-1"))
        self.processor.on_trace_end(_trace("trace_g"))
        self.assertEqual(self.exported()["Agent workflow"].attributes["agents.group_id"], "conv-1")

    def test_callback_timestamps_are_preserved(self):
        t0 = datetime.now(timezone.utc) - timedelta(seconds=10)
        p = self.processor
        p.on_trace_start(_trace("trace_t", started_at=_iso(t0)))
        agent = dict(started_at=_iso(t0 + timedelta(milliseconds=5)), ended_at=_iso(t0 + timedelta(milliseconds=900)))
        tool = dict(started_at=_iso(t0 + timedelta(milliseconds=100)), ended_at=_iso(t0 + timedelta(milliseconds=350)))
        p.on_span_start(_span("a", "trace_t", None, "agent", "triage", **agent))
        p.on_span_start(_span("f", "trace_t", "a", "function", "lookup", **tool))
        time.sleep(0.02)  # callbacks arrive well after the framework's timestamps
        p.on_span_end(_span("f", "trace_t", "a", "function", "lookup", **tool))
        p.on_span_end(_span("a", "trace_t", None, "agent", "triage", **agent))
        p.on_trace_end(_trace("trace_t"))

        spans = self.exported()
        self.assertEqual(spans["Agent workflow"].start_time_ns, _ns(t0))
        self.assertEqual(spans["agent.triage"].start_time_ns, _ns(t0 + timedelta(milliseconds=5)))
        self.assertEqual(spans["agent.triage"].end_time_ns, _ns(t0 + timedelta(milliseconds=900)))
        self.assertEqual(spans["agent.tool.lookup"].start_time_ns, _ns(t0 + timedelta(milliseconds=100)))
        self.assertEqual(spans["agent.tool.lookup"].end_time_ns, _ns(t0 + timedelta(milliseconds=350)))
        self.assertGreaterEqual(spans["Agent workflow"].end_time_ns, spans["agent.triage"].end_time_ns)

    def test_child_never_starts_before_parent_or_ends_before_it_starts(self):
        p = self.processor
        p.on_trace_start(_trace("trace_c"))
        root_start = None
        early = datetime.now(timezone.utc) - timedelta(seconds=5)
        p.on_span_start(_span("a", "trace_c", None, "agent", "triage", started_at=_iso(early)))
        p.on_span_end(_span("a", "trace_c", None, "agent", "triage", started_at=_iso(early),
                            ended_at=_iso(early - timedelta(seconds=1))))
        p.on_trace_end(_trace("trace_c"))
        spans = self.exported()
        root_start = spans["Agent workflow"].start_time_ns
        agent = spans["agent.triage"]
        self.assertEqual(agent.start_time_ns, root_start)  # clamped into the root
        self.assertEqual(agent.end_time_ns, agent.start_time_ns)

    def test_root_covers_a_child_timestamped_past_our_clock(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=2)
        p = self.processor
        p.on_trace_start(_trace("trace_f"))
        p.on_span_start(_span("a", "trace_f", None, "agent", "triage"))
        p.on_span_end(_span("a", "trace_f", None, "agent", "triage", ended_at=_iso(future)))
        p.on_trace_end(_trace("trace_f"))
        spans = self.exported()
        self.assertEqual(spans["agent.triage"].end_time_ns, _ns(future))
        self.assertGreaterEqual(spans["Agent workflow"].end_time_ns, _ns(future))

    def test_missing_timestamps_fall_back_to_callback_time(self):
        before = time.time_ns()
        p = self.processor
        p.on_trace_start(_trace("trace_m"))
        p.on_span_start(_span("a", "trace_m", None, "agent", "triage"))
        p.on_span_end(_span("a", "trace_m", None, "agent", "triage"))
        p.on_trace_end(_trace("trace_m"))
        after = time.time_ns()
        agent = self.exported()["agent.triage"]
        self.assertTrue(before <= agent.start_time_ns <= agent.end_time_ns <= after)

    def test_trace_nests_inside_the_callers_span(self):
        with get_tracer("app").start_as_current_span("request") as outer:
            p = self.processor
            p.on_trace_start(_trace("trace_n"))
            p.on_span_start(_span("a", "trace_n", None, "agent", "triage"))
            p.on_span_end(_span("a", "trace_n", None, "agent", "triage"))
            p.on_trace_end(_trace("trace_n"))
        spans = self.exported()
        self.assertEqual(spans["Agent workflow"].parent_span_id, outer.context.span_id)
        self.assertEqual(spans["agent.triage"].parent_span_id, spans["Agent workflow"].context.span_id)
        self.assertEqual({s.context.trace_id for s in spans.values()}, {outer.context.trace_id})

    def test_parent_from_callbacks_wins_over_an_unrelated_current_span(self):
        p = self.processor
        p.on_trace_start(_trace("trace_u"))
        p.on_span_start(_span("a", "trace_u", None, "agent", "triage"))
        with get_tracer("app").start_as_current_span("unrelated"):
            p.on_span_start(_span("f", "trace_u", "a", "function", "lookup"))
            p.on_span_end(_span("f", "trace_u", "a", "function", "lookup"))
        p.on_span_end(_span("a", "trace_u", None, "agent", "triage"))
        p.on_trace_end(_trace("trace_u"))
        spans = self.exported()
        self.assertEqual(spans["agent.tool.lookup"].parent_span_id, spans["agent.triage"].context.span_id)
        self.assertEqual(spans["agent.tool.lookup"].context.trace_id, spans["agent.triage"].context.trace_id)
        self.assertNotEqual(spans["unrelated"].context.trace_id, spans["agent.triage"].context.trace_id)

    def test_interleaved_traces_stay_separate(self):
        p = self.processor
        p.on_trace_start(_trace("t1", name="run-1"))
        p.on_trace_start(_trace("t2", name="run-2"))
        p.on_span_start(_span("a1", "t1", None, "agent", "one"))
        p.on_span_start(_span("a2", "t2", None, "agent", "two"))
        p.on_span_start(_span("f2", "t2", "a2", "function", "tool2"))
        p.on_span_start(_span("f1", "t1", "a1", "function", "tool1"))
        for sid, tid, pid, typ, name in (("f1", "t1", "a1", "function", "tool1"), ("a2", "t2", None, "agent", "two"),
                                         ("f2", "t2", "a2", "function", "tool2"), ("a1", "t1", None, "agent", "one")):
            p.on_span_end(_span(sid, tid, pid, typ, name))
        p.on_trace_end(_trace("t2", name="run-2"))
        p.on_trace_end(_trace("t1", name="run-1"))

        spans = self.exported()
        self.assertEqual(len(spans), 6)
        for run, agent, tool in (("run-1", "agent.one", "agent.tool.tool1"), ("run-2", "agent.two", "agent.tool.tool2")):
            self.assertIsNone(spans[run].parent_span_id)
            self.assertEqual(spans[agent].parent_span_id, spans[run].context.span_id)
            self.assertEqual(spans[tool].parent_span_id, spans[agent].context.span_id)
            self.assertEqual({spans[n].context.trace_id for n in (run, agent, tool)}, {spans[run].context.trace_id})
        self.assertNotEqual(spans["run-1"].context.trace_id, spans["run-2"].context.trace_id)

    def test_unknown_trace_falls_back_without_raising(self):
        """Processor registered mid-run: no trace start seen. Children still link to known parents."""
        p = self.processor
        p.on_span_start(_span("a", "trace_x", "never_seen", "agent", "triage"))
        p.on_span_start(_span("f", "trace_x", "a", "function", "lookup"))
        p.on_span_end(_span("f", "trace_x", "a", "function", "lookup"))
        p.on_span_end(_span("a", "trace_x", "never_seen", "agent", "triage"))
        p.on_trace_end(_trace("trace_x"))  # no matching start: ignored
        p.on_span_end(_span("nope", "trace_x", None, "agent", "x"))  # no matching start: ignored
        spans = self.exported()
        self.assertEqual(sorted(spans), ["agent.tool.lookup", "agent.triage"])
        self.assertIsNone(spans["agent.triage"].parent_span_id)
        self.assertEqual(spans["agent.tool.lookup"].parent_span_id, spans["agent.triage"].context.span_id)

    def test_error_status_is_kept(self):
        from traccia.tracer.span import SpanStatus

        p = self.processor
        p.on_trace_start(_trace("trace_e"))
        p.on_span_start(_span("a", "trace_e", None, "agent", "triage"))
        p.on_span_end(_span("a", "trace_e", None, "agent", "triage", error={"message": "boom", "data": {}}))
        p.on_trace_end(_trace("trace_e"))
        agent = self.exported()["agent.triage"]
        self.assertEqual(agent.status, SpanStatus.ERROR)
        self.assertEqual(agent.status_description, "boom")

    def test_metrics_use_the_framework_duration(self):
        recorder = unittest.mock.MagicMock()
        t0 = datetime.now(timezone.utc) - timedelta(seconds=3)
        times = dict(started_at=_iso(t0), ended_at=_iso(t0 + timedelta(milliseconds=1500)))
        p = self.processor
        with unittest.mock.patch("traccia.metrics.recorder.get_metrics_recorder", return_value=recorder):
            p.on_trace_start(_trace("trace_metrics"))
            p.on_span_start(_span("a", "trace_metrics", None, "agent", "triage", **times))
            p.on_span_start(_span("g", "trace_metrics", "a", "generation", model="gpt-4o-mini",
                                  usage={"input_tokens": 3, "output_tokens": 4}, **times))
            p.on_span_end(_span("g", "trace_metrics", "a", "generation", model="gpt-4o-mini",
                                usage={"input_tokens": 3, "output_tokens": 4}, **times))
            p.on_span_end(_span("a", "trace_metrics", None, "agent", "triage", **times))
            p.on_trace_end(_trace("trace_metrics"))
        self.assertAlmostEqual(recorder.record_duration.call_args.args[0], 1.5, places=6)
        self.assertAlmostEqual(recorder.record_agent_execution_time.call_args.args[0], 1.5, places=6)
        recorder.record_agent_run.assert_called_once()


@unittest.skipUnless(HAS_AGENTS, "openai-agents not installed")
class TestRealAgentsSdk(_TracingTest):
    """Drive the real Agents SDK tracing runtime, with only our processor registered."""

    def setUp(self):
        super().setUp()
        from agents.tracing import get_trace_provider, set_trace_processors

        multi = getattr(get_trace_provider(), "_multi_processor", None)
        saved = list(getattr(multi, "_processors", ()) or ())
        set_trace_processors([self.processor])
        self.addCleanup(set_trace_processors, saved)

    def test_real_sdk_run_is_one_trace_with_framework_times(self):
        from agents.tracing import agent_span, function_span, generation_span, trace

        with trace("Agent workflow") as t:
            with agent_span(name="triage") as a:
                with function_span(name="lookup", input="q") as f:
                    time.sleep(0.01)
                with generation_span(model="gpt-4o-mini") as g:
                    pass
        spans = self.exported()

        self.assertEqual(sorted(spans), ["Agent workflow", "agent.tool.lookup", "agent.triage", "llm.agents.generation"])
        root = spans["Agent workflow"]
        self.assertIsNone(root.parent_span_id)
        self.assertEqual({s.context.trace_id for s in spans.values()}, {root.context.trace_id})
        self.assertEqual(root.attributes["agents.trace_id"], t.trace_id)
        self.assertEqual(spans["agent.triage"].parent_span_id, root.context.span_id)
        for name in ("agent.tool.lookup", "llm.agents.generation"):
            self.assertEqual(spans[name].parent_span_id, spans["agent.triage"].context.span_id, name)
        for name, sdk_span in (("agent.triage", a), ("agent.tool.lookup", f), ("llm.agents.generation", g)):
            self.assertEqual(spans[name].start_time_ns, max(_iso_to_ns(sdk_span.started_at), root.start_time_ns), name)
            self.assertEqual(spans[name].end_time_ns, _iso_to_ns(sdk_span.ended_at), name)


if __name__ == "__main__":
    unittest.main()
