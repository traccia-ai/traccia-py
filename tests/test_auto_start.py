"""Tests for auto-start trace functionality."""

import unittest
import asyncio
import os
import subprocess
import sys
import tempfile
import unittest.mock
from traccia import init, stop_tracing, get_tracer, end_auto_trace, trace
from traccia import auto


class TestAutoStartTrace(unittest.TestCase):
    """Test auto-start trace functionality."""
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            stop_tracing()
        except Exception:
            pass
    
    def test_auto_start_creates_root_trace(self):
        """Test that init() with auto_start_trace=True creates a root trace."""
        init(enable_patching=False, auto_start_trace=True)
        
        # Should have auto-trace context
        self.assertIsNotNone(auto._auto_trace_context)
        self.assertIsNotNone(auto._auto_trace_context.get("span"))
    
    def test_auto_start_disabled(self):
        """Test that auto_start_trace=False doesn't create a root trace."""
        init(enable_patching=False, auto_start_trace=False)
        
        # Should NOT have auto-trace context
        self.assertIsNone(auto._auto_trace_context)
    
    def test_auto_start_span_in_otel_context(self):
        """Auto-started trace must be the active OTel parent for child spans."""
        init(enable_patching=False, auto_start_trace=True)

        auto_span = auto._auto_trace_context["span"]
        from opentelemetry.trace import get_current_span

        current = get_current_span()
        self.assertTrue(current.get_span_context().is_valid)
        self.assertEqual(
            current.get_span_context().trace_id,
            auto_span._otel_span.get_span_context().trace_id,
        )
        self.assertEqual(
            current.get_span_context().span_id,
            auto_span._otel_span.get_span_context().span_id,
        )

    def test_auto_start_parents_manual_child(self):
        """Child spans created after auto-start share the auto trace id."""
        init(enable_patching=False, auto_start_trace=True)

        auto_span = auto._auto_trace_context["span"]
        tracer = get_tracer("test")

        with tracer.start_as_current_span("child-span") as child:
            self.assertEqual(
                child.context.trace_id,
                auto_span.context.trace_id,
            )
    
    def test_auto_trace_custom_name(self):
        """Test that auto-trace can have a custom name."""
        custom_name = "my-custom-root"
        init(enable_patching=False, auto_start_trace=True, auto_trace_name=custom_name)
        
        # Should have auto-trace with custom name
        self.assertIsNotNone(auto._auto_trace_context)
        self.assertEqual(auto._auto_trace_name, custom_name)
    
    def test_end_auto_trace_explicit(self):
        """Test that end_auto_trace() ends the auto-trace."""
        init(enable_patching=False, auto_start_trace=True)
        
        # Should have auto-trace
        self.assertIsNotNone(auto._auto_trace_context)
        
        # End it explicitly
        end_auto_trace()
        
        # Should no longer have auto-trace
        self.assertIsNone(auto._auto_trace_context)
    
    def test_stop_tracing_ends_auto_trace(self):
        """Test that stop_tracing() ends the auto-trace."""
        init(enable_patching=False, auto_start_trace=True)
        
        # Should have auto-trace
        self.assertIsNotNone(auto._auto_trace_context)
        
        # Stop tracing
        stop_tracing()
        
        # Should no longer have auto-trace
        self.assertIsNone(auto._auto_trace_context)
    
    def test_trace_context_manager_ends_auto_trace(self):
        """Test that trace() context manager ends auto-trace."""
        init(enable_patching=False, auto_start_trace=True)
        
        # Should have auto-trace
        self.assertIsNotNone(auto._auto_trace_context)
        
        # Use trace() context manager
        with trace("explicit-trace"):
            # Auto-trace should be ended
            self.assertIsNone(auto._auto_trace_context)
    
    def test_root_span_warning(self):
        """Test that creating a span with 'root' in name logs a warning."""
        init(enable_patching=False, auto_start_trace=True)
        
        tracer = get_tracer("test")
        
        # Create a span with "root" in the name
        # Note: The warning is logged when span is created, not when context exits
        # This is a best-effort test to verify no errors occur
        with tracer.start_as_current_span("my-root-span"):
            pass
    
    def test_decorator_with_auto_trace(self):
        """Test that @observe() decorator works with auto-trace."""
        from traccia.instrumentation import observe
        
        init(enable_patching=False, auto_start_trace=True)
        
        @observe()
        def test_function():
            return "test"
        
        # Should work without errors
        result = test_function()
        self.assertEqual(result, "test")
    
    def test_async_decorator_with_auto_trace(self):
        """Test that @observe() decorator works with auto-trace for async functions."""
        from traccia.instrumentation import observe
        
        init(enable_patching=False, auto_start_trace=True)
        
        @observe()
        async def test_async_function():
            return "async-test"
        
        # Should work without errors
        result = asyncio.run(test_async_function())
        self.assertEqual(result, "async-test")


def _clean_env(home: str) -> dict:
    """The process environment without TRACCIA_*/AGENT_DASHBOARD_* settings, with HOME moved."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TRACCIA_", "AGENT_DASHBOARD_"))}
    env["HOME"] = home
    return env


class TestAutoStartConfigPriority(unittest.TestCase):
    """init() argument > environment > traccia.toml > default (TRC-358, TRC-359)."""

    def setUp(self):
        # Isolated from this machine: no traccia.toml or .env in cwd or HOME, no TRACCIA_* variables.
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        cwd = os.getcwd()
        os.chdir(self._dir.name)
        self.addCleanup(os.chdir, cwd)
        patcher = unittest.mock.patch.dict(os.environ, _clean_env(self._dir.name), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        try:
            stop_tracing()
        except Exception:
            pass

    def _config(self, body: str) -> str:
        path = os.path.join(self._dir.name, "traccia.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[tracing]\n" + body)
        return path

    def test_default_is_off_without_config(self):
        init(enable_patching=False)
        self.assertIsNone(auto._auto_trace_context)

    def test_explicit_false_wins_over_config_file(self):
        path = self._config("auto_start_trace = true\n")
        init(config_file=path, enable_patching=False, auto_start_trace=False)
        self.assertIsNone(auto._auto_trace_context)

    def test_config_file_decides_when_not_passed(self):
        path = self._config("auto_start_trace = false\n")
        init(config_file=path, enable_patching=False)
        self.assertIsNone(auto._auto_trace_context)

    def test_config_file_can_turn_it_on(self):
        path = self._config("auto_start_trace = true\n")
        init(config_file=path, enable_patching=False)
        self.assertIsNotNone(auto._auto_trace_context)

    def test_environment_wins_over_config_file(self):
        path = self._config("auto_start_trace = true\n")
        os.environ["TRACCIA_AUTO_START_TRACE"] = "false"
        init(config_file=path, enable_patching=False)
        self.assertIsNone(auto._auto_trace_context)

    def test_explicit_true_wins_over_environment(self):
        os.environ["TRACCIA_AUTO_START_TRACE"] = "0"
        init(enable_patching=False, auto_start_trace=True)
        self.assertIsNotNone(auto._auto_trace_context)

    def test_dotenv_counts_on_first_init(self):
        with open(os.path.join(self._dir.name, ".env"), "w", encoding="utf-8") as f:
            f.write("TRACCIA_AUTO_START_TRACE=true\nTRACCIA_AUTO_TRACE_NAME=from-dotenv\n")
        init(enable_patching=False)
        self.assertIsNotNone(auto._auto_trace_context)
        self.assertEqual(auto._auto_trace_name, "from-dotenv")

    def test_dotenv_ignored_with_load_env_false(self):
        with open(os.path.join(self._dir.name, ".env"), "w", encoding="utf-8") as f:
            f.write("TRACCIA_AUTO_START_TRACE=true\n")
        init(enable_patching=False, load_env=False)
        self.assertIsNone(auto._auto_trace_context)

    def test_explicit_name_wins_over_config_file(self):
        path = self._config('auto_trace_name = "from-file"\n')
        init(config_file=path, enable_patching=False, auto_trace_name="from-arg")
        self.assertEqual(auto._auto_trace_name, "from-arg")


class TestAutoTraceCleanup(unittest.TestCase):
    """Ending the auto root from another thread or under an active child (TRC-358)."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        cwd = os.getcwd()
        os.chdir(self._dir.name)  # no traccia.toml from this machine
        self.addCleanup(os.chdir, cwd)
        patcher = unittest.mock.patch.dict(os.environ, _clean_env(self._dir.name), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        init(enable_patching=False, enable_metrics=False, auto_start_trace=True, load_env=False)
        self.root = auto._auto_trace_context["span"]
        self._token, self._ctx = auto._auto_trace_context["activation"][:2]

    def tearDown(self):
        stop_tracing()
        # These tests leave the ended root in this thread's context on purpose; restore it for later tests.
        from opentelemetry import context
        if context.get_current() is self._ctx:
            context.detach(self._token)

    def _new_span_parent(self):
        span = get_tracer("test").start_span("after")
        span.end()
        return span.parent_span_id, span.context.trace_id

    def _end_without_errors(self, end_somewhere_else):
        # Neither Traccia nor OTel (which logs "Failed to detach context") may log an error.
        with self.assertNoLogs("traccia.auto", level="ERROR"), \
                self.assertNoLogs("opentelemetry.context", level="ERROR"):
            end_somewhere_else()
        self.assertIsNone(auto._auto_trace_context)
        self.assertTrue(self.root._ended)
        parent, trace_id = self._new_span_parent()
        self.assertIsNone(parent)
        self.assertNotEqual(trace_id, self.root.context.trace_id)

    def test_end_from_another_thread_does_not_fail_and_root_is_no_longer_a_parent(self):
        import threading

        def end_in_thread():
            t = threading.Thread(target=end_auto_trace)
            t.start()
            t.join()

        self._end_without_errors(end_in_thread)

    def test_end_from_an_asyncio_task_does_not_fail_and_root_is_no_longer_a_parent(self):
        async def child():
            end_auto_trace()

        self._end_without_errors(lambda: asyncio.run(child()))

    def test_trace_inside_an_active_child_keeps_the_child(self):
        from opentelemetry import context
        from opentelemetry.trace import get_current_span, set_span_in_context

        child = get_tracer("test").start_span("child")
        token = context.attach(set_span_in_context(child._otel_span))
        with trace("explicit"):
            pass
        # Back in the child, which trace() must not have dropped.
        self.assertEqual(get_current_span().get_span_context().span_id, child._otel_span.get_span_context().span_id)
        child.end()
        context.detach(token)
        # The ended root is still in this context, but it is not used as a parent.
        parent, trace_id = self._new_span_parent()
        self.assertIsNone(parent)
        self.assertNotEqual(trace_id, self.root.context.trace_id)

    def test_exit_hook_settles_calls_before_ending_the_root(self):
        order = []
        with unittest.mock.patch("traccia.governance.pep.settle_open_calls", lambda *a, **k: order.append("settle")), \
                unittest.mock.patch.object(auto, "_cleanup_auto_trace", lambda: order.append("end-root")):
            auto._end_auto_trace_at_exit()
        self.assertEqual(order, ["settle", "end-root"])


class TestLeakedActivation(unittest.TestCase):
    """A span made current by Traccia whose activation was undone in another thread (TRC-359)."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        cwd = os.getcwd()
        os.chdir(self._dir.name)
        self.addCleanup(os.chdir, cwd)
        patcher = unittest.mock.patch.dict(os.environ, _clean_env(self._dir.name), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        init(enable_patching=False, enable_metrics=False, load_env=False)

    def tearDown(self):
        stop_tracing()

    def test_stays_the_parent_until_it_ends_then_is_skipped(self):
        import threading
        from opentelemetry import context
        from traccia.utils.helpers import activate_span, deactivate_span

        tracer = get_tracer("test")
        outer = tracer.start_span("outer")
        outer_activation = activate_span(outer._otel_span)
        span = tracer.start_span("leaky")
        activation = activate_span(span._otel_span)
        t = threading.Thread(target=deactivate_span, args=(activation,))
        t.start()
        t.join()  # could not detach there: still current here

        child = tracer.start_span("while-running")
        self.assertEqual(child.parent_span_id, span.context.span_id)  # still recording: still the parent
        child.end()
        span.end()
        after = tracer.start_span("after-it-ended")
        self.assertEqual(after.parent_span_id, outer.context.span_id)  # the context before it is resumed
        self.assertEqual(tracer.get_current_span().context.span_id, outer.context.span_id)
        after.end()
        context.detach(activation[0])
        deactivate_span(outer_activation)
        outer.end()


class TestLeakedChainBound(unittest.TestCase):
    """Even past the resolver's bound, an ended leaked span is never returned (TRC-359)."""

    def test_exhausted_resolution_returns_no_span(self):
        from opentelemetry.context import Context
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import set_span_in_context
        from traccia.utils import helpers

        tracer = TracerProvider().get_tracer("t")
        ctx = Context()
        for i in range(helpers._MAX_SKIPS + 5):  # a chain no activation would build, built by hand
            span = tracer.start_span(f"s{i}", context=Context())
            span.end()
            setattr(span, helpers._LEAKED, ctx)
            ctx = set_span_in_context(span)
        span, resolved, skipped = helpers._effective(ctx)
        self.assertIsNone(span)
        self.assertTrue(skipped)
        self.assertEqual(dict(resolved), {})


_EXIT_SCRIPT = """
import sys
from traccia import init, observe, trace, stop_tracing
class Exp:
    def export(self, spans):
        for s in spans:
            print("SPAN", s.name, s.context.trace_id, s.context.span_id, s.parent_span_id, flush=True)
        return True
    def shutdown(self):
        pass
class OtelExp:
    # Stands in for OTLP's network exporter: the OTLP path (OTel's own provider and batch processor).
    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        for s in spans:
            parent = format(s.parent.span_id, "016x") if s.parent else "None"
            print("SPAN", s.name, format(s.context.trace_id, "032x"), format(s.context.span_id, "016x"), parent, flush=True)
        return SpanExportResult.SUCCESS
    def shutdown(self):
        pass
    def force_flush(self, timeout_millis=30000):
        return True
def exporter():
    if mode.endswith("otlp"):
        from traccia.exporter import OTLPExporter
        otlp = OTLPExporter(endpoint="http://127.0.0.1:9/v1/traces")
        otlp._otel_exporter = OtelExp()
        return otlp
    return Exp()
mode = sys.argv[1]
kwargs = {"on": {"auto_start_trace": True}, "off": {"auto_start_trace": False},
          "restart": {"auto_start_trace": True}, "restart-otlp": {"auto_start_trace": True}}.get(mode, {})
init(exporter=exporter(), enable_patching=False, enable_metrics=False, **kwargs)
@observe()
def inner(i):
    return i
@observe()
def run(i):
    return inner(i)
if mode == "grouped":
    with trace("run-group"):
        for i in range(3):
            run(i)
elif mode.startswith("restart"):
    run(0)
    stop_tracing()
    init(exporter=exporter(), enable_patching=False, enable_metrics=False, auto_start_trace=True, auto_trace_name="second")
    run(1)
else:
    for i in range(3):
        run(i)
# No stop_tracing() at the end: the app relies on the exit hooks.
"""


class TestAutoTraceAtExit(unittest.TestCase):
    """What a process exports for top-level calls, by auto_start_trace setting (TRC-358, TRC-359)."""

    def _run(self, mode: str, env_extra=None, toml=None, dotenv=None) -> list:
        with tempfile.TemporaryDirectory() as home:
            script = os.path.join(home, "app.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(_EXIT_SCRIPT)
            if toml is not None:
                with open(os.path.join(home, "traccia.toml"), "w", encoding="utf-8") as f:
                    f.write("[tracing]\n" + toml)
            if dotenv is not None:
                with open(os.path.join(home, ".env"), "w", encoding="utf-8") as f:
                    f.write(dotenv)
            # A fresh HOME and cwd, so no traccia.toml or .env on this machine is picked up.
            env = _clean_env(home)
            env.update(PYTHONPATH=os.pathsep.join(p for p in sys.path if p), **(env_extra or {}))
            result = subprocess.run(
                [sys.executable, script, mode], capture_output=True, text=True, timeout=60, cwd=home, env=env
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        # name, trace id, span id, parent span id
        return [line.split()[1:] for line in result.stdout.splitlines() if line.startswith("SPAN ")]

    def _assert_one_trace_under(self, spans, root_name, runs=3):
        roots = [s for s in spans if s[0] == root_name]
        self.assertEqual(len(roots), 1, spans)
        self.assertEqual(roots[0][3], "None")
        trace_id, root_id = roots[0][1], roots[0][2]
        self.assertEqual(len(spans), 1 + 2 * runs, spans)
        self.assertEqual({s[1] for s in spans}, {trace_id})
        run_spans = [s for s in spans if s[0] == "run"]
        self.assertEqual(sorted(s[3] for s in run_spans), [root_id] * runs)
        self._assert_inners_under_runs(spans, run_spans, runs)

    def _assert_inners_under_runs(self, spans, run_spans, runs):
        inners = [s for s in spans if s[0] == "inner"]
        self.assertEqual(len(inners), runs)
        by_id = {s[2]: s for s in run_spans}
        self.assertEqual(sorted(s[3] for s in inners), sorted(by_id))  # one inner under each run
        for s in inners:
            self.assertEqual(s[1], by_id[s[3]][1])

    def _assert_one_trace_per_call(self, spans):
        runs = [s for s in spans if s[0] == "run"]
        self.assertEqual(len(runs), 3)
        self.assertEqual(len(spans), 6, spans)  # no root span
        self.assertTrue(all(s[3] == "None" for s in runs))
        self.assertEqual(len({s[1] for s in runs}), 3)
        self._assert_inners_under_runs(spans, runs, 3)

    def test_default_gives_each_top_level_call_its_own_trace(self):
        self._assert_one_trace_per_call(self._run("default"))

    def test_disabled_gives_each_top_level_call_its_own_trace(self):
        self._assert_one_trace_per_call(self._run("off"))

    def test_enabled_exports_the_root_span_and_it_parents_every_call(self):
        self._assert_one_trace_under(self._run("on"), "root")

    def test_config_file_can_enable_it(self):
        self._assert_one_trace_under(self._run("default", toml="auto_start_trace = true\n"), "root")

    def test_environment_can_enable_it_without_a_config_file(self):
        spans = self._run("default", env_extra={"TRACCIA_AUTO_START_TRACE": "true", "TRACCIA_AUTO_TRACE_NAME": "job"})
        self._assert_one_trace_under(spans, "job")

    def test_dotenv_can_enable_it(self):
        spans = self._run("default", dotenv="TRACCIA_AUTO_START_TRACE=true\nTRACCIA_AUTO_TRACE_NAME=dotenv-job\n")
        self._assert_one_trace_under(spans, "dotenv-job")

    def test_explicit_false_wins_over_environment(self):
        self._assert_one_trace_per_call(self._run("off", env_extra={"TRACCIA_AUTO_START_TRACE": "true"}))

    def test_trace_groups_calls_into_one_trace(self):
        self._assert_one_trace_under(self._run("grouped"), "run-group")

    def test_root_after_stop_tracing_and_a_second_init_is_exported_too(self):
        for mode in ("restart", "restart-otlp"):
            with self.subTest(mode=mode):
                self._assert_restart(self._run(mode))

    def _assert_restart(self, spans):
        first = [s for s in spans if s[1] == next(r[1] for r in spans if r[0] == "root")]
        second = [s for s in spans if s[1] == next(r[1] for r in spans if r[0] == "second")]
        self._assert_one_trace_under(first, "root", runs=1)
        self._assert_one_trace_under(second, "second", runs=1)
        self.assertEqual(len(spans), 6, spans)


class TestTraceContextManager(unittest.TestCase):
    """Test the trace() context manager."""
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            stop_tracing()
        except Exception:
            pass
    
    def test_trace_context_manager_basic(self):
        """Test basic trace() context manager usage."""
        init(enable_patching=False, auto_start_trace=False)
        
        with trace("test-trace") as span:
            # Should have a valid span
            self.assertIsNotNone(span)
    
    def test_trace_context_manager_with_exception(self):
        """Test that trace() context manager handles exceptions."""
        init(enable_patching=False, auto_start_trace=False)
        
        with self.assertRaises(ValueError):
            with trace("test-trace"):
                raise ValueError("test error")
    
    def test_trace_context_manager_with_attributes(self):
        """Test trace() context manager with custom attributes."""
        init(enable_patching=False, auto_start_trace=False)
        
        with trace("test-trace", custom_attr="value") as span:
            # Should have a valid span
            self.assertIsNotNone(span)


if __name__ == "__main__":
    unittest.main()
