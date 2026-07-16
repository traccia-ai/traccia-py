"""Tests for auto-start trace functionality."""

import unittest
import asyncio
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
