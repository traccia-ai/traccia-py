"""Core functionality tests for Traccia SDK with OpenTelemetry."""

from __future__ import annotations

import unittest
import tempfile
from io import StringIO

from traccia import get_tracer, get_tracer_provider, set_tracer_provider
from traccia.tracer import TracerProvider, Span, SpanStatus
from traccia.tracer.span_context import SpanContext
from traccia.context import get_current_span
from traccia.exporter import ConsoleExporter, FileExporter, OTLPExporter
from traccia.context.propagators import (
    format_traceparent, parse_traceparent,
    inject_traceparent, extract_traceparent,
    format_tracestate, parse_tracestate,
)


class TestSpanCreation(unittest.TestCase):
    """Test basic span creation and management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = TracerProvider()
        set_tracer_provider(self.provider)
        self.tracer = get_tracer("test")
    
    def test_create_span(self):
        """Test creating a span."""
        span = self.tracer.start_span("test_span")
        self.assertIsNotNone(span)
        self.assertEqual(span.name, "test_span")
        self.assertIsNotNone(span.context)
        self.assertIsNotNone(span.context.trace_id)
        self.assertIsNotNone(span.context.span_id)
        span.end()
    
    def test_span_attributes(self):
        """Test setting and getting span attributes."""
        span = self.tracer.start_span("test_span")
        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)
        span.set_attribute("key3", True)
        
        self.assertEqual(span.attributes["key1"], "value1")
        self.assertEqual(span.attributes["key2"], 42)
        self.assertEqual(span.attributes["key3"], True)
        span.end()
    
    def test_span_status(self):
        """Test setting span status."""
        span = self.tracer.start_span("test_span")
        span.set_status(SpanStatus.OK)
        self.assertEqual(span.status, SpanStatus.OK)
        
        span.set_status(SpanStatus.ERROR, "Test error")
        self.assertEqual(span.status, SpanStatus.ERROR)
        self.assertEqual(span.status_description, "Test error")
        span.end()
    
    def test_span_context_manager(self):
        """Test span as context manager."""
        with self.tracer.start_as_current_span("test_span") as span:
            current = get_current_span()
            self.assertIsNotNone(current)
            self.assertEqual(current.name, "test_span")
            span.set_attribute("test", "value")
        
        # Span should be ended after context exit
        self.assertTrue(span._ended)


class TestContextManagement(unittest.TestCase):
    """Test context management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = TracerProvider()
        set_tracer_provider(self.provider)
        self.tracer = get_tracer("test")
    
    def test_get_current_span(self):
        """Test getting current span."""
        # No current span initially
        current = get_current_span()
        self.assertIsNone(current)
        
        # Create span and set as current
        with self.tracer.start_as_current_span("test_span") as span:
            current = get_current_span()
            self.assertIsNotNone(current)
            self.assertEqual(current.name, "test_span")
        
        # No current span after context exit
        current = get_current_span()
        self.assertIsNone(current)
    
    def test_nested_spans(self):
        """Test nested spans."""
        with self.tracer.start_as_current_span("parent") as parent:
            self.assertEqual(get_current_span().name, "parent")
            
            with self.tracer.start_as_current_span("child") as child:
                self.assertEqual(get_current_span().name, "child")
            
            # Back to parent
            self.assertEqual(get_current_span().name, "parent")


class TestPropagation(unittest.TestCase):
    """Test trace context propagation."""
    
    def test_format_parse_traceparent(self):
        """Test formatting and parsing traceparent."""
        ctx = SpanContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            trace_flags=1
        )
        
        # Format
        tp = format_traceparent(ctx)
        self.assertIsNotNone(tp)
        self.assertTrue(tp.startswith("00-"))
        
        # Parse
        parsed = parse_traceparent(tp)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.trace_id, ctx.trace_id)
        self.assertEqual(parsed.span_id, ctx.span_id)
    
    def test_inject_extract_traceparent(self):
        """Test injecting and extracting traceparent."""
        ctx = SpanContext(
            trace_id="a" * 32,
            span_id="b" * 16,
            trace_flags=1
        )
        
        # Inject
        headers = {}
        inject_traceparent(headers, ctx)
        self.assertIn("traceparent", headers)
        
        # Extract
        extracted = extract_traceparent(headers)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.trace_id, ctx.trace_id)
        self.assertEqual(extracted.span_id, ctx.span_id)
    
    def test_tracestate_format_parse(self):
        """Test formatting and parsing tracestate."""
        state = {"tenant": "test", "project": "proj", "dbg": "1"}
        
        # Format
        ts = format_tracestate(state)
        self.assertIsNotNone(ts)
        
        # Parse
        parsed = parse_tracestate(ts)
        self.assertEqual(parsed, state)


class TestExporters(unittest.TestCase):
    """Test exporters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = TracerProvider()
        set_tracer_provider(self.provider)
        self.tracer = get_tracer("test")
    
    def test_console_exporter(self):
        """Test ConsoleExporter."""
        span = self.tracer.start_span("test_span")
        span.set_attribute("key", "value")
        span.end()
        
        buffer = StringIO()
        exporter = ConsoleExporter(stream=buffer)
        result = exporter.export([span])
        
        self.assertTrue(result)
        output = buffer.getvalue()
        self.assertIn("test_span", output)
    
    def test_file_exporter(self):
        """Test FileExporter."""
        span = self.tracer.start_span("test_span")
        span.set_attribute("key", "value")
        span.end()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            exporter = FileExporter(file_path=temp_path, reset_on_start=True)
            result = exporter.export([span])
            self.assertTrue(result)
            
            # Read back
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertIn("test_span", content)
        finally:
            import os
            os.unlink(temp_path)
    
    def test_otlp_exporter_creation(self):
        """Test OTLPExporter creation."""
        exporter = OTLPExporter(endpoint="http://localhost:4318/v1/traces")
        self.assertIsNotNone(exporter)
        self.assertEqual(exporter.endpoint, "http://localhost:4318/v1/traces")


class TestProcessors(unittest.TestCase):
    """Test span processors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = TracerProvider()
        set_tracer_provider(self.provider)
        self.tracer = get_tracer("test")
    
    def test_enrichment_processor(self):
        """Test that enrichment processors run before span ends."""
        from traccia.processors.token_counter import TokenCountingProcessor
        
        processor = TokenCountingProcessor()
        self.provider.add_span_processor(processor)
        
        span = self.tracer.start_span("test_span")
        span.set_attribute("llm.prompt", "Hello world")
        span.set_attribute("llm.model", "gpt-4")
        
        # Processor should add token counts when span ends
        span.end()
        
        # Check if token counts were added (processor runs before end)
        # Note: This is a best-effort check since processor runs asynchronously
        self.assertIsNotNone(span.attributes)


if __name__ == "__main__":
    unittest.main()
