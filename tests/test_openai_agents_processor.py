"""Tests for OpenAI Agents SDK span mapping."""

from __future__ import annotations

from types import SimpleNamespace

from traccia.integrations.openai_agents.processor import TracciaAgentsTracingProcessor


def _function_span_data(name: str = "check_credit_score", **extra):
    return SimpleNamespace(type="function", name=name, **extra)


class TestAgentsToolSpanTyping:
    def setup_method(self):
        self.processor = TracciaAgentsTracingProcessor()

    def test_function_span_name_uses_agent_tool_prefix(self):
        span_data = _function_span_data("verify_employment")
        assert self.processor._get_span_name(span_data) == "agent.tool.verify_employment"

    def test_function_span_sets_span_type_tool(self):
        span_data = _function_span_data("calculate_dti_ratio")
        attrs = self.processor._extract_attributes(span_data)

        assert attrs["span.type"] == "tool"
        assert attrs["agent.span.type"] == "function"
        assert attrs["agent.tool.name"] == "calculate_dti_ratio"
        assert attrs["tool.name"] == "calculate_dti_ratio"

    def test_function_span_without_name_still_typed_as_tool(self):
        span_data = SimpleNamespace(type="function")
        attrs = self.processor._extract_attributes(span_data)

        assert attrs["span.type"] == "tool"
        assert attrs["agent.span.type"] == "function"
        assert "agent.tool.name" not in attrs
        assert "tool.name" not in attrs

    def test_non_function_spans_do_not_get_tool_type(self):
        agent_attrs = self.processor._extract_attributes(
            SimpleNamespace(type="agent", name="LoanApprovalAgent")
        )
        generation_attrs = self.processor._extract_attributes(
            SimpleNamespace(type="generation", model="gpt-4o")
        )

        assert agent_attrs.get("span.type") != "tool"
        assert generation_attrs.get("span.type") != "tool"

    def test_on_span_start_attaches_tool_type_to_created_span(self):
        created = {}

        class FakeTracer:
            def start_span(self, name, attributes=None):
                created["name"] = name
                created["attributes"] = dict(attributes or {})
                return SimpleNamespace(attributes=created["attributes"], end=lambda: None)

        self.processor._tracer = FakeTracer()
        agents_span = SimpleNamespace(
            span_id="span-1",
            span_data=_function_span_data("check_credit_score"),
        )

        self.processor.on_span_start(agents_span)

        assert created["name"] == "agent.tool.check_credit_score"
        assert created["attributes"]["span.type"] == "tool"
        assert created["attributes"]["tool.name"] == "check_credit_score"
        assert "span-1" in self.processor._span_map
