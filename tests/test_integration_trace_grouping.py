"""Framework integrations group a run into one trace without an auto-started root (TRC-359)."""

import os
import tempfile
import unittest
import unittest.mock
import uuid
from types import SimpleNamespace

from traccia import stop_tracing
from traccia.auto import start_tracing


class _Capture:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return True

    def shutdown(self):
        pass


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

    def tearDown(self):
        stop_tracing()

    def exported(self):
        stop_tracing()  # exports what is queued
        return self.capture.spans

    def assert_tree(self, spans, root_name, children):
        """One trace: root_name has no parent; children maps span name -> parent span name."""
        by_name = {s.name: s for s in spans}
        self.assertEqual(sorted(by_name), sorted([root_name, *children]), [s.name for s in spans])
        root = by_name[root_name]
        self.assertIsNone(root.parent_span_id)
        self.assertEqual({s.context.trace_id for s in spans}, {root.context.trace_id})
        for name, parent in children.items():
            self.assertEqual(by_name[name].parent_span_id, by_name[parent].context.span_id, name)


class TestOpenAIAgentsGrouping(_TracingTest):
    def _run(self, processor, trace_id):
        trace = SimpleNamespace(trace_id=trace_id, name="Agent workflow")
        agent = SimpleNamespace(span_id=f"{trace_id}-a", trace_id=trace_id, parent_id=None,
                                span_data=SimpleNamespace(type="agent", name="triage"), error=None)
        tool = SimpleNamespace(span_id=f"{trace_id}-t", trace_id=trace_id, parent_id=agent.span_id,
                               span_data=SimpleNamespace(type="function", name="lookup"), error=None)
        handoff = SimpleNamespace(span_id=f"{trace_id}-h", trace_id=trace_id, parent_id=agent.span_id,
                                  span_data=SimpleNamespace(type="handoff"), error=None)
        processor.on_trace_start(trace)
        for span in (agent, tool, handoff):
            processor.on_span_start(span)
        for span in (tool, handoff, agent):
            processor.on_span_end(span)
        processor.on_trace_end(trace)

    def test_each_agents_trace_is_one_traccia_trace(self):
        from traccia.integrations.openai_agents.processor import TracciaAgentsTracingProcessor

        processor = TracciaAgentsTracingProcessor()
        self._run(processor, "trace_1")
        self._run(processor, "trace_2")
        spans = self.exported()

        self.assertEqual(len(spans), 8, [s.name for s in spans])
        traces = {}
        for s in spans:
            traces.setdefault(s.context.trace_id, []).append(s)
        self.assertEqual(len(traces), 2)
        for group in traces.values():
            self.assert_tree(group, "Agent workflow", {
                "agent.triage": "Agent workflow",
                "agent.tool.lookup": "agent.triage",
                "agent.handoff": "agent.triage",
            })


class TestLangChainGrouping(_TracingTest):
    def setUp(self):
        super().setUp()
        from traccia.integrations.langchain import CallbackHandler

        self.handler = CallbackHandler()

    def _chat(self, run_id, parent_run_id):
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        self.handler.on_chat_model_start({"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4o-mini"}},
                                         [[HumanMessage(content="hi")]], run_id=run_id, parent_run_id=parent_run_id)
        self.handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="hello"))]]),
                                run_id=run_id, parent_run_id=parent_run_id)

    def _invoke_chain(self):
        root, step = uuid.uuid4(), uuid.uuid4()
        self.handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=root, parent_run_id=None)
        self.handler.on_chain_start({"name": "RunnableLambda"}, {}, run_id=step, parent_run_id=root)
        self._chat(uuid.uuid4(), parent_run_id=step)  # inside a nested run: joins the root chain's span
        self.handler.on_chain_end({}, run_id=step, parent_run_id=root)
        self._chat(uuid.uuid4(), parent_run_id=root)
        self.handler.on_chain_end({}, run_id=root, parent_run_id=None)

    def test_each_chain_invocation_is_one_trace(self):
        self._invoke_chain()
        self._invoke_chain()
        spans = self.exported()

        self.assertEqual(len(spans), 6, [s.name for s in spans])
        traces = {}
        for s in spans:
            traces.setdefault(s.context.trace_id, []).append(s)
        self.assertEqual(len(traces), 2)
        for group in traces.values():
            roots = [s for s in group if s.name == "chain.langchain.run"]
            self.assertEqual(len(roots), 1)
            self.assertIsNone(roots[0].parent_span_id)
            self.assertEqual(roots[0].attributes.get("langchain.chain.name"), "RunnableSequence")
            llms = [s for s in group if s.name == "llm.langchain.run"]
            self.assertEqual([s.parent_span_id for s in llms], [roots[0].context.span_id] * 2)

    def test_nested_chains_get_no_span_of_their_own(self):
        self._invoke_chain()
        names = sorted(s.name for s in self.exported())
        self.assertEqual(names, ["chain.langchain.run", "llm.langchain.run", "llm.langchain.run"])

    def test_llm_called_directly_is_its_own_trace(self):
        self._chat(uuid.uuid4(), parent_run_id=None)
        self._chat(uuid.uuid4(), parent_run_id=None)
        spans = self.exported()
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(s.parent_span_id is None for s in spans))
        self.assertEqual(len({s.context.trace_id for s in spans}), 2)

    def _real_chain(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.runnables import RunnableLambda
        from traccia import observe

        @observe()
        def work(text):
            return text + "!"

        return RunnableLambda(work) | FakeListChatModel(responses=["hello"])

    def _assert_real_chain_trace(self, spans):
        by_name = {s.name: s for s in spans}
        self.assertEqual(sorted(by_name), ["chain.langchain.run", "llm.langchain.run", "work"], [s.name for s in spans])
        root = by_name["chain.langchain.run"]
        self.assertIsNone(root.parent_span_id)
        self.assertEqual({s.context.trace_id for s in spans}, {root.context.trace_id})
        self.assertEqual(by_name["work"].parent_span_id, root.context.span_id)
        self.assertEqual(by_name["llm.langchain.run"].parent_span_id, root.context.span_id)

    def test_real_runnable_puts_observed_work_and_llm_call_in_the_chain_trace(self):
        from opentelemetry.trace import get_current_span

        with self.assertNoLogs("opentelemetry.context", level="ERROR"):
            self._real_chain().invoke("hi", config={"callbacks": [self.handler]})
        self.assertFalse(get_current_span().get_span_context().is_valid)  # chain span no longer current
        self._assert_real_chain_trace(self.exported())

    def test_real_runnable_async(self):
        import asyncio

        async def main():
            await self._real_chain().ainvoke("hi", config={"callbacks": [self.handler]})

        with self.assertNoLogs("opentelemetry.context", level="ERROR"):
            asyncio.run(main())
        self._assert_real_chain_trace(self.exported())

    def _effective_parent(self):
        """The span a new Traccia span, the public lookup and governance see here (None: none)."""
        from traccia import get_tracer
        from traccia.context import get_current_span
        from traccia.governance import pep

        tracer_span = get_tracer("test").get_current_span()
        public_span = get_current_span()
        trace_id, span_id = pep._trace_ids()
        tracer_id = tracer_span.context.span_id if tracer_span else None
        public_id = public_span.context.span_id if public_span else None
        self.assertEqual(tracer_id, public_id)
        self.assertEqual(span_id, public_id)
        self.assertEqual(trace_id, public_span.context.trace_id if public_span else None)
        return public_id

    def test_two_async_invocations_in_one_coroutine_are_two_traces(self):
        # LangChain ends async chains in another task, so the chain span cannot be detached in
        # this coroutine; it must still not become the parent of what runs next.
        import asyncio

        chain, parents = self._real_chain(), []

        async def main():
            for text in ("a", "b"):
                await chain.ainvoke(text, config={"callbacks": [self.handler]})
                parents.append(self._effective_parent())

        with self.assertNoLogs("opentelemetry.context", level="ERROR"):
            asyncio.run(main())
        self.assertEqual(parents, [None, None])  # tracer, public lookup and governance agree
        spans = self.exported()
        self.assertEqual(len(spans), 6, [s.name for s in spans])
        traces = {}
        for s in spans:
            traces.setdefault(s.context.trace_id, []).append(s)
        self.assertEqual(len(traces), 2)
        for group in traces.values():
            self._assert_real_chain_trace(group)

    def test_many_async_invocations_in_one_coroutine_stay_separate_traces(self):
        # More than the resolver's 64-step bound: leaked activations must not chain up.
        import asyncio
        from opentelemetry.trace import get_current_span as otel_current
        from traccia.utils.helpers import _LEAKED

        chain, parents, chain_lengths = self._real_chain(), [], []

        async def main():
            for i in range(70):
                await chain.ainvoke(str(i), config={"callbacks": [self.handler]})
                parents.append(self._effective_parent())
                leaked = otel_current()
                previous = getattr(leaked, _LEAKED, None)
                # What a leaked chain span resumes is never itself a leaked span.
                chain_lengths.append(previous is not None and getattr(otel_current(previous), _LEAKED, None) is None)

        asyncio.run(main())
        self.assertEqual(parents, [None] * 70)
        self.assertEqual(chain_lengths, [True] * 70)
        spans = self.exported()
        roots = [s for s in spans if s.name == "chain.langchain.run"]
        self.assertEqual(len(roots), 70)
        self.assertTrue(all(s.parent_span_id is None for s in roots))
        self.assertEqual(len({s.context.trace_id for s in spans}), 70)

    def test_two_async_invocations_inside_an_outer_span_both_join_it(self):
        import asyncio
        from traccia import trace

        chain, parents = self._real_chain(), []

        async def main():
            with trace("outer") as outer:
                for text in ("a", "b"):
                    await chain.ainvoke(text, config={"callbacks": [self.handler]})
                    parents.append(self._effective_parent())
            return outer.context.span_id

        outer_id = asyncio.run(main())
        self.assertEqual(parents, [outer_id, outer_id])  # the outer span is the parent again after each call
        spans = self.exported()
        self.assertEqual(len(spans), 7, [s.name for s in spans])
        self.assertEqual(len({s.context.trace_id for s in spans}), 1)
        roots = [s for s in spans if s.name == "chain.langchain.run"]
        self.assertEqual([s.parent_span_id for s in roots], [outer_id, outer_id])

    def test_two_real_invocations_are_two_traces(self):
        chain = self._real_chain()
        chain.invoke("a", config={"callbacks": [self.handler]})
        chain.invoke("b", config={"callbacks": [self.handler]})
        spans = self.exported()
        self.assertEqual(len(spans), 6)
        self.assertEqual(len({s.context.trace_id for s in spans}), 2)

    def test_chain_error_ends_its_span_with_an_error(self):
        root = uuid.uuid4()
        self.handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=root, parent_run_id=None)
        self.handler.on_chain_error(ValueError("boom"), run_id=root, parent_run_id=None)
        [span] = self.exported()
        self.assertEqual(span.name, "chain.langchain.run")
        self.assertEqual(str(span.status).split(".")[-1], "ERROR")


if __name__ == "__main__":
    unittest.main()
