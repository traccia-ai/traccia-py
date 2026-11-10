"""Unit tests for traccia.eval (builtins + local-only evaluate)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traccia.eval.builtins import run_builtin_scorer
from traccia.eval.evaluate import EvaluateResult, bind_scorer_name, evaluate, _score_span_name
from traccia.eval.errors import EvaluateError


def test_builtin_exact_match():
    assert run_builtin_scorer("exact_match", output=" Hi ", expected="hi")["passed"] is True
    assert run_builtin_scorer("exact_match", output="a", expected="b")["passed"] is False


def test_builtin_contains_and_json():
    assert run_builtin_scorer("contains", output="hello world", expected="world")["passed"]
    assert run_builtin_scorer("json_valid", output='{"a":1}')["passed"]
    assert run_builtin_scorer("json_valid", output="nope")["passed"] is False


def test_evaluate_empty_raises():
    with pytest.raises(EvaluateError):
        evaluate("x", data=[], task=lambda inp: "y", persist=False, progress=False)


def test_builtin_stamps_type_and_scorer_name():
    scored = run_builtin_scorer("contains", output="hello world", expected="world")
    assert scored["type"] == "contains"
    assert scored["scorer_name"] == "contains"
    assert scored["name"] == "contains"
    assert scored["status"] == "judged"


def test_evaluate_local_only():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=True), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "abc"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        result = evaluate(
            "local-smoke",
            data=[{"input": {"q": "hi"}, "expected": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            persist=False,
            progress=False,
            max_concurrency=2,
        )
    assert isinstance(result, EvaluateResult)
    assert result.url is None
    assert result.aggregates["item_count"] == 1
    assert result.aggregates["pass_count"] == 1
    assert "local-smoke" in result.summary()
    assert result.rows[0]["panels"][0].get("trace_id") == "abc"
    panel = result.rows[0]["panels"][0]
    assert panel.get("label") == "Task"
    assert panel.get("source") == "evaluate"
    assert panel["scores"][0].get("type") == "exact_match"
    assert panel["scores"][0].get("scorer_name") == "exact_match"
    assert isinstance(panel.get("latency_ms"), (int, float))
    assert result.aggregates.get("source") == "evaluate"
    span_names = [c.args[0] for c in tracer.start_as_current_span.call_args_list]
    assert "evaluate.item" in span_names
    assert "scorer.exact_match" in span_names


def test_evaluate_skips_trace_id_without_export():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=False), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "phantom"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        result = evaluate(
            "no-export",
            data=[{"input": {"q": "hi"}, "expected": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            persist=False,
            progress=False,
        )
    assert result.rows[0]["panels"][0].get("trace_id") is None


def test_evaluate_error_isolation():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=True), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "t1"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        def task(inp):
            if inp.get("q") == "boom":
                raise RuntimeError("task failed")
            return inp["q"]

        result = evaluate(
            "iso",
            data=[
                {"input": {"q": "ok"}, "expected": "ok"},
                {"input": {"q": "boom"}, "expected": "x"},
            ],
            task=task,
            scorers=["exact_match"],
            persist=False,
            progress=False,
        )
    assert len(result.rows) == 2
    assert len(result.errors) == 1
    assert result.rows[0]["panels"][0]["passed"] is True
    assert result.rows[1]["panels"][0]["error"]


def test_evaluate_prompt_label_and_expected_output_alias():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=False), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "t1"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        labeled = evaluate(
            "labeled",
            data=[{"input": {"q": "hi"}, "expected": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            prompt="support-reply",
            persist=False,
            progress=False,
        )
        aliased = evaluate(
            "alias",
            data=[{"input": {"q": "hi"}, "expected_output": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            persist=False,
            progress=False,
        )
    assert labeled.rows[0]["panels"][0]["label"] == "support-reply"
    assert aliased.aggregates["pass_count"] == 1


def test_platform_jev_score_keeps_questions():
    from traccia.eval.evaluate import _run_scorer

    remote = {
        "scorer_id": "scorer-1",
        "scorer_name": "Answer Quality",
        "type": "jev",
        "passed": False,
        "score": 0.4,
        "model": "jev-1.13.0",
        "questions": [{"key": "grounded", "label": "Grounded", "passed": False, "score": 0.1}],
        "unsure": False,
    }
    with patch("traccia.eval.evaluate.eval_client.score_remote", return_value=remote):
        scored = _run_scorer(
            "Answer Quality",
            row={"input": "q", "expected_output": "a"},
            output="out",
            provider_keys={"typesafe": "k"},
            scorer_cache={
                "Answer Quality": {
                    "id": "scorer-1",
                    "name": "Answer Quality",
                    "type": "jev",
                    "config": {},
                }
            },
            cred={},
        )
    assert scored["questions"][0]["label"] == "Grounded"
    assert scored["unsure"] is False


def test_platform_judge_score_omits_jev_fields():
    from traccia.eval.evaluate import _run_scorer

    remote = {
        "scorer_id": "judge-1",
        "scorer_name": "Helpfulness",
        "type": "llm_judge",
        "passed": True,
        "score": 1,
        "reason": "The reply answers the question.",
        "model": "gpt-4o-mini",
        "latency_ms": 120,
        "cost_usd": 0.001,
    }
    with patch("traccia.eval.evaluate.eval_client.score_remote", return_value=remote):
        scored = _run_scorer(
            "Helpfulness",
            row={"input": "q", "expected_output": "a"},
            output="out",
            provider_keys={"openai": "k"},
            scorer_cache={
                "Helpfulness": {
                    "id": "judge-1",
                    "name": "Helpfulness",
                    "type": "llm_judge",
                    "config": {},
                }
            },
            cred={},
        )
    assert scored["passed"] is True
    assert scored["score"] == 1
    assert scored["reason"] == "The reply answers the question."
    assert scored["model"] == "gpt-4o-mini"
    assert "questions" not in scored
    assert "unsure" not in scored


def test_score_span_name_and_judge_annotation():
    assert _score_span_name("Helpfulness") == "scorer.Helpfulness"
    span = MagicMock()
    from traccia.eval.evaluate import _annotate_score_span

    _annotate_score_span(
        span,
        {
            "type": "llm_judge",
            "scorer_name": "Helpfulness",
            "passed": True,
            "model": "gemini-3.6-flash",
            "cost_usd": 0.001,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
    )
    attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    assert attrs["span.type"] == "llm"
    assert attrs["llm.model"] == "gemini-3.6-flash"
    assert attrs["traccia.eval.scorer"] == "Helpfulness"
    assert attrs["llm.usage.total_tokens"] == 14


def _quiet_evaluate(**kwargs):
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=False), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer
        return evaluate(persist=False, progress=False, **kwargs)


def test_mapping_key_is_the_recorded_scorer_name():
    def score_accuracy(input, output, expected, metadata):
        return {"passed": True, "score": 1.0}

    result = _quiet_evaluate(
        name="alias",
        data=[{"input": {"q": "a"}, "expected": "a"}],
        task=lambda inp: inp["q"],
        scorers=[bind_scorer_name(score_accuracy, "accuracy")],
    )
    score = result.rows[0]["panels"][0]["scores"][0]
    assert score["name"] == "accuracy"
    from traccia.eval.gate import report_from_rows

    reported = report_from_rows(result.rows, scorers=["local:accuracy"])
    assert reported[0]["name"] == "accuracy"
    assert reported[0]["missing"] == 0
    assert reported[0]["completed"] == 1


def test_local_scorer_passed_flag_keeps_the_spec_name():
    def score_accuracy(input, output, expected, metadata):
        return {"passed": True, "name": "not-the-key"}

    result = _quiet_evaluate(
        name="alias",
        data=[{"input": {"q": "a"}, "expected": "a"}],
        task=lambda inp: inp["q"],
        scorers=[bind_scorer_name(score_accuracy, "accuracy")],
    )
    score = result.rows[0]["panels"][0]["scores"][0]
    assert score["name"] == "accuracy"
    assert score["scorer_name"] == "accuracy"
    assert score["score"] == 1.0
    from traccia.eval.gate import report_from_rows

    reported = report_from_rows(result.rows, scorers=["local:accuracy"])
    assert reported[0]["completed"] == 1
    assert reported[0]["errored"] == 0


def test_platform_builtin_keeps_fetched_identity():
    fetched = {
        "id": "scorer-2",
        "name": "wraps-exact-match",
        "type": "exact_match",
        "definition_hash": "sha256:platform",
        "implementation_version": "1",
        "scorer_snapshot_token": "token",
        "config": {},
    }
    with patch("traccia.eval.evaluate.eval_client.fetch_scorer", return_value=fetched), patch(
        "traccia.eval.evaluate.eval_client.score_remote"
    ) as remote:
        result = _quiet_evaluate(
            name="platform-builtin",
            data=[{"input": {"q": "hi"}, "expected": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["wraps-exact-match"],
        )
    remote.assert_not_called()
    score = result.rows[0]["panels"][0]["scores"][0]
    assert score["passed"] is True
    assert score["scorer_name"] == "wraps-exact-match"
    assert score["scorer_id"] == "scorer-2"
    assert score["definition_hash"] == "sha256:platform"
    assert score["scorer_snapshot_token"] == "token"
    assert "definition_identity" not in score


def test_task_cost_reads_mapping_attributes_and_provider_children():
    from types import MappingProxyType, SimpleNamespace

    from traccia.eval.evaluate import _cost_from_span

    priced = SimpleNamespace(attributes=MappingProxyType({"llm.cost.usd": 0.25}))
    assert _cost_from_span(priced) == 0.25
    child = SimpleNamespace(
        name="chat.completions",
        attributes={"llm.cost.usd": 0.1, "llm.model": "gpt-4o-mini"},
    )
    parent = SimpleNamespace(name="evaluate.item", attributes=MappingProxyType({}), children=[child])
    assert _cost_from_span(parent) == 0.1


def test_provider_child_cost_is_collected_from_the_sdk_provider():
    import importlib
    from types import SimpleNamespace

    import traccia

    evaluate_mod = importlib.import_module("traccia.eval.evaluate")

    evaluate_mod._ensure_child_cost_processor()
    token = evaluate_mod._task_child_costs.set([])
    try:
        tracer = traccia.get_tracer("traccia.eval.test")
        with tracer.start_as_current_span("chat.completions") as span:
            span.set_attribute("llm.model", "gpt-4o-mini")
            span.set_attribute("llm.cost.usd", 0.25)
        captured = list(evaluate_mod._task_child_costs.get() or [])
    finally:
        evaluate_mod._task_child_costs.reset(token)
    assert captured == [{"cost": 0.25}]
    parent = SimpleNamespace(name="evaluate.item", attributes={})
    assert evaluate_mod._cost_from_span(parent, captured) == 0.25
