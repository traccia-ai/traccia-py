"""CLI exit codes stay off the quality-miss code when the command itself fails."""

import json
from unittest.mock import patch

from traccia.cli import main
from traccia.prompts.client import PromptFetchError


def test_unknown_flag_is_a_usage_error(tmp_path):
    out = tmp_path / "summary.json"
    code = main(["eval", "task.py", "--output", str(out), "--dataset", "demo", "--nope"])
    assert code == 2
    body = json.loads(out.read_text())
    assert body["exit_code"] == 2
    assert body["code"] == "bad_flags"


def test_missing_api_key_on_baseline_writes_a_summary(tmp_path):
    script = tmp_path / "task.py"
    script.write_text("def task(item):\n    return 'ok'\n")
    out = tmp_path / "summary.json"
    with patch(
        "traccia.eval.client.fetch_experiment",
        side_effect=PromptFetchError("API key is required"),
    ):
        code = main([
            "eval",
            str(script),
            "--output",
            str(out),
            "--dataset",
            "demo",
            "--baseline-experiment",
            "exp-1",
        ])
    assert code == 2
    body = json.loads(out.read_text())
    assert body["exit_code"] == 2
    assert body["code"] == "bad_flags"
    assert body["quality_gate"] is False


def test_script_exit_zero_writes_a_summary(tmp_path):
    script = tmp_path / "task.py"
    script.write_text("import sys\nsys.exit(0)\n")
    out = tmp_path / "summary.json"
    code = main(["eval", str(script), "--output", str(out), "--dataset", "demo"])
    assert code == 3
    body = json.loads(out.read_text())
    assert body["exit_code"] == 3
    assert body["code"] == "task_error"


def test_keyboard_interrupt_writes_a_summary(tmp_path):
    script = tmp_path / "task.py"
    script.write_text("raise KeyboardInterrupt\n")
    out = tmp_path / "summary.json"
    code = main(["eval", str(script), "--output", str(out), "--dataset", "demo"])
    assert code == 130
    body = json.loads(out.read_text())
    assert body["exit_code"] == 130
    assert body["code"] == "task_error"


def test_create_experiment_retries_a_server_error(monkeypatch):
    from traccia.eval.client import create_experiment
    from traccia.eval.errors import EvaluateError

    calls = {"n": 0}

    def fake(method, path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EvaluateError("Eval runtime 503: unavailable")
        return {"id": "exp"}

    monkeypatch.setattr("traccia.eval.client._request", fake)
    monkeypatch.setattr("traccia.eval.client.time.sleep", lambda _seconds: None)
    assert create_experiment(dataset_id="ds")["id"] == "exp"
    assert calls["n"] == 2


def test_create_experiment_does_not_retry_a_client_error(monkeypatch):
    from traccia.eval.client import create_experiment
    from traccia.eval.errors import EvaluateError

    calls = {"n": 0}

    def fake(method, path, **kwargs):
        calls["n"] += 1
        raise EvaluateError("Eval runtime 422: no")

    monkeypatch.setattr("traccia.eval.client._request", fake)
    try:
        create_experiment(dataset_id="ds")
    except EvaluateError as exc:
        assert "422" in str(exc)
    else:
        raise AssertionError("expected the client error")
    assert calls["n"] == 1


def _passing_result():
    from types import SimpleNamespace

    return SimpleNamespace(
        rows=[{
            "item_id": "a",
            "panels": [{"scores": [{"name": "exact_match", "scorer_name": "exact_match", "status": "judged", "passed": True, "score": 1}]}],
        }],
        errors=[],
        run_context={"dataset_verified": True},
        experiment_id="exp-1",
        url="http://example.test/exp-1",
        persist_error=None,
    )


def test_dataclass_task_with_postponed_annotations_loads(tmp_path):
    script = tmp_path / "task.py"
    script.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Node:\n"
        "    child: Node | None = None\n\n"
        "def task(item):\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.json"
    with patch("traccia.eval.evaluate", return_value=_passing_result()) as evaluate:
        code = main(["eval", str(script), "--output", str(out), "--dataset", "demo", "--scorer", "exact_match"])
    assert code == 0
    evaluate.assert_called_once()
    assert "NoneType" not in out.read_text(encoding="utf-8")


def test_provider_keys_are_forwarded_and_not_written_to_the_summary(tmp_path, monkeypatch):
    script = tmp_path / "task.py"
    script.write_text("def task(item):\n    return 'ok'\n", encoding="utf-8")
    out = tmp_path / "summary.json"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-forward")
    monkeypatch.setenv("TRACCIA_PROVIDER_KEYS", '{"typesafe": "ts-test-forward"}')
    with patch("traccia.eval.evaluate", return_value=_passing_result()) as evaluate:
        code = main(["eval", str(script), "--output", str(out), "--dataset", "demo", "--scorer", "exact_match"])
    assert code == 0
    forwarded = evaluate.call_args.kwargs["provider_keys"]
    assert forwarded["openai"] == "sk-test-forward"
    assert forwarded["typesafe"] == "ts-test-forward"
    text = out.read_text(encoding="utf-8")
    assert "sk-test-forward" not in text
    assert "ts-test-forward" not in text


def test_invalid_provider_keys_do_not_echo_the_secret(tmp_path, monkeypatch, capsys):
    script = tmp_path / "task.py"
    script.write_text("def task(item):\n    return 'ok'\n", encoding="utf-8")
    out = tmp_path / "summary.json"
    monkeypatch.setenv("TRACCIA_PROVIDER_KEYS", '{"openai": "sk-test-secret"')
    code = main(["eval", str(script), "--output", str(out), "--dataset", "demo", "--scorer", "exact_match"])
    assert code == 2
    captured = capsys.readouterr()
    body = out.read_text(encoding="utf-8")
    assert "sk-test-secret" not in captured.err
    assert "sk-test-secret" not in body
    assert json.loads(body)["code"] == "bad_flags"
