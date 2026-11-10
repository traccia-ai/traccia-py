"""CI gate exit codes. The fixture is the contract shared with the Node SDK."""

from __future__ import annotations

import json
from pathlib import Path

from traccia.eval.gate import decide, preflight, report_from_rows

FIXTURE = Path(__file__).parent / "fixtures" / "ci_gate_cases.json"


def test_shared_fixture_codes():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert cases
    for case in cases:
        outcome = decide(case["report"])
        assert outcome["exit_code"] == case["exit_code"], case["name"]
        if "code" in case:
            assert outcome["code"] == case["code"], case["name"]
        if "quality_gate" in case:
            assert outcome["quality_gate"] is case["quality_gate"], case["name"]


def test_preflight_rules():
    assert preflight({"scorers": [], "min_pass_rate": 0.9})["code"] == "no_scorer"
    assert preflight({"scorers": ["exact_match"], "max_drop": 0.05})["code"] == "baseline_required"
    assert preflight({
        "scorers": ["exact_match"],
        "baseline_prompt": "support",
        "baseline_experiment": "exp",
        "max_drop": 0.05,
    })["code"] == "baseline_conflict"
    assert preflight({"scorers": ["exact_match"], "min_pass_rate": float("nan")})["code"] == "bad_flags"
    assert preflight({"scorers": ["exact_match"], "informational": ["tone"], "min_pass_rate": 0.9})["code"] == "unknown_scorer"
    assert preflight({"scorers": ["tone"], "informational": ["tone"], "min_pass_rate": 0.9})["code"] == "no_scorer"
    assert preflight({"scorers": ["exact_match"], "min_pass_rate": 0.9}) is None


def test_missing_judgment_is_incomplete_coverage():
    rows = [
        {"item_id": "a", "panels": [{"scores": [{"name": "exact_match", "passed": True, "score": 1.0}]}]},
        {"item_id": "b", "panels": [{"scores": []}]},
    ]
    scorers = report_from_rows(rows, scorers=["exact_match"])
    assert scorers[0]["expected"] == 2
    assert scorers[0]["completed"] == 1
    assert scorers[0]["missing"] == 1
    outcome = decide({
        "rows": rows,
        "min_pass_rate": 0.9,
        "dataset_verified": True,
        "scorers": scorers,
    })
    assert outcome["exit_code"] == 3
    assert outcome["code"] == "scorer_coverage"


def test_relative_gate_fails_without_a_baseline_measurement():
    from traccia.eval.gate import apply_baseline, validate_baseline

    missing = validate_baseline({"missing": True}, {"baseline_experiment": "exp", "scorers": ["exact_match"], "max_drop": 0.05})
    assert missing["code"] == "baseline_missing"
    report = {
        "rows": [{"item_id": "a"}],
        "max_drop": 0.05,
        "dataset_verified": True,
        "scorers": [{
            "name": "exact_match",
            "pass_rate": 1,
            "definition_hash": "sha256:a",
            "baseline_definition_hash": "sha256:a",
            "expected": 1,
            "completed": 1,
            "missing": 0,
            "errored": 0,
        }],
    }
    outcome = decide(report)
    assert outcome["exit_code"] == 3
    assert outcome["code"] == "baseline_missing"
    unverified = decide({
        "rows": [{"item_id": "a"}],
        "min_pass_rate": 0.9,
        "dataset_verified": False,
        "scorers": [{"name": "exact_match", "pass_rate": 1, "expected": 1, "completed": 1, "missing": 0, "errored": 0}],
    })
    assert unverified["exit_code"] == 3
    assert unverified["quality_gate"] is False
    assert unverified["code"] == "dataset_unverified"
    cost = {
        "rows": [{"item_id": "a", "panels": [{}]}],
        "max_task_cost_increase": 0.2,
        "dataset_fingerprint": "same",
        "scorers": [],
    }
    apply_baseline(cost, {"dataset_fingerprint": "same", "task_cost_usd": 1, "task_cost_coverage": "complete", "scorers": []})
    assert decide(cost)["code"] == "cost_incomplete"


def test_rows_feed_the_pass_rate():
    rows = [
        {
            "item_id": "a",
            "panels": [{"scores": [{"name": "exact_match", "passed": False, "score": 0.0}]}],
        }
    ]
    outcome = decide({"rows": rows, "min_pass_rate": 0.9, "dataset_verified": True, "scorers": report_from_rows(rows)})
    assert outcome["exit_code"] == 1
    assert outcome["code"] == "threshold_missed"


def test_missing_judgment_is_not_complete_coverage():
    rows = [
        {"item_id": "a", "panels": [{"scores": [{"name": "exact_match", "passed": True, "score": 1.0}]}]},
        {"item_id": "b", "panels": [{"scores": []}]},
    ]
    scorers = report_from_rows(rows, scorers=["exact_match"])
    assert scorers[0]["expected"] == 2
    assert scorers[0]["completed"] == 1
    assert scorers[0]["missing"] == 1
    outcome = decide({"rows": rows, "min_pass_rate": 0.9, "scorers": scorers})
    assert outcome["exit_code"] == 3
    assert outcome["code"] == "scorer_coverage"


def test_dataset_change_is_a_quality_miss():
    from traccia.eval.gate import apply_baseline, validate_baseline

    assert validate_baseline({"missing": True}, {"baseline_experiment": "exp", "max_drop": 0.05, "scorers": ["exact_match"]})["code"] == "baseline_missing"
    report = {
        "rows": [{"item_id": "a", "panels": [{"latency_ms": 10, "task_cost": {"usd": 1, "coverage": "complete"}, "scores": []}]}],
        "max_task_cost_increase": 0.2,
        "max_drop": 0.05,
        "dataset_verified": True,
        "dataset_fingerprint": "sha256:new",
        "scorers": [{"name": "exact_match", "pass_rate": 1, "definition_hash": "sha256:a", "expected": 1, "completed": 1, "missing": 0, "errored": 0}],
    }
    apply_baseline(
        report,
        {
            "dataset_fingerprint": "sha256:old",
            "dataset_verified": True,
            "scorers": [{"name": "exact_match", "pass_rate": 1, "definition_hash": "sha256:a"}],
            "scorer_evidence": [{"name": "exact_match", "verified": True}],
            "task_cost_usd": 1,
            "task_cost_coverage": "complete",
            "judge_cost_usd": 0,
            "judge_cost_coverage": "complete",
            "judge_definition_hashes": [],
            "mean_latency_ms": 10,
            "latency_concurrency": 1,
        },
    )
    assert report["dataset_changed"] is True
    assert report["task_cost_increase"] == 0
    outcome = decide(report)
    assert outcome["exit_code"] == 1
    assert outcome["code"] == "dataset_changed"


def test_baseline_evidence_is_scoped_to_the_selected_panel():
    from traccia.eval.gate import apply_baseline, validate_baseline

    evidence = [
        {"name": "faithfulness", "panel_index": 0, "verified": False},
        {"name": "faithfulness", "panel_index": 1, "verified": True},
    ]
    scorers = [{"name": "faithfulness", "pass_rate": 1, "definition_hash": "sha256:h"}]
    options = {"baseline_experiment": "exp", "scorers": ["faithfulness"], "max_drop": 0.05}
    blocked = validate_baseline(
        {"panel_index": 0, "dataset_verified": True, "scorers": scorers, "scorer_evidence": evidence},
        options,
    )
    assert blocked["code"] == "scorer_unverified"
    assert validate_baseline(
        {"panel_index": 1, "dataset_verified": True, "scorers": scorers, "scorer_evidence": evidence},
        options,
    ) is None
    report = {
        "rows": [{"item_id": "a"}],
        "max_drop": 0.05,
        "dataset_verified": True,
        "dataset_fingerprint": "same",
        "scorers": [{
            "name": "faithfulness",
            "pass_rate": 1,
            "definition_hash": "sha256:h",
            "expected": 1,
            "completed": 1,
            "missing": 0,
            "errored": 0,
        }],
    }
    apply_baseline(
        report,
        {
            "panel_index": 0,
            "dataset_fingerprint": "same",
            "scorers": scorers,
            "scorer_evidence": evidence,
        },
    )
    assert report["scorers"][0]["verified"] is False
    assert decide(report)["code"] == "scorer_unverified"


def test_renamed_scorer_still_requires_evidence_for_its_id():
    from traccia.eval.gate import apply_baseline, validate_baseline

    options = {"baseline_experiment": "exp", "scorers": ["new-name"], "max_drop": 0.05}
    baseline = {
        "panel_index": 0,
        "dataset_verified": True,
        "scorers": [{"name": "new-name", "scorer_id": "sc-1", "pass_rate": 1, "definition_hash": "sha256:h"}],
        "scorer_evidence": [{"name": "old-name", "scorer_id": "sc-1", "panel_index": 0, "verified": False}],
    }
    assert validate_baseline(baseline, options)["code"] == "scorer_unverified"
    missing = {**baseline, "scorer_evidence": []}
    assert validate_baseline(missing, options)["code"] == "scorer_unverified"
    verified = {
        **baseline,
        "scorer_evidence": [{"name": "old-name", "scorer_id": "sc-1", "panel_index": 0, "verified": True}],
    }
    assert validate_baseline(verified, options) is None
    report = {
        "rows": [{"item_id": "a"}],
        "max_drop": 0.05,
        "dataset_verified": True,
        "dataset_fingerprint": "same",
        "scorers": [{
            "name": "new-name",
            "pass_rate": 1,
            "definition_hash": "sha256:h",
            "expected": 1,
            "completed": 1,
            "missing": 0,
            "errored": 0,
        }],
    }
    apply_baseline(report, {**baseline, "dataset_fingerprint": "same"})
    assert report["scorers"][0]["verified"] is False
    assert decide(report)["code"] == "scorer_unverified"


def test_concurrency_override_keeps_the_latency_limit():
    from traccia.eval.gate import apply_baseline

    report = {
        "rows": [{"item_id": "a", "panels": [{"latency_ms": 1000, "task_cost": {"usd": 0, "coverage": "complete"}, "scores": []}]}],
        "max_latency_increase": 0.2,
        "allow_concurrency_mismatch": True,
        "latency_concurrency": 8,
        "dataset_fingerprint": "same",
        "scorers": [],
    }
    apply_baseline(
        report,
        {
            "dataset_fingerprint": "same",
            "mean_latency_ms": 10,
            "latency_concurrency": 1,
            "scorers": [],
        },
    )
    assert report["latency_concurrency_mismatch"] is True
    assert report["latency_increase"] == 99
    outcome = decide(report)
    assert outcome["exit_code"] == 1
    assert outcome["code"] == "latency_increase"


def test_partial_judge_cost_and_duplicate_hashes():
    from traccia.eval.gate import apply_baseline

    partial = {
        "rows": [{
            "item_id": "a",
            "panels": [{
                "scores": [
                    {"type": "llm_judge", "definition_hash": "h", "invocation": {"id": "one", "cost_usd": 1}},
                    {"type": "llm_judge", "definition_hash": "h"},
                ]
            }],
        }],
        "max_judge_cost_increase": 0.2,
        "dataset_fingerprint": "same",
        "scorers": [],
    }
    baseline = {
        "dataset_fingerprint": "same",
        "judge_cost_usd": 2,
        "judge_cost_coverage": "complete",
        "judge_definition_hashes": ["h"],
        "scorers": [],
    }
    apply_baseline(partial, baseline)
    assert decide(partial)["code"] == "cost_incomplete"

    same = {
        "rows": [{
            "item_id": "a",
            "panels": [{
                "scores": [
                    {"type": "llm_judge", "definition_hash": "h", "invocation": {"id": "one", "cost_usd": 1}},
                    {"type": "llm_judge", "definition_hash": "h", "invocation": {"id": "one", "cost_usd": 1}},
                ]
            }],
        }],
        "max_judge_cost_increase": 0.2,
        "dataset_fingerprint": "same",
        "scorers": [],
    }
    apply_baseline(same, {**baseline, "judge_cost_usd": 1})
    assert same.get("judge_workload_mismatch") is not True
    assert same["judge_cost_increase"] == 0


def test_renamed_scorer_matches_its_baseline_by_id():
    from traccia.eval.gate import apply_baseline, decide, report_from_rows, validate_baseline

    baseline = {
        "panel_index": 0,
        "dataset_verified": True,
        "dataset_fingerprint": "same",
        "scorers": [{
            "name": "ci-rename",
            "scorer_id": "sc-1",
            "pass_rate": 1,
            "definition_hash": "sha256:h",
        }],
        "scorer_evidence": [{
            "name": "ci-rename",
            "scorer_id": "sc-1",
            "panel_index": 0,
            "verified": True,
            "definition_hash": "sha256:h",
        }],
    }
    options = {
        "baseline_experiment": "exp",
        "scorers": ["ci-rename-v2"],
        "max_drop": 0.05,
        "scorer_ids": {"ci-rename-v2": "sc-1"},
    }
    assert validate_baseline(baseline, options) is None
    assert validate_baseline(baseline, {**options, "scorer_ids": {}})["code"] == "baseline_missing"
    rows = [{
        "item_id": "a",
        "panels": [{
            "scores": [{
                "scorer_name": "ci-rename-v2",
                "scorer_id": "sc-1",
                "passed": True,
                "score": 1,
                "status": "judged",
                "definition_hash": "sha256:h",
            }],
        }],
    }]
    report = {
        "rows": rows,
        "max_drop": 0.05,
        "dataset_verified": True,
        "dataset_fingerprint": "same",
        "scorers": report_from_rows(rows, scorers=["ci-rename-v2"]),
    }
    assert report["scorers"][0]["scorer_id"] == "sc-1"
    apply_baseline(report, baseline)
    outcome = decide(report)
    assert report["scorers"][0]["baseline_pass_rate"] == 1
    assert outcome["exit_code"] == 0
    assert outcome["quality_gate"] is True


def test_unhashed_local_scorer_is_not_hidden_by_baseline_preflight():
    from traccia.eval.gate import decide, validate_baseline

    baseline = {
        "panel_index": 0,
        "dataset_verified": True,
        "scorers": [{"name": "exact_match", "pass_rate": 1, "definition_hash": "sha256:exact"}],
        "scorer_evidence": [{"name": "exact_match", "panel_index": 0, "verified": True}],
    }
    options = {
        "baseline_experiment": "exp",
        "scorers": ["local:accuracy"],
        "max_drop": 0.05,
        "unhashed_scorers": ["accuracy"],
    }
    assert validate_baseline(baseline, options) is None
    hashed = {**options, "scorers": ["accuracy"], "unhashed_scorers": []}
    assert validate_baseline(baseline, hashed)["code"] == "baseline_missing"
    outcome = decide({
        "rows": [{"item_id": "a"}],
        "max_drop": 0.05,
        "scorers": [{
            "name": "accuracy",
            "definition_identity": "source_fallback",
            "pass_rate": 1,
            "expected": 1,
            "completed": 1,
            "missing": 0,
            "errored": 0,
            "verified": False,
        }],
    })
    assert outcome["exit_code"] == 3
    assert outcome["code"] == "scorer_definition_unhashed"
    assert outcome["quality_gate"] is False


def test_uuid_selected_scorer_uses_the_result_name():
    scorer_id = "11111111-1111-1111-1111-111111111111"
    rows = [{
        "item_id": "a",
        "panels": [{
            "scores": [{
                "scorer_name": "accuracy",
                "scorer_id": scorer_id,
                "status": "judged",
                "score": 1,
                "passed": True,
            }],
        }],
    }]
    scorers = report_from_rows(rows, informational=[scorer_id], scorers=[scorer_id])
    assert [row["name"] for row in scorers] == ["accuracy"]
    assert scorers[0]["missing"] == 0
    assert scorers[0]["pass_rate"] == 1
    assert scorers[0]["informational"] is True


def test_jev_questions_are_gated_on_their_own():
    questions = [{"key": f"q{i}", "status": "judged", "passed": True, "score": 1} for i in range(9)]
    questions.append({"key": "q9", "status": "judged", "passed": False, "score": 0})
    rows = [{
        "item_id": "a",
        "panels": [{
            "scores": [{
                "scorer_name": "rubric",
                "role": "summary",
                "questions": questions,
            }],
        }],
    }]
    scorers = report_from_rows(rows, scorers=["rubric"])
    assert len(scorers) == 10
    failed = next(row for row in scorers if row["name"] == "rubric:q9")
    assert failed["pass_rate"] == 0
    assert next(row["pass_rate"] for row in scorers if row["name"] == "rubric:q0") == 1
    outcome = decide({
        "rows": rows,
        "min_pass_rate": 0.9,
        "dataset_verified": True,
        "scorers": scorers,
    })
    assert outcome["exit_code"] == 1
    assert outcome["code"] == "threshold_missed"


def test_unwritable_summary_is_a_boolean_failure():
    from traccia.eval.gate import write_summary

    assert write_summary("/dev/null/summary.json", {"label": "Failed"}) is False
