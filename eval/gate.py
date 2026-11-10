"""Decide whether a CI eval run is allowed to merge.

Progress and diagnostics belong on stderr. The summary file is the machine
contract. A green run with no pass-rate bar is execution only.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = 1

EXIT_PASS = 0
EXIT_QUALITY = 1
EXIT_PREFLIGHT = 2
EXIT_EVIDENCE = 3
EXIT_PERSIST = 4
EXIT_SUMMARY = 5


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_summary(path: str, body: Dict[str, Any]) -> bool:
    """Write the summary atomically. False means the check must exit 5."""
    payload = {"schema_version": SCHEMA_VERSION, **body}
    tmp = None
    try:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".traccia-eval-", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
        tmp = None
        return True
    except Exception:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def preflight(options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a failure before any task runs, or None when the run may start."""
    scorers = list(options.get("scorers") or [])
    informational = list(options.get("informational") or [])
    min_rate = options.get("min_pass_rate")
    max_drop = options.get("max_drop")
    if min_rate is not None and (not _finite(min_rate) or not 0 <= float(min_rate) <= 1):
        return _fail(EXIT_PREFLIGHT, "bad_flags", "min pass rate must be between 0 and 1")
    if max_drop is not None and (not _finite(max_drop) or not 0 <= float(max_drop) <= 1):
        return _fail(EXIT_PREFLIGHT, "bad_flags", "max drop must be between 0 and 1")
    for key in ("max_latency_increase", "max_task_cost_increase", "max_judge_cost_increase"):
        value = options.get(key)
        if value is not None and (not _finite(value) or float(value) < 0):
            return _fail(EXIT_PREFLIGHT, "bad_flags", f"{key} must be a finite number >= 0")
    if options.get("baseline_prompt") and options.get("baseline_experiment"):
        return _fail(EXIT_PREFLIGHT, "baseline_conflict", "Pass one baseline, not both")
    needs_rate = min_rate is not None or max_drop is not None
    counted = {
        str(item).split(":", 1)[1] if str(item).startswith("local:") else str(item)
        for item in scorers
    } - set(informational)
    if needs_rate and not counted:
        return _fail(EXIT_PREFLIGHT, "no_scorer", "A pass rate bar needs a scorer that counts toward the pass rate")
    selected = {
        str(item).split(":", 1)[1] if str(item).startswith("local:") else str(item)
        for item in scorers
    }
    for name in informational:
        if name not in selected:
            return _fail(EXIT_PREFLIGHT, "unknown_scorer", f"{name} is not a selected scorer")
    relative = max_drop is not None or any(
        options.get(key) is not None
        for key in ("max_latency_increase", "max_task_cost_increase", "max_judge_cost_increase")
    )
    if relative and not options.get("baseline_prompt") and not options.get("baseline_experiment"):
        return _fail(EXIT_PREFLIGHT, "baseline_required", "This gate needs a baseline")
    if options.get("baseline_dataset_verified") is False:
        return _fail(EXIT_PREFLIGHT, "dataset_unverified", "The baseline dataset is not verified")
    if options.get("baseline_scorer_unverified"):
        return _fail(EXIT_PREFLIGHT, "scorer_unverified", "The baseline scorer is not verified")
    if options.get("baseline_missing"):
        return _fail(EXIT_PREFLIGHT, "baseline_missing", "Baseline was not found")
    if options.get("baseline_panel_required"):
        return _fail(EXIT_PREFLIGHT, "bad_flags", "Choose a baseline panel")
    return None


def decide(report: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a finished run into an exit code. Evidence outranks a low score."""
    failures: List[Dict[str, str]] = []
    if report.get("summary_failed"):
        return _result(EXIT_SUMMARY, "summary_failed", failures, report, quality_gate=False)
    if report.get("persist_error"):
        failures.append({"code": "persist_failed", "message": str(report.get("persist_error"))})
        return _result(EXIT_PERSIST, "persist_failed", failures, report, quality_gate=False)

    if report.get("dataset_drift"):
        failures.append({"code": "dataset_drift", "message": "Dataset changed while the run was in progress"})
    if not report.get("rows"):
        failures.append({"code": "empty_results", "message": "The run has no items"})
    if report.get("task_errors"):
        failures.append({"code": "task_error", "message": "A task failed"})

    quality = False
    for scorer in report.get("scorers") or []:
        name = scorer.get("name")
        if (
            report.get("max_drop") is not None
            and not scorer.get("informational")
            and (
                scorer.get("definition_identity") == "source_fallback"
                or not scorer.get("definition_hash")
                or not scorer.get("baseline_definition_hash")
            )
        ):
            failures.append({"code": "scorer_definition_unhashed", "message": f"{name} has no definition hash"})
        if scorer.get("errored") or scorer.get("missing"):
            code = "scorer_error" if scorer.get("errored") else "scorer_coverage"
            failures.append({"code": code, "message": f"{name} coverage is incomplete"})
        if scorer.get("non_finite"):
            failures.append({"code": "non_finite_score", "message": f"{name} returned a non-finite score"})
        if scorer.get("duplicate"):
            failures.append({"code": "duplicate_score", "message": f"{name} scored an item more than once"})

    if report.get("cost_gate") and report.get("cost_coverage") in ("partial", "unknown", None):
        failures.append({"code": "cost_incomplete", "message": "Task cost is not complete"})
    if report.get("latency_gate") and report.get("latency_incomplete"):
        failures.append({"code": "latency_incomplete", "message": "Every item needs a finite task latency"})
    if report.get("latency_concurrency_mismatch") and not report.get("allow_concurrency_mismatch"):
        failures.append({"code": "latency_concurrency_mismatch", "message": "Latency concurrency does not match the baseline"})
    if report.get("judge_workload_mismatch"):
        failures.append({"code": "judge_workload_mismatch", "message": "Judge scorers do not match the baseline"})
    if report.get("implementation_mismatch"):
        failures.append({"code": "scorer_implementation_mismatch", "message": "Scorer implementation changed during the run"})
    if report.get("scorer_definition_changed"):
        failures.append({"code": "scorer_definition_changed", "message": "Scorer definition does not match the baseline"})
    if (report.get("min_pass_rate") is not None or report.get("max_drop") is not None) and report.get("dataset_verified") is not True:
        failures.append({"code": "dataset_unverified", "message": "The dataset snapshot was not verified"})
    compared_now = [s for s in (report.get("scorers") or []) if not s.get("informational")]
    if report.get("max_drop") is not None:
        if not compared_now:
            failures.append({"code": "baseline_missing", "message": "No scorer has a baseline pass rate"})
        for scorer in compared_now:
            if scorer.get("verified") is False:
                failures.append({"code": "scorer_unverified", "message": f"{scorer.get('name')} is not verified"})
            base = scorer.get("baseline_pass_rate")
            if base is None or not _finite(base):
                failures.append({"code": "baseline_missing", "message": f"{scorer.get('name')} has no baseline pass rate"})
    if report.get("max_task_cost_increase") is not None and report.get("task_cost_increase") is None:
        failures.append({"code": "cost_incomplete", "message": "Task cost is not complete"})
    if (
        report.get("max_judge_cost_increase") is not None
        and not report.get("judge_workload_mismatch")
        and report.get("judge_cost_increase") is None
    ):
        failures.append({"code": "cost_incomplete", "message": "Judge cost is not complete"})
    if report.get("max_latency_increase") is not None and report.get("latency_increase") is None:
        failures.append({"code": "latency_incomplete", "message": "Every item needs a finite task latency"})

    if failures:
        return _result(EXIT_EVIDENCE, failures[0]["code"], failures, report, quality_gate=False)

    if report.get("dataset_changed") and not report.get("allow_dataset_change"):
        failures.append({"code": "dataset_changed", "message": "Dataset fingerprint does not match the baseline"})
        return _result(EXIT_QUALITY, "dataset_changed", failures, report, quality_gate=True)

    min_rate = report.get("min_pass_rate")
    max_drop = report.get("max_drop")
    compared = [s for s in (report.get("scorers") or []) if not s.get("informational")]
    if (min_rate is not None or max_drop is not None) and compared:
        quality = True
        for scorer in compared:
            rate = scorer.get("pass_rate")
            if min_rate is not None and (rate is None or float(rate) < float(min_rate)):
                failures.append({"code": "threshold_missed", "message": f"{scorer.get('name')} is below the pass rate"})
            if max_drop is not None:
                base = scorer.get("baseline_pass_rate")
                if base is not None and rate is not None and float(base) - float(rate) > float(max_drop) + 1e-12:
                    failures.append({"code": "regression", "message": f"{scorer.get('name')} dropped more than allowed"})

    if report.get("task_cost_increase") is not None and report.get("max_task_cost_increase") is not None:
        if float(report["task_cost_increase"]) > float(report["max_task_cost_increase"]) + 1e-12:
            failures.append({"code": "cost_increase", "message": "Task cost increased more than allowed"})
    if report.get("judge_cost_increase") is not None and report.get("max_judge_cost_increase") is not None:
        if float(report["judge_cost_increase"]) > float(report["max_judge_cost_increase"]) + 1e-12:
            failures.append({"code": "cost_increase", "message": "Judge cost increased more than allowed"})
    if report.get("latency_increase") is not None and report.get("max_latency_increase") is not None:
        if float(report["latency_increase"]) > float(report["max_latency_increase"]) + 1e-12:
            failures.append({"code": "latency_increase", "message": "Latency increased more than allowed"})

    if failures:
        return _result(EXIT_QUALITY, failures[0]["code"], failures, report, quality_gate=quality)
    return _result(EXIT_PASS, None, [], report, quality_gate=quality)


def _fail(code: int, name: str, message: str) -> Dict[str, Any]:
    return {
        "exit_code": code,
        "code": name,
        "failures": [{"code": name, "message": message}],
        "quality_gate": False,
    }


def _result(
    exit_code: int,
    code: Optional[str],
    failures: Sequence[Dict[str, str]],
    report: Dict[str, Any],
    *,
    quality_gate: bool,
) -> Dict[str, Any]:
    label = "Quality Gate" if quality_gate and exit_code == EXIT_PASS else (
        "Execution Only" if exit_code == EXIT_PASS else "Failed"
    )
    body: Dict[str, Any] = {
        "exit_code": exit_code,
        "code": code,
        "failures": list(failures),
        "quality_gate": quality_gate if exit_code == EXIT_PASS else False,
        "label": label if exit_code == EXIT_PASS else "Failed",
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint": report.get("dataset_fingerprint"),
        "dataset_verified": report.get("dataset_verified"),
        "experiment_id": report.get("experiment_id"),
        "url": report.get("url"),
        "scorers": report.get("scorers") or [],
    }
    gates = {
        key: report.get(key)
        for key in (
            "min_pass_rate",
            "max_drop",
            "max_task_cost_increase",
            "max_judge_cost_increase",
            "max_latency_increase",
            "baseline_experiment_id",
        )
        if report.get(key) is not None
    }
    if gates:
        body["gates"] = gates
    return body


def _selected_names(scorers: Sequence[Any]) -> List[str]:
    names = []
    for item in scorers or []:
        text = str(item)
        names.append(text.split(":", 1)[1] if text.startswith("local:") else text)
    return names


def _question_definition(score: Dict[str, Any]) -> List[str]:
    config = score.get("config") if isinstance(score.get("config"), dict) else {}
    raw = config.get("questions") if isinstance(config.get("questions"), list) else []
    keys = []
    for question in raw:
        if isinstance(question, dict) and question.get("key") is not None:
            keys.append(str(question.get("key")))
        elif isinstance(question, str):
            keys.append(question)
    return keys


def report_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    informational: Sequence[str] = (),
    item_ids: Optional[Sequence[str]] = None,
    scorers: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Coverage per scorer. Expected units come from the dataset, not from returned scores."""
    info = set(informational)
    ids = [str(item) for item in item_ids] if item_ids is not None else []
    if item_ids is None:
        for row in rows or []:
            if isinstance(row, dict) and row.get("item_id") is not None:
                ids.append(str(row.get("item_id")))
    unique_ids = list(dict.fromkeys(ids))
    duplicate_items = len(ids) != len(unique_ids)

    observations: Dict[str, List[Dict[str, Any]]] = {}
    id_to_name: Dict[str, str] = {}
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, dict) or row.get("item_id") is None:
            continue
        item_id = str(row.get("item_id"))
        for panel in row.get("panels") or []:
            if not isinstance(panel, dict):
                continue
            for score in panel.get("scores") or []:
                if not isinstance(score, dict):
                    continue
                questions = score.get("questions") if isinstance(score.get("questions"), list) else None
                if score.get("role") == "summary" and not questions:
                    continue
                name = str(score.get("scorer_name") or score.get("name") or "")
                scorer_id = str(score.get("scorer_id") or "")
                if not name and not scorer_id:
                    continue
                if scorer_id and name:
                    id_to_name[scorer_id] = name
                bucket = name or scorer_id
                units = questions if questions else [score]
                for unit in units:
                    if not isinstance(unit, dict) or unit.get("role") == "summary":
                        continue
                    key = str(unit.get("key")) if questions and unit.get("key") is not None else None
                    by_name.setdefault(bucket, []).append(
                        {"item_id": item_id, "key": key, "unit": unit, "score": score}
                    )

    selected = _selected_names(scorers or [])
    used = set()
    order: List[str] = []
    for token in selected:
        canonical = id_to_name.get(token, token)
        if canonical in used:
            continue
        used.add(canonical)
        order.append(canonical)
        observations[canonical] = list(by_name.get(canonical, []))
    for name, seen in by_name.items():
        if name not in used:
            order.append(name)
            observations[name] = seen
    info_names = {id_to_name.get(str(token), str(token)) for token in informational}

    out = []
    for name in order:
        seen = observations[name]
        config_sets = []
        returned_keys: List[str] = []
        hashes = []
        identities = []
        scorer_ids = []
        for obs in seen:
            keys = _question_definition(obs["score"])
            if keys:
                config_sets.append(tuple(keys))
            if obs["key"] is not None:
                returned_keys.append(obs["key"])
            if obs["score"].get("definition_hash"):
                hashes.append(obs["score"].get("definition_hash"))
            if obs["score"].get("definition_identity"):
                identities.append(obs["score"].get("definition_identity"))
            if obs["score"].get("scorer_id"):
                scorer_ids.append(str(obs["score"].get("scorer_id")))
        conflict = len(set(config_sets)) > 1 or len(set(hashes)) > 1
        if config_sets:
            question_keys = list(config_sets[0])
        else:
            question_keys = list(dict.fromkeys(returned_keys))
        counted: Dict[tuple, Dict[str, Any]] = {}
        duplicated = set()
        for obs in seen:
            pair = (obs["item_id"], obs["key"])
            if pair in counted:
                duplicated.add(pair)
                continue
            counted[pair] = obs["unit"]

        def append_rate(row_name: str, pairs: List[tuple]) -> None:
            completed = errored = passes = 0
            for pair in pairs:
                unit = counted.get(pair)
                if unit is None:
                    continue
                status = unit.get("status")
                value = unit.get("score")
                passed = unit.get("passed")
                finite = isinstance(value, (int, float)) and math.isfinite(float(value))
                if status is None and finite and isinstance(passed, bool):
                    status = "judged"
                if status == "judged" and finite and isinstance(passed, bool):
                    completed += 1
                    if passed:
                        passes += 1
                else:
                    errored += 1
            missing = len([pair for pair in pairs if pair not in counted])
            expected = len(pairs)
            duplicate = duplicate_items or any(pair in duplicated for pair in pairs)
            incomplete = errored or missing or duplicate or completed != expected
            parent = row_name.split(":", 1)[0]
            row = {
                "name": row_name,
                "expected": expected,
                "completed": completed,
                "missing": missing,
                "errored": errored,
                "duplicate": duplicate,
                "passes": passes,
                "pass_rate": None if incomplete or not completed else passes / completed,
                "informational": parent in info or parent in info_names or row_name in info or row_name in info_names,
                "definition_hash": None if conflict or not hashes else hashes[0],
                "definition_conflict": conflict,
                "definition_identity": identities[0] if identities else None,
            }
            if scorer_ids:
                row["scorer_id"] = scorer_ids[0]
            out.append(row)

        if question_keys:
            for key in question_keys:
                append_rate(f"{name}:{key}", [(item_id, key) for item_id in unique_ids])
        else:
            append_rate(name, [(item_id, None) for item_id in unique_ids])
    return out


def _relative_increase(baseline: float, candidate: float) -> float:
    if float(baseline) == 0:
        return 0.0 if float(candidate) == 0 else float("inf")
    return (float(candidate) - float(baseline)) / float(baseline)


def _judge_measurements(scores: Sequence[Any]) -> Dict[str, Any]:
    """Every model judgment needs an invocation with a finite price. Question rows share that invocation."""
    invocations: Dict[str, float] = {}
    hashes = []
    missing = False
    saw_model = False
    for score in scores or []:
        if not isinstance(score, dict):
            continue
        if score.get("key") is not None and not isinstance(score.get("invocation"), dict):
            continue
        is_model = str(score.get("type") or "") in {"llm_judge", "jev"} or isinstance(score.get("invocation"), dict)
        if score.get("role") == "summary" and not is_model:
            continue
        if not is_model:
            continue
        saw_model = True
        if score.get("definition_hash"):
            hashes.append(str(score.get("definition_hash")))
        invocation = score.get("invocation") if isinstance(score.get("invocation"), dict) else None
        if not invocation or not invocation.get("id") or not _finite(invocation.get("cost_usd")):
            missing = True
        else:
            invocations[str(invocation.get("id"))] = float(invocation["cost_usd"])
    hashes = sorted({item for item in hashes if item})
    if not saw_model:
        return {"judge_cost_usd": 0.0, "judge_cost_coverage": "complete", "judge_definition_hashes": [], "invocations": {}}
    if missing:
        return {"judge_cost_usd": None, "judge_cost_coverage": "partial", "judge_definition_hashes": hashes, "invocations": invocations}
    return {
        "judge_cost_usd": sum(invocations.values()),
        "judge_cost_coverage": "complete",
        "judge_definition_hashes": hashes,
        "invocations": invocations,
    }


def _candidate_measurements(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    task_usd = 0.0
    task_coverage = "complete"
    saw_task = False
    latencies: List[float] = []
    latency_incomplete = False
    judge_coverage = "complete"
    invocations: Dict[str, float] = {}
    judge_hashes: List[str] = []
    if not rows:
        task_coverage = "unknown"
        latency_incomplete = True
    for row in rows or []:
        panels = row.get("panels") if isinstance(row, dict) else None
        panel = panels[0] if isinstance(panels, list) and panels else None
        if not isinstance(panel, dict):
            task_coverage = "unknown"
            latency_incomplete = True
            continue
        task_cost = panel.get("task_cost") if isinstance(panel.get("task_cost"), dict) else None
        if task_cost and task_cost.get("coverage") == "complete" and _finite(task_cost.get("usd")):
            task_usd += float(task_cost["usd"])
            saw_task = True
        elif task_cost and task_cost.get("coverage") == "partial":
            task_coverage = "partial"
            saw_task = True
        elif task_cost is None and _finite(panel.get("cost_usd")):
            task_usd += float(panel["cost_usd"])
            saw_task = True
        elif task_coverage == "complete":
            task_coverage = "unknown"
        if _finite(panel.get("latency_ms")):
            latencies.append(float(panel["latency_ms"]))
        else:
            latency_incomplete = True
        judge = _judge_measurements(panel.get("scores") or [])
        judge_hashes.extend(judge["judge_definition_hashes"])
        if judge["judge_cost_coverage"] != "complete":
            judge_coverage = "partial"
        for key, cost in judge["invocations"].items():
            invocations[key] = cost
    if not saw_task:
        task_coverage = "unknown"
    return {
        "task_cost_usd": task_usd if task_coverage == "complete" else None,
        "task_cost_coverage": task_coverage,
        "judge_cost_usd": sum(invocations.values()) if judge_coverage == "complete" else None,
        "judge_cost_coverage": judge_coverage,
        "judge_definition_hashes": sorted(set(judge_hashes)),
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies and not latency_incomplete else None,
        "latency_incomplete": latency_incomplete or not latencies,
    }


def _remember_evidence(bucket: Dict[str, Dict[str, Any]], key: str, item: Dict[str, Any]) -> None:
    previous = bucket.get(key)
    if previous and (previous.get("verified") is not True or item.get("verified") is not True):
        item = {**item, "verified": False}
    bucket[key] = item


def _evidence_index(baseline: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Evidence for the selected panel, addressable by scorer id and by name."""
    panel_index = baseline.get("panel_index")
    by_id: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    for item in baseline.get("scorer_evidence") or []:
        if not isinstance(item, dict):
            continue
        if panel_index is not None and int(item.get("panel_index", 0)) != int(panel_index):
            continue
        scorer_id = str(item.get("scorer_id") or "")
        name = str(item.get("name") or "")
        if scorer_id:
            _remember_evidence(by_id, scorer_id, item)
        if name:
            _remember_evidence(by_name, name, item)
    return {"by_id": by_id, "by_name": by_name}


def _evidence_for_scorer(index: Dict[str, Dict[str, Dict[str, Any]]], scorer: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prefer the stable scorer id so a rename cannot hide an evidence record."""
    if not isinstance(scorer, dict):
        return None
    scorer_id = str(scorer.get("scorer_id") or "")
    if scorer_id:
        return index["by_id"].get(scorer_id)
    name = str(scorer.get("name") or "")
    return index["by_name"].get(name) if name else None


def _baseline_row(scorers: Sequence[Any], name: Any, scorer_id: Any = None) -> Optional[Dict[str, Any]]:
    """Prefer the current name. A rename still matches the stored scorer id.

    Question rows share a scorer id, so an id match is used only when it is unique
    and the requested name is the parent scorer.
    """
    by_name = None
    id_matches = []
    wanted = str(scorer_id or "")
    for item in scorers or []:
        if not isinstance(item, dict):
            continue
        if by_name is None and str(item.get("name") or "") == str(name or ""):
            by_name = item
        item_id = str(item.get("scorer_id") or "")
        if wanted and item_id and item_id == wanted:
            id_matches.append(item)
    if by_name is not None:
        return by_name
    if len(id_matches) == 1 and ":" not in str(name or ""):
        return id_matches[0]
    return None


def _baseline_rows_for_name(scorers: Sequence[Any], name: Any, scorer_id: Any = None) -> List[Dict[str, Any]]:
    """A Jev scorer is stored as one row per question. The parent name still selects them."""
    prefix = f"{name}:"
    children = [
        item
        for item in scorers or []
        if isinstance(item, dict) and str(item.get("name") or "").startswith(prefix)
    ]
    if children:
        return children
    row = _baseline_row(scorers, name, scorer_id)
    return [row] if row else []


def validate_baseline(baseline: Optional[Dict[str, Any]], options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fail before execution when a requested baseline cannot be compared."""
    if not options.get("baseline_prompt") and not options.get("baseline_experiment"):
        return None
    if not baseline or baseline.get("missing"):
        return _fail(EXIT_PREFLIGHT, "baseline_missing", "Baseline was not found")
    if baseline.get("panel_required"):
        return _fail(EXIT_PREFLIGHT, "bad_flags", "Choose a baseline panel")
    panel_index = baseline.get("panel_index")
    if panel_index is not None and int(panel_index) < 0:
        return _fail(EXIT_PREFLIGHT, "bad_flags", "Choose a baseline panel")
    if baseline.get("dataset_verified") is not True:
        return _fail(EXIT_PREFLIGHT, "dataset_unverified", "The baseline dataset is not verified")
    evidence = _evidence_index(baseline)
    informational = set(options.get("informational") or [])
    names = [name for name in _selected_names(options.get("scorers") or []) if name not in informational]
    ids = options.get("scorer_ids") or {}
    unhashed = {str(name) for name in (options.get("unhashed_scorers") or [])}
    if options.get("max_drop") is not None:
        for name in names:
            # A local function has no definition hash. Let the run finish so the
            # gate can report that, instead of a baseline problem that hides it.
            if name in unhashed:
                continue
            matches = _baseline_rows_for_name(baseline.get("scorers"), name, ids.get(name))
            if not matches or any(not _finite(item.get("pass_rate")) for item in matches):
                return _fail(EXIT_PREFLIGHT, "baseline_missing", f"{name} has no baseline pass rate")
            for base in matches:
                ev = _evidence_for_scorer(evidence, base)
                if ev is None or ev.get("verified") is not True:
                    return _fail(EXIT_PREFLIGHT, "scorer_unverified", f"{name} is not verified on the baseline")
    if options.get("max_task_cost_increase") is not None and (
        baseline.get("task_cost_coverage") != "complete" or not _finite(baseline.get("task_cost_usd"))
    ):
        return _fail(EXIT_PREFLIGHT, "baseline_missing", "Baseline task cost is incomplete")
    if options.get("max_judge_cost_increase") is not None and (
        baseline.get("judge_cost_coverage") != "complete" or not _finite(baseline.get("judge_cost_usd"))
    ):
        return _fail(EXIT_PREFLIGHT, "baseline_missing", "Baseline judge cost is incomplete")
    if options.get("max_latency_increase") is not None and not _finite(baseline.get("mean_latency_ms")):
        return _fail(EXIT_PREFLIGHT, "baseline_missing", "Baseline latency is incomplete")
    return None


def apply_baseline(report: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> None:
    """Attach comparison measurements. Absent numbers stay absent so the gate fails closed."""
    if not baseline or baseline.get("missing") or baseline.get("panel_required"):
        return
    base_fp = baseline.get("dataset_fingerprint")
    cand_fp = report.get("dataset_fingerprint")
    comparing = any(
        report.get(key) is not None
        for key in ("max_drop", "max_task_cost_increase", "max_judge_cost_increase", "max_latency_increase")
    )
    if comparing and (not base_fp or not cand_fp or base_fp != cand_fp):
        report["dataset_changed"] = True
    evidence = _evidence_index(baseline)
    for scorer in report.get("scorers") or []:
        base = _baseline_row(baseline.get("scorers"), scorer.get("name"), scorer.get("scorer_id"))
        ev = _evidence_for_scorer(evidence, base or scorer)
        if report.get("max_drop") is not None and not scorer.get("informational") and (ev is None or ev.get("verified") is not True):
            scorer["verified"] = False
        if not base:
            continue
        scorer["baseline_pass_rate"] = base.get("pass_rate")
        base_hash = base.get("definition_hash")
        cand_hash = scorer.get("definition_hash")
        if base_hash:
            scorer["baseline_definition_hash"] = base_hash
        if base_hash and cand_hash and base_hash != cand_hash:
            report["scorer_definition_changed"] = True
        if scorer.get("definition_conflict"):
            report["scorer_definition_changed"] = True
    measured = _candidate_measurements(report.get("rows") or [])
    if report.get("max_task_cost_increase") is not None:
        report["cost_gate"] = True
        report["cost_coverage"] = measured["task_cost_coverage"]
        if (
            measured["task_cost_coverage"] == "complete"
            and baseline.get("task_cost_coverage") == "complete"
            and _finite(measured.get("task_cost_usd"))
            and _finite(baseline.get("task_cost_usd"))
        ):
            report["task_cost_increase"] = _relative_increase(baseline["task_cost_usd"], measured["task_cost_usd"])
    if report.get("max_judge_cost_increase") is not None:
        base_hashes = sorted(baseline.get("judge_definition_hashes") or [])
        if base_hashes != measured["judge_definition_hashes"]:
            report["judge_workload_mismatch"] = True
        elif (
            measured["judge_cost_coverage"] == "complete"
            and baseline.get("judge_cost_coverage") == "complete"
            and _finite(measured.get("judge_cost_usd"))
            and _finite(baseline.get("judge_cost_usd"))
        ):
            report["judge_cost_increase"] = _relative_increase(baseline["judge_cost_usd"], measured["judge_cost_usd"])
    if report.get("max_latency_increase") is not None:
        report["latency_gate"] = True
        base_concurrency = baseline.get("latency_concurrency")
        cand_concurrency = report.get("latency_concurrency")
        if base_concurrency is None or cand_concurrency is None or int(base_concurrency) != int(cand_concurrency):
            report["latency_concurrency_mismatch"] = True
        if measured["latency_incomplete"] or not _finite(measured.get("mean_latency_ms")) or not _finite(baseline.get("mean_latency_ms")):
            report["latency_incomplete"] = True
        else:
            report["latency_increase"] = _relative_increase(baseline["mean_latency_ms"], measured["mean_latency_ms"])


def comment_body(summary: Dict[str, Any], comment_key: str) -> str:
    label = summary.get("label") or "Failed"
    lines = [f"<!-- traccia-eval:{comment_key} -->", f"**{label}**"]
    for failure in summary.get("failures") or []:
        lines.append(f"- `{failure.get('code')}` {failure.get('message')}")
    return "\n".join(lines) + "\n"
