"""evaluate() runner — Braintrust/Langfuse-style offline experiments."""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from traccia.eval import client as eval_client
from traccia.eval.builtins import BUILTIN_SCORERS, run_builtin_scorer
from traccia.eval.errors import EvaluateError

ScorerSpec = Union[str, Callable[..., Any]]
DataSpec = Union[str, Sequence[Dict[str, Any]]]

logger = logging.getLogger("traccia.eval")


def _ensure_eval_tracing(
    *,
    api_key: Optional[str] = None,
    prompt_api_base: Optional[str] = None,
) -> bool:
    """Ensure OTLP export is configured so Open Trace links resolve.

    Without init()/start_tracing(), get_tracer() still allocates local span/trace
    ids that never reach Observe — phantom Open Trace URLs.
    """
    try:
        from traccia import auto as traccia_auto
        from traccia import init
    except Exception:  # noqa: BLE001
        return False

    if getattr(traccia_auto, "_started", False):
        return True

    agent_id = (
        os.environ.get("TRACCIA_AGENT_ID")
        or os.environ.get("AGENT_ID")
        or "sdk-evaluate"
    )
    resolved_key = api_key or os.environ.get("TRACCIA_API_KEY")
    try:
        init(
            api_key=resolved_key,
            prompt_api_base=prompt_api_base or os.environ.get("TRACCIA_PROMPT_API_BASE"),
            auto_start_trace=False,
            enable_patching=False,
            agent_id=agent_id,
            agent_name=os.environ.get("TRACCIA_AGENT_NAME") or "SDK Evaluate",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("evaluate(): could not initialize tracing for Open Trace: %s", exc)
        return False

    return bool(getattr(traccia_auto, "_started", False))


@dataclass
class EvaluateResult:
    name: str
    rows: List[Dict[str, Any]]
    aggregates: Dict[str, Any]
    experiment_id: Optional[str] = None
    dataset_id: Optional[str] = None
    url: Optional[str] = None
    persist_error: Optional[str] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    run_context: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        agg = self.aggregates or {}
        lines = [
            f"Experiment: {self.name}",
            f"Items: {agg.get('item_count', len(self.rows))}",
            f"Pass rate: {agg.get('pass_rate')}",
            f"Scored: {agg.get('scored_count', 0)}",
            f"Errors: {len(self.errors)}",
        ]
        if self.url:
            lines.append(f"URL: {self.url}")
        elif self.persist_error:
            lines.append(f"Persist error: {self.persist_error}")
        elif not self.experiment_id:
            lines.append("Persisted: no (local-only)")
        return "\n".join(lines)


def _normalize_inline_rows(data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise EvaluateError(f"Inline data row {i} must be a dict")
        inp = raw.get("input", raw)
        if not isinstance(inp, dict):
            raise EvaluateError(f"Inline data row {i}: input must be an object")
        expected = raw.get("expected", raw.get("expected_output"))
        meta = raw.get("metadata")
        rows.append(
            {
                "id": str(raw.get("id") or uuid.uuid4()),
                "input": inp,
                "expected_output": expected,
                "metadata": meta if isinstance(meta, dict) else None,
            }
        )
    return rows


def _call_task(task: Callable[..., Any], row: Dict[str, Any]) -> Any:
    sig = inspect.signature(task)
    params = list(sig.parameters.values())
    if not params:
        result = task()
    else:
        name0 = params[0].name
        if name0 in ("row", "item", "example", "case"):
            result = task(row)
        elif name0 == "input":
            result = task(row["input"])
        else:
            result = task(row["input"])
    if inspect.isawaitable(result):
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Called from sync worker thread — safe to make a new loop
            return asyncio.run(_await(result))  # type: ignore[arg-type]
        return asyncio.run(_await(result))  # type: ignore[arg-type]
    return result


async def _await(value: Any) -> Any:
    return await value


def _panel_label(*, prompt: Optional[str]) -> str:
    """Playground cells are prompt versions; SDK cells are a Task unless prompt= was set."""
    if prompt and str(prompt).strip():
        return str(prompt).strip()
    return "Task"


def bind_scorer_name(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Keep the exported mapping key when the function itself has another name."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    wrapper.__name__ = name
    return wrapper


def _spec_name(spec: ScorerSpec) -> str:
    if isinstance(spec, str):
        return spec.strip() or "scorer"
    return str(getattr(spec, "__name__", None) or "scorer")


def _score_span_name(spec: ScorerSpec) -> str:
    raw = _spec_name(spec)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)[:80]
    return f"scorer.{safe or 'unnamed'}"


def _annotate_score_span(span: Any, scored: Dict[str, Any]) -> None:
    """Stamp scorer outcome; llm_judge becomes an LLM child so the waterfall fills."""
    if span is None:
        return
    try:
        name = str(scored.get("scorer_name") or scored.get("name") or "")
        if name:
            span.set_attribute("traccia.eval.scorer", name)
        if scored.get("passed") is not None:
            span.set_attribute("traccia.eval.passed", bool(scored["passed"]))
        reason = scored.get("reason")
        if reason:
            span.set_attribute("traccia.eval.reason", str(reason)[:500])
        stype = str(scored.get("type") or "")
        if stype == "llm_judge" or scored.get("model"):
            span.set_attribute("span.type", "llm")
            model = scored.get("model")
            if model:
                span.set_attribute("llm.model", str(model))
                span.set_attribute("gen_ai.request.model", str(model))
            if scored.get("cost_usd") is not None:
                span.set_attribute("llm.cost.usd", float(scored["cost_usd"]))
            usage = scored.get("usage") if isinstance(scored.get("usage"), dict) else {}
            prompt_t = usage.get("prompt_tokens") or usage.get("input_tokens")
            completion_t = usage.get("completion_tokens") or usage.get("output_tokens")
            if prompt_t is not None:
                span.set_attribute("llm.usage.prompt_tokens", int(prompt_t))
                span.set_attribute("gen_ai.usage.input_tokens", int(prompt_t))
            if completion_t is not None:
                span.set_attribute("llm.usage.completion_tokens", int(completion_t))
                span.set_attribute("gen_ai.usage.output_tokens", int(completion_t))
            if prompt_t is not None or completion_t is not None:
                total = int(prompt_t or 0) + int(completion_t or 0)
                span.set_attribute("llm.usage.total_tokens", total)
                span.set_attribute("gen_ai.usage.total_tokens", total)
        else:
            span.set_attribute("span.type", "eval")
    except Exception:  # noqa: BLE001
        pass


_task_child_costs: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "traccia_eval_task_child_costs", default=None
)
_cost_processor_installed = False


def _read_cost(attrs: Any) -> Optional[float]:
    if not isinstance(attrs, Mapping):
        return None
    for key in ("llm.cost.usd", "platform_cost_usd", "traccia.cost.usd"):
        raw = attrs.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _provider_span(span: Any) -> bool:
    attrs = getattr(span, "attributes", None)
    if isinstance(attrs, Mapping) and (attrs.get("llm.model") or attrs.get("gen_ai.request.model")):
        return True
    name = str(getattr(span, "name", "") or "")
    return name not in {"evaluate.item"} and not name.startswith("scorer.") and "llm" in name


def _child_cost_processor_cls():
    from opentelemetry.sdk.trace import SpanProcessor as OTelSpanProcessor

    class _TaskChildCostProcessor(OTelSpanProcessor):
        """Record provider span costs that end while a task is running."""

        def on_start(self, span: Any, parent_context: Any = None) -> None:
            return None

        def on_end(self, span: Any) -> None:
            bucket = _task_child_costs.get()
            if bucket is None:
                return
            name = str(getattr(span, "name", "") or "")
            if name == "evaluate.item" or name.startswith("scorer."):
                return
            if not _provider_span(span):
                return
            bucket.append({"cost": _read_cost(getattr(span, "attributes", None))})

        def shutdown(self) -> None:
            return None

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    return _TaskChildCostProcessor


def _ensure_child_cost_processor() -> None:
    """Attach the collector to Traccia's provider, which is separate from the global one."""
    global _cost_processor_installed
    if _cost_processor_installed:
        return
    _cost_processor_installed = True
    try:
        import traccia

        traccia.get_tracer_provider().add_span_processor(_child_cost_processor_cls()())
    except Exception:  # noqa: BLE001
        _cost_processor_installed = False


def _cost_from_span(span: Any, child_costs: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[float]:
    own = _read_cost(getattr(span, "attributes", None))
    if own is not None:
        return own
    nested: List[Dict[str, Any]] = list(child_costs or [])
    for child in getattr(span, "children", None) or []:
        if _provider_span(child):
            nested.append({"cost": _cost_from_span(child)})
    if not nested:
        return None
    if any(item.get("cost") is None for item in nested):
        return None
    return sum(float(item["cost"]) for item in nested)


def _normalize_score(raw: Any, *, default_name: str) -> Dict[str, Any]:
    builtin = default_name if default_name in BUILTIN_SCORERS else None
    if isinstance(raw, dict):
        out = dict(raw)
        out["name"] = default_name
        out["scorer_name"] = default_name
        if "type" not in out and default_name in BUILTIN_SCORERS:
            out["type"] = default_name
        elif "type" not in out and builtin:
            out["type"] = builtin
        if out.get("score") is None and isinstance(out.get("passed"), bool):
            out["score"] = 1.0 if out["passed"] else 0.0
        if "passed" not in out and "score" in out:
            try:
                out["passed"] = float(out["score"]) >= 0.5
            except (TypeError, ValueError):
                out["passed"] = False
        out.setdefault("status", "judged")
        out.setdefault("passed", False)
        return out
    if isinstance(raw, (int, float)):
        score = float(raw)
        if score != score or score in (float("inf"), float("-inf")):
            return {
                "name": default_name,
                "scorer_name": default_name,
                "status": "error",
                "passed": None,
                "score": None,
                "reason": "non_finite_score",
            }
        out = {
            "name": default_name,
            "scorer_name": default_name,
            "status": "judged",
            "score": score,
            "passed": score >= 0.5,
        }
        if builtin:
            out["type"] = builtin
        return out
    if isinstance(raw, bool):
        out = {
            "name": default_name,
            "scorer_name": default_name,
            "score": 1.0 if raw else 0.0,
            "passed": raw,
        }
        if builtin:
            out["type"] = builtin
        out.setdefault("status", "judged")
        return out
    out = {
        "name": default_name,
        "scorer_name": default_name,
        "status": "error",
        "passed": None,
        "score": None,
        "reason": "invalid_scorer_return",
    }
    if builtin:
        out["type"] = builtin
    return out


def _snapshot_needs_refresh(token: Optional[str], *, margin: float = 300) -> bool:
    """True when a signed snapshot will expire before a long run finishes scoring."""
    if not token or "." not in str(token):
        return False
    try:
        payload = json.loads(str(token).rsplit(".", 1)[0])
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    exp = payload.get("exp") if isinstance(payload, dict) else None
    return isinstance(exp, (int, float)) and float(exp) <= time.time() + margin


def _run_scorer(
    spec: ScorerSpec,
    *,
    row: Dict[str, Any],
    output: Any,
    provider_keys: Optional[Dict[str, str]],
    scorer_cache: Dict[str, Dict[str, Any]],
    cred: Dict[str, Any],
) -> Dict[str, Any]:
    if callable(spec) and not isinstance(spec, str):
        name = getattr(spec, "__name__", "scorer")
        try:
            raw = spec(
                input=row.get("input"),
                output=output,
                expected=row.get("expected_output"),
                metadata=row.get("metadata"),
            )
            if inspect.isawaitable(raw):
                import asyncio

                raw = asyncio.get_event_loop().run_until_complete(raw)
        except Exception as exc:  # noqa: BLE001
            return {
                "name": name,
                "scorer_name": name,
                "status": "error",
                "passed": None,
                "score": None,
                "reason": f"scorer_error: {exc}",
                "definition_identity": "source_fallback",
            }
        scored = _normalize_score(raw, default_name=name)
        if not scored.get("definition_hash") and not scored.get("definition_identity"):
            scored["definition_identity"] = "source_fallback"
        return scored

    name = str(spec).strip()
    if name in BUILTIN_SCORERS:
        return run_builtin_scorer(
            name, output=output, expected=row.get("expected_output")
        )

    # Platform scorer by name/id
    if name not in scorer_cache:
        scorer_cache[name] = eval_client.fetch_scorer(name, **cred)
    scorer = scorer_cache[name]
    if _snapshot_needs_refresh(scorer.get("scorer_snapshot_token")):
        scorer = eval_client.fetch_scorer(name, **cred)
        scorer_cache[name] = scorer
    stype = str(scorer.get("type") or "")
    if stype in BUILTIN_SCORERS:
        scored = run_builtin_scorer(
            stype,
            output=output,
            expected=row.get("expected_output"),
            config=scorer.get("config") or {},
        )
        platform_name = str(scorer.get("name") or name)
        scored["scorer_id"] = str(scorer.get("id") or "")
        scored["scorer_name"] = platform_name
        scored["name"] = platform_name
        scored["type"] = stype
        scored["config"] = scorer.get("config") or {}
        if scorer.get("definition_hash"):
            scored["definition_hash"] = scorer.get("definition_hash")
        if scorer.get("implementation_version"):
            scored["implementation_version"] = scorer.get("implementation_version")
        if scorer.get("scorer_snapshot_token"):
            scored["scorer_snapshot_token"] = scorer.get("scorer_snapshot_token")
            scored.pop("definition_identity", None)
        return scored

    remote = eval_client.score_remote(
        scorer_id=str(scorer["id"]),
        output=output,
        expected_output=row.get("expected_output"),
        input_data=row.get("input"),
        provider_keys=provider_keys,
        scorer_snapshot_token=scorer.get("scorer_snapshot_token"),
        **cred,
    )
    status = remote.get("status")
    if status not in ("judged", "error"):
        status = "judged" if remote.get("score") is not None and isinstance(remote.get("passed"), bool) else "error"
    out = {
        "scorer_id": str(remote.get("scorer_id") or scorer.get("id") or ""),
        "scorer_name": str(remote.get("scorer_name") or scorer.get("name") or name),
        "type": remote.get("type") or stype,
        "config": remote.get("config") or scorer.get("config") or {},
        "name": str(remote.get("scorer_name") or name),
        "status": status,
        "passed": remote.get("passed") if status == "judged" else None,
        "reason": remote.get("reason"),
        "score": remote.get("score") if status == "judged" else None,
        "model": remote.get("model"),
        "latency_ms": remote.get("latency_ms"),
        "usage": remote.get("usage") if isinstance(remote.get("usage"), dict) else None,
        "definition_hash": remote.get("definition_hash") or scorer.get("definition_hash"),
        "implementation_version": remote.get("implementation_version") or scorer.get("implementation_version"),
        "scorer_snapshot_token": remote.get("scorer_snapshot_token") or scorer.get("scorer_snapshot_token"),
    }
    if isinstance(remote.get("invocation"), dict):
        out["invocation"] = remote["invocation"]
    elif remote.get("cost_usd") is not None:
        out["invocation"] = {"cost_usd": remote.get("cost_usd"), "model": remote.get("model")}
    if isinstance(remote.get("questions"), list):
        out["questions"] = remote["questions"]
    if "unsure" in remote:
        out["unsure"] = remote.get("unsure")
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list)):
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)
    return str(value)


def evaluate(
    name: str,
    *,
    data: DataSpec,
    task: Callable[..., Any],
    scorers: Optional[Sequence[ScorerSpec]] = None,
    prompt: Optional[str] = None,
    max_concurrency: int = 10,
    persist: bool = True,
    provider_keys: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    prompt_api_base: Optional[str] = None,
    progress: bool = True,
    on_item_complete: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    prompt_version: Optional[int] = None,
    prompt_label: Optional[str] = None,
    run_context: Optional[Dict[str, Any]] = None,
) -> EvaluateResult:
    """Run task + scorers over a dataset or inline rows.

    Args:
        name: Experiment name (also used in ephemeral dataset naming).
        data: Platform dataset name/id, or list of {input, expected, metadata}.
        task: Callable receiving input dict (or row/item) returning output.
        scorers: Builtin names, platform scorer names/ids, and/or callables.
        prompt: Optional prompt name to attach prompt_version_ids when persisting.
        max_concurrency: Parallel item workers (default 10).
        persist: When True, create an Experiment in Traccia (default).
        provider_keys: BYO keys for platform llm_judge scorers, and typesafe for Jev Decision.
        progress: Print N/M to stderr.
    """
    if not name or not str(name).strip():
        raise EvaluateError("name is required")
    if task is None or not callable(task):
        raise EvaluateError("task must be a callable")
    if max_concurrency < 1:
        raise EvaluateError("max_concurrency must be >= 1")

    scorers = list(scorers or [])
    cred = {"api_key": api_key, "prompt_api_base": prompt_api_base}
    tracing_ok = _ensure_eval_tracing(api_key=api_key, prompt_api_base=prompt_api_base)

    dataset_id: Optional[str] = None
    items: List[Dict[str, Any]]
    platform_scorer_ids: List[str] = []
    dataset_snapshot_token: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    prompt_attachment = None
    if prompt and prompt_version is None and not prompt_label:
        prompt_attachment = "deprecated_production_default"
        sys.stderr.write(
            "warning: prompt without prompt_version or prompt_label still attaches production. "
            "The next SDK release removes that default.\n"
        )

    if isinstance(data, str):
        ds = eval_client.fetch_dataset(data.strip(), **cred)
        dataset_id = str(ds["id"])
        raw_payload = eval_client.fetch_dataset_items_payload(dataset_id, **cred)
        raw_items = list(raw_payload.get("items") or [])
        dataset_snapshot_token = raw_payload.get("dataset_snapshot_token")
        dataset_fingerprint = raw_payload.get("dataset_fingerprint")
        items = [
            {
                "id": str(it["id"]),
                "input": it.get("input") or {},
                "expected_output": it.get("expected_output"),
                "metadata": it.get("metadata"),
            }
            for it in raw_items
        ]
    elif isinstance(data, Sequence):
        items = _normalize_inline_rows(data)
        if persist:
            short = uuid.uuid4().hex[:8]
            safe_name = str(name).strip().replace("/", "-")[:80]
            ephemeral = eval_client.create_ephemeral_dataset(
                name=f"sdk-eval/{safe_name}/{short}",
                description="Created by SDK evaluate() (ephemeral)",
                items=[
                    {
                        "input": it["input"],
                        "expected_output": it.get("expected_output"),
                        "metadata": it.get("metadata"),
                    }
                    for it in items
                ],
                **cred,
            )
            dataset_id = str(ephemeral["id"])
            # Prefer server item ids for experiment join
            created = ephemeral.get("items") or []
            if created and len(created) == len(items):
                for i, it in enumerate(created):
                    items[i]["id"] = str(it["id"])
    else:
        raise EvaluateError("data must be a dataset name/id string or a list of rows")

    if not items:
        raise EvaluateError("No items to evaluate (empty dataset or list)")

    experiment_id = str(uuid.uuid4()) if persist else None
    prompt_version_ids: List[str] = []
    if persist and prompt:
        prompt_version_ids = eval_client.resolve_prompt_version_ids(
            prompt,
            version=prompt_version,
            label=prompt_label,
            **cred,
        )

    # Pre-resolve platform scorers for scorer_ids list
    scorer_cache: Dict[str, Dict[str, Any]] = {}
    for spec in scorers:
        if isinstance(spec, str) and spec.strip() not in BUILTIN_SCORERS:
            try:
                s = eval_client.fetch_scorer(spec.strip(), **cred)
                scorer_cache[spec.strip()] = s
                platform_scorer_ids.append(str(s["id"]))
            except EvaluateError:
                # May be a builtin alias typo — leave to per-item path
                pass

    total = len(items)
    results_by_index: Dict[int, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []
    completed = 0

    def _process(idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
        from traccia import get_tracer, force_flush

        cell: Dict[str, Any] = {
            "panel_index": 0,
            "label": _panel_label(prompt=prompt),
            "source": "evaluate",
            "prompt_version_id": prompt_version_ids[0] if prompt_version_ids else None,
        }
        attrs = {
            "traccia.eval.source": "evaluate",
            "span.type": "eval",
        }
        if experiment_id:
            attrs["traccia.experiment.id"] = experiment_id
            attrs["traccia.experiment.name"] = name
        if dataset_id:
            attrs["traccia.dataset.id"] = dataset_id
        attrs["traccia.dataset.item_id"] = str(row["id"])

        tracer = get_tracer("traccia.eval")
        with tracer.start_as_current_span("evaluate.item", attributes=attrs) as span:
            cost_token = _task_child_costs.set([])
            try:
                t0 = time.perf_counter()
                _ensure_child_cost_processor()
                output = _call_task(task, row)
                child_costs = list(_task_child_costs.get() or [])
                _task_child_costs.set(None)
                cell["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                output = _jsonable(output)
                cell["output"] = output
                cell["error"] = None
                scores = []
                for spec in scorers:
                    spec_name = _spec_name(spec)
                    score_attrs = {
                        "span.type": "eval",
                        "traccia.eval.source": "evaluate",
                        "traccia.eval.scorer": spec_name,
                    }
                    with tracer.start_as_current_span(
                        _score_span_name(spec), attributes=score_attrs
                    ) as score_span:
                        try:
                            scored = _run_scorer(
                                spec,
                                row=row,
                                output=output,
                                provider_keys=provider_keys,
                                scorer_cache=scorer_cache,
                                cred=cred,
                            )
                        except Exception as exc:  # noqa: BLE001
                            scored = {
                                "name": spec_name,
                                "scorer_name": spec_name,
                                "status": "error",
                                "passed": None,
                                "score": None,
                                "reason": f"scorer_error: {exc}",
                            }
                            if spec_name in BUILTIN_SCORERS:
                                scored["type"] = spec_name
                        _annotate_score_span(score_span, scored)
                        scores.append(scored)
                cell["scores"] = scores
                span_cost = _cost_from_span(span, child_costs)
                if span_cost is None:
                    cell["task_cost"] = {"usd": None, "coverage": "unknown", "basis": "instrumented_spans"}
                else:
                    cell["task_cost"] = {"usd": span_cost, "coverage": "complete", "basis": "instrumented_spans"}
                cell["cost_usd"] = span_cost
                span_attrs = getattr(span, "attributes", None) or {}
                if isinstance(span_attrs, Mapping) and span_attrs.get("llm.model") and not cell.get("model"):
                    cell["model"] = str(span_attrs["llm.model"])
                if scorers:
                    cell["passed"] = all(bool(s.get("passed")) for s in scores) if scores else None
                else:
                    cell["passed"] = None
                try:
                    span.set_attribute("traccia.eval.passed", bool(cell.get("passed")))
                    if cell.get("latency_ms") is not None:
                        span.set_attribute("traccia.eval.latency_ms", float(cell["latency_ms"]))
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                cell["output"] = ""
                cell["error"] = str(exc)
                cell["scores"] = []
                cell["passed"] = False
                try:
                    span.record_exception(exc)
                except Exception:  # noqa: BLE001
                    pass
            finally:
                _task_child_costs.reset(cost_token)

            # Only attach when export is configured — otherwise Open Trace 404s.
            if tracing_ok:
                try:
                    ctx = span.context
                    if ctx and getattr(ctx, "trace_id", None):
                        cell["trace_id"] = str(ctx.trace_id)
                except Exception:  # noqa: BLE001
                    pass

        return {
            "item_id": str(row["id"]),
            "input": row.get("input"),
            "expected_output": row.get("expected_output"),
            "panels": [cell],
            "_error": cell.get("error"),
        }

    workers = min(max_concurrency, total)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, i, items[i]): i for i in range(total)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                row_out = fut.result()
            except Exception as exc:  # noqa: BLE001
                row_out = {
                    "item_id": str(items[idx]["id"]),
                    "input": items[idx].get("input"),
                    "expected_output": items[idx].get("expected_output"),
                    "panels": [
                        {
                            "panel_index": 0,
                            "label": _panel_label(prompt=prompt),
                            "source": "evaluate",
                            "output": "",
                            "error": str(exc),
                            "scores": [],
                            "passed": False,
                        }
                    ],
                    "_error": str(exc),
                }
            results_by_index[idx] = row_out
            if row_out.get("_error"):
                errors.append({"item_id": row_out["item_id"], "error": row_out["_error"]})
            completed += 1
            if progress:
                sys.stderr.write(f"\r{completed}/{total}")
                sys.stderr.flush()
            if on_item_complete:
                on_item_complete(completed, total, row_out)

    if progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    try:
        from traccia import force_flush

        force_flush()
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for i in range(total):
        r = results_by_index[i]
        r.pop("_error", None)
        rows.append(r)

    pass_count = 0
    scored_count = 0
    latency_vals: List[float] = []
    cost_vals: List[float] = []
    for r in rows:
        for p in r.get("panels") or []:
            if isinstance(p.get("latency_ms"), (int, float)):
                latency_vals.append(float(p["latency_ms"]))
            if isinstance(p.get("cost_usd"), (int, float)):
                cost_vals.append(float(p["cost_usd"]))
            for s in p.get("scores") or []:
                scored_count += 1
                if s.get("passed"):
                    pass_count += 1

    aggregates = {
        "item_count": len(rows),
        "panel_count": 1,
        "scorer_count": len(scorers),
        "pass_count": pass_count,
        "scored_count": scored_count,
        "pass_rate": (pass_count / scored_count) if scored_count else None,
        "error_count": len(errors),
        "source": "evaluate",
    }
    if latency_vals:
        aggregates["mean_latency_ms"] = sum(latency_vals) / len(latency_vals)
    if cost_vals:
        aggregates["total_cost_usd"] = sum(cost_vals)


    result = EvaluateResult(
        name=name,
        rows=rows,
        aggregates=aggregates,
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        errors=errors,
    )

    if persist:
        if not dataset_id:
            result.persist_error = "Missing dataset_id for persist"
            return result
        try:
            created = eval_client.create_experiment(
                dataset_id=dataset_id,
                name=name,
                experiment_id=experiment_id,
                prompt_version_ids=prompt_version_ids,
                scorer_ids=platform_scorer_ids,
                results={"rows": rows},
                aggregates=aggregates,
                run_context={
                    **(run_context or {}),
                    **({"dataset_fingerprint": dataset_fingerprint} if dataset_fingerprint else {}),
                    **({"prompt_attachment": prompt_attachment} if prompt_attachment else {}),
                    "latency_concurrency": max_concurrency,
                    "latency_semantics": "task_wall_per_item",
                },
                dataset_snapshot_token=dataset_snapshot_token,
                **cred,
            )
            result.url = created.get("url")
            result.experiment_id = str(created.get("id") or experiment_id)
            result.run_context = created.get("run_context") or result.run_context
        except Exception as exc:  # noqa: BLE001
            result.persist_error = str(exc)
            # Keep in-memory results

    return result
