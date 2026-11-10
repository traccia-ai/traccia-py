"""HTTP client for /api/v1/eval-runtime/*."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import requests

from traccia.eval.errors import EvaluateError
from traccia.prompts.client import resolve_credentials

logger = logging.getLogger("traccia.eval")


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _request(
    method: str,
    path: str,
    *,
    api_key: Optional[str] = None,
    prompt_api_base: Optional[str] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Any:
    key, base = resolve_credentials(api_key=api_key, prompt_api_base=prompt_api_base)
    url = f"{base}{path}"
    try:
        resp = requests.request(
            method, url, headers=_headers(key), json=json_body, timeout=timeout
        )
    except requests.RequestException as exc:
        raise EvaluateError(f"Eval runtime request failed: {exc}") from exc
    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        raise EvaluateError(f"Eval runtime {resp.status_code}: {detail}")
    if resp.status_code == 204 or not resp.content:
        return None
    return resp.json()


def fetch_dataset(name_or_id: str, **cred) -> Dict[str, Any]:
    return _request("GET", f"/api/v1/eval-runtime/datasets/{name_or_id}", **cred)


def fetch_dataset_items(dataset_id: str, **cred) -> List[Dict[str, Any]]:
    data = fetch_dataset_items_payload(dataset_id, **cred)
    return list((data or {}).get("items") or [])


def fetch_dataset_items_payload(dataset_id: str, **cred) -> Dict[str, Any]:
    data = _request("GET", f"/api/v1/eval-runtime/datasets/{dataset_id}/items", **cred)
    return data if isinstance(data, dict) else {"items": []}


def create_ephemeral_dataset(
    *,
    name: str,
    items: List[Dict[str, Any]],
    description: Optional[str] = None,
    **cred,
) -> Dict[str, Any]:
    return _request(
        "POST",
        "/api/v1/eval-runtime/datasets",
        json_body={"name": name, "description": description, "items": items},
        **cred,
    )


def fetch_scorer(name_or_id: str, **cred) -> Dict[str, Any]:
    return _request("GET", f"/api/v1/eval-runtime/scorers/{name_or_id}", **cred)


def score_remote(
    *,
    scorer_id: Optional[str] = None,
    scorer_name: Optional[str] = None,
    output: Any,
    expected_output: Any = None,
    input_data: Any = None,
    provider_keys: Optional[Dict[str, str]] = None,
    scorer_snapshot_token: Optional[str] = None,
    **cred,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "output": output,
        "expected_output": expected_output,
        "input": input_data,
        "provider_keys": provider_keys or {},
    }
    if scorer_id:
        body["scorer_id"] = scorer_id
    if scorer_name:
        body["scorer_name"] = scorer_name
    if scorer_snapshot_token:
        body["scorer_snapshot_token"] = scorer_snapshot_token
    return _request("POST", "/api/v1/eval-runtime/score", json_body=body, **cred)


def create_experiment(
    *,
    dataset_id: str,
    name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    prompt_version_ids: Optional[List[str]] = None,
    scorer_ids: Optional[List[str]] = None,
    results: Optional[Dict[str, Any]] = None,
    aggregates: Optional[Dict[str, Any]] = None,
    run_context: Optional[Dict[str, Any]] = None,
    dataset_snapshot_token: Optional[str] = None,
    **cred,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "name": name,
        "prompt_version_ids": prompt_version_ids or [],
        "scorer_ids": scorer_ids or [],
        "results": results or {},
        "aggregates": aggregates or {},
        "run_context": run_context or {},
    }
    if dataset_snapshot_token:
        body["dataset_snapshot_token"] = dataset_snapshot_token
    if experiment_id:
        body["id"] = experiment_id
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            return _request("POST", "/api/v1/eval-runtime/experiments", json_body=body, **cred)
        except EvaluateError as exc:
            last_error = exc
            if not str(exc).startswith("Eval runtime 5") or attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise last_error or EvaluateError("Eval runtime request failed")


def fetch_experiment(experiment_id: str, **cred) -> Dict[str, Any]:
    return _request("GET", f"/api/v1/eval-runtime/experiments/{experiment_id}", **cred)


def resolve_prompt_version_ids(
    prompt_name: str,
    *,
    version: Optional[int] = None,
    label: Optional[str] = None,
    **cred,
) -> List[str]:
    """Best-effort: load_prompt runtime and return [version_id]."""
    from traccia.prompts.client import fetch_prompt_runtime

    try:
        payload, _ = fetch_prompt_runtime(prompt_name, version=version, label=label, **cred)
    except Exception as exc:  # noqa: BLE001
        raise EvaluateError(f"Could not resolve prompt {prompt_name!r}: {exc}") from exc
    vid = payload.get("version_id") or payload.get("id")
    if not vid:
        return []
    # validate uuid-ish
    try:
        UUID(str(vid))
    except ValueError as exc:
        raise EvaluateError(f"Invalid prompt version_id from runtime: {vid}") from exc
    return [str(vid)]
