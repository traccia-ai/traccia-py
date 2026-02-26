"""Span processor that enriches spans with agent metadata and cost."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from traccia.processors.cost_engine import compute_cost
from traccia.tracer.provider import SpanProcessor
from traccia import runtime_config


def _load_agent_catalog(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load agent metadata from a JSON file.
    Supports:
      { "agents": [ { "id": "...", "name": "...", ... } ] }
      or { "agent-id": { "name": "...", ... }, ... }
    """
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if isinstance(data, dict) and "agents" in data and isinstance(data["agents"], list):
        out = {}
        for agent in data["agents"]:
            if not isinstance(agent, dict):
                continue
            aid = agent.get("id")
            if aid:
                out[str(aid)] = agent
        return out
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


class AgentEnrichmentProcessor(SpanProcessor):
    """
    Enrich spans with agent metadata (id/name/env/owner/team/org) and compute llm.cost.usd if missing.

    Precedence (highest to lowest):
      1. Span attributes (agent.id, agent.name, env) — span-level overrides
      2. Run-scoped identity (runtime_config run_identity context) or init-time global
      3. Default identity passed at init (from init()/start_tracing() or TRACCIA_*)
      4. AGENT_DASHBOARD_* env vars (legacy)
      5. Single-agent catalog from AGENT_DASHBOARD_AGENT_CONFIG
    """

    def __init__(
        self,
        *,
        agent_config_path: Optional[str] = None,
        default_agent_id: Optional[str] = None,
        default_agent_name: Optional[str] = None,
        default_env: Optional[str] = None,
        legacy_default_env: str = "production",
    ) -> None:
        # Init-time identity (from init()/start_tracing() or TRACCIA_*) takes precedence over legacy env
        self.default_agent_id = default_agent_id or os.getenv("TRACCIA_AGENT_ID") or os.getenv("AGENT_DASHBOARD_AGENT_ID")
        self.default_agent_name = default_agent_name or os.getenv("TRACCIA_AGENT_NAME") or os.getenv("AGENT_DASHBOARD_AGENT_NAME")
        self.default_env = default_env or os.getenv("TRACCIA_ENV") or os.getenv("AGENT_DASHBOARD_ENV") or legacy_default_env
        self.default_name = self.default_agent_name  # alias for internal use
        self.default_type = os.getenv("AGENT_DASHBOARD_AGENT_TYPE")
        self.default_owner = os.getenv("AGENT_DASHBOARD_AGENT_OWNER")
        self.default_team = os.getenv("AGENT_DASHBOARD_AGENT_TEAM")
        self.default_org = os.getenv("AGENT_DASHBOARD_ORG_ID")
        self.default_sub_org = os.getenv("AGENT_DASHBOARD_SUB_ORG_ID")
        self.default_description = os.getenv("AGENT_DASHBOARD_AGENT_DESCRIPTION")
        cfg_path = (
            agent_config_path
            or os.getenv("AGENT_DASHBOARD_AGENT_CONFIG")
            or "agent_config.json"
        )
        self.catalog = _load_agent_catalog(cfg_path)
        # If only one agent is declared, remember it for convenient fallback.
        self.single_agent_id: Optional[str] = None
        if len(self.catalog) == 1:
            self.single_agent_id = next(iter(self.catalog.keys()))

    def on_end(self, span) -> None:
        attrs = span.attributes
        # Resolve agent id (span attrs > run-scoped/global runtime_config > init-time default)
        agent_id = (
            attrs.get("agent.id")
            or attrs.get("agent")
            or runtime_config.get_agent_id()
            or self.default_agent_id
        )
        # Try using tracer instrumentation scope as a fallback id
        if not agent_id and getattr(span, "tracer", None) is not None:
            agent_id = getattr(span.tracer, "instrumentation_scope", None)
        # If not found in attributes/env/scope, and only one agent exists in catalog, use it
        if not agent_id and self.single_agent_id:
            agent_id = self.single_agent_id
        # If still missing, skip enrichment
        if not agent_id:
            return

        # Look up static metadata
        meta = self.catalog.get(agent_id, {})
        # If the resolved id is not in catalog but we have a single agent defined, use that entry
        if not meta and self.single_agent_id:
            agent_id = self.single_agent_id
            meta = self.catalog.get(agent_id, {})

        def set_if_missing(key: str, value: Any) -> None:
            if value is None:
                return
            if key not in attrs or attrs.get(key) in (None, ""):
                attrs[key] = value

        attrs["agent.id"] = agent_id
        set_if_missing(
            "agent.name",
            meta.get("name") or runtime_config.get_agent_name() or self.default_name or agent_id,
        )
        set_if_missing("agent.type", meta.get("type") or self.default_type or "workflow")
        set_if_missing("agent.description", meta.get("description") or self.default_description or "")
        set_if_missing("owner", meta.get("owner") or self.default_owner)
        set_if_missing("team", meta.get("team") or self.default_team)
        set_if_missing("org.id", meta.get("org_id") or self.default_org)
        set_if_missing("sub_org.id", meta.get("sub_org_id") or self.default_sub_org)

        # Environment (run-scoped or init-time default)
        env_val = meta.get("env") or runtime_config.get_env() or self.default_env
        set_if_missing("env", env_val)
        set_if_missing("environment", env_val)

        # Consumers (store as list)
        consumers = meta.get("consuming_teams")
        if consumers and "agent.consuming_teams" not in attrs:
            attrs["agent.consuming_teams"] = consumers

        # Cost: fill llm.cost.usd if we have tokens + model
        if "llm.cost.usd" not in attrs:
            model = attrs.get("llm.model")
            prompt_tokens = attrs.get("llm.usage.prompt_tokens") or 0
            completion_tokens = attrs.get("llm.usage.completion_tokens") or 0
            if model and (prompt_tokens or completion_tokens):
                try:
                    cost = compute_cost(
                        model=model,
                        prompt_tokens=int(prompt_tokens or 0),
                        completion_tokens=int(completion_tokens or 0),
                    )
                    if cost is not None:
                        attrs["llm.cost.usd"] = cost
                except Exception:
                    pass

        # Span type inference if missing
        if "span.type" not in attrs and "type" not in attrs:
            span_type = None
            if attrs.get("llm.model"):
                span_type = "LLM"
            elif attrs.get("tool.name") or attrs.get("tool") or attrs.get("http.url"):
                span_type = "TOOL"
            if span_type:
                attrs["span.type"] = span_type

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout: Optional[float] = None) -> None:
        return
