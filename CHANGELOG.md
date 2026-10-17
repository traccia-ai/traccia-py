# Changelog

All notable changes to the Traccia Python SDK (`pip install traccia`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Dates are the PyPI publish date (UTC).

## [Unreleased]

## [0.1.32] - 2026-10-08

### Added
 HEAD
- `ApprovalPending` is raised when Refund Guard or Purchase Guard holds a tool for a person. It is not `AgentBlockedError`. Catch it, do not run the tool, and do not retry the check. `pending_tool_result()` is a normal tool result for frameworks that retry raised errors

- Auto-instrumentation for `google.genai.models.Models.generate_content` in `instrumentation/gemini.py`, populating token usage, completion text, duration, and cost metadata
2568759 (feat: instrument google.genai models.generate_content)
- Local cost estimates price `llm.usage.cache_read_tokens` and `llm.usage.cache_write_tokens` at the model's cache read and cache write rates from the pricing snapshot. `llm.usage.prompt_tokens` is treated as uncached input
- `llm.pricing.cache_fallback` is set to `true` when a model has no cache rate and those tokens were priced at the input rate
- `compute_cost_detail()` in `traccia.processors.cost_engine` returns the cost and whether a cache fallback was used

### Fixed
- Model names without a provider prefix match provider-prefixed catalog keys (`grok-4.7` matches `xai/grok-4.7`) before prefix matching, so a shorter key such as `grok-4` no longer wins
- Spans with only cache tokens and no prompt or completion tokens now get a cost

## [0.1.31] - 2026-09-26

### Added
- `evaluate()` can grade with a saved Jev Decision scorer. Pass the key as `provider_keys={"typesafe": ...}`
- Per-question scores (`questions`) and the `unsure` flag from Jev Decision are kept on each experiment cell, so the run shows one result per question like a platform dataset run

## [0.1.30] - 2026-09-25

### Added
- Matched policy checks stamp `traccia.policy.activation` (`observe`, `warn`, or `block`) on the span
- `@govern` tool checks send the last read timestamp (`as_of` or `updated_at`) remembered from an earlier tool result, for Freshness Guard
- `@govern` tool checks send `context.customer_id` when tool arguments include `customer.id`, for Unique Customer Cap
- `@govern` LLM checks send retrieval evidence only from real `traccia.retrieval.*` span attributes (presence, and chunk count when that attribute was recorded). A missing retrieval span is not reported as zero chunks
- Prompt name, label, and version id stamped at `LoadedPrompt.compile` are attached to the following LLM policy check, including when the provider call opens its own child span

### Fixed
- `@observe` no longer treats `http.url` as a tool, so an HTTP span stays a normal span and does not run a tool policy check
- `requests` instrumentation skips tiktoken encoding downloads, so they no longer appear as HTTP spans

## [0.1.29] - 2026-09-07

### Added
- Per-call policy check under `@govern` (`governance/pep.py`): `POST /api/v1/policy/check` on instrumented LLM and tool spans (Spend Cap, Model Boundary, Loop Cap). Optional `check_policy()` for custom tools.
- `@govern` inherits `agent_id` from `init` / `TRACCIA_AGENT_ID`; pass `agent_id=` only to override
- `AgentBlockedError.decision_id`, `remaining_budget_usd`, and `reasons` on deny
- README example: instrumented OpenAI + tool observe under `@govern`
- Redaction allowlist for `traccia.policy.*` span attributes (same pattern as `traccia.prompt.*`)
- HTTP client skip for `@govern()` status/block calls, policy check/settle, and prompt-runtime fetches, matching the Node SDK

## [0.1.28] - 2026-08-18

### Added
- Gemini auto-instrumentation (`instrumentation/gemini.py`) for sync and async `client.interactions.create` in `google-genai`. Supports both SDK layouts (`Interactions` before 2.x, `GeminiNextGenInteractions` from 2.x)
- Gemini spans record `llm.vendor`, `llm.model`, `llm.interaction_id`, and `llm.response.status`, plus cost and duration metrics
- Token usage read directly from provider `total_tokens`, `total_cached_tokens`, and `total_tool_use_tokens` instead of being synthesized from input plus output
- Soft-skips when `google-genai` is not installed

### Fixed
- `start_tracing()` reads `api_key` and `endpoint` from a nested `[tracing]` section of the env config, not only from flat keys

## [0.1.27] - 2026-08-13

### Added
- `evaluate()` for offline experiments (platform dataset, local+persist, local-only)
- Eval-runtime client; local built-in scorers (`exact_match`, `contains`, `json_valid`); platform judge/code via server score
- Eval span attrs (`traccia.experiment.*`, `traccia.eval.source`, `traccia.dataset.*`) with redaction allowlist
- `result.summary()`, `result.url`, progress `N/M`, per-item error isolation, default persist and max_concurrency=10

### Fixed
- `start_tracing()` / `init()` now apply `TRACCIA_API_KEY` / `TRACCIA_ENDPOINT` from env (flat config); nested env load previously dropped Authorization on OTLP export
- `evaluate()` auto-inits tracing when needed and only attaches `trace_id` when export is configured (avoids phantom Open Trace links)
- Builtin scores now include `type` + `scorer_name`; panel label is `Task` (or `prompt=` name); cells record `latency_ms` and LLM/`judge` cost when present

## [0.1.26] - 2026-08-09

### Fixed
- `LoadedPrompt.compile` now stamps `traccia.prompt.id` on the active span (with name, version, version_id, label, is_fallback). Prompt Metrics joins prefer this id over name-only matching.
- OpenAI Agents SDK spans get the right `span.type` when it is missing: `agent.span.type` of `generation` is an LLM span, and `function` or `agent.tool.name` is a tool span

## [0.1.25] - 2026-07-16

### Added
- `load_prompt` / `prefetch_prompts` with TTL cache (~60s), stale-while-revalidate, and explicit fallback
- `{{var}}` compile helpers (`LoadedPrompt.compile`) with shared golden fixtures
- Auto span attributes `traccia.prompt.*` on compile (name, version, version_id, label, is_fallback)
- `init(prompt_cache_ttl_s=...)` / `TRACCIA_PROMPT_CACHE_TTL_S` for cache TTL
- `init(prompt_api_base=...)` / `TRACCIA_PROMPT_API_BASE` when prompt-runtime host differs from the traces host (advanced deployments only)
- Redaction allowlist so `traccia.prompt.*` identity keys are not wiped by `"prompt"` substring matching

### Fixed
- `init(auto_start_trace=True)` now attaches the auto-started root span to OTel context so child spans share one trace

## [0.1.24] - 2026-07-13

### Added
- Opt-in `compliance.frameworks: ["hipaa"]` stamps `hipaa.*` span attributes and turns redaction on by default unless explicitly disabled
- Best-effort MRN, NPI, and date-of-birth redaction heuristics (regex only, not medical NER)
- Console warnings when HIPAA mode is enabled. Ingestion is never blocked. Traccia does not sign a BAA

## [0.1.23] - 2026-07-10

### Added
- `@govern` decorator: observability plus runtime policy enforcement. Checks the platform agent-status API (`{base}/api/v1/agents/{agent_id}/status`, derived from the tracing endpoint) before each run
- `AgentBlockedError` exported from `traccia` and `traccia.governance`, raised on a hard block
- `fail_open`, `TRACCIA_AGENT_ID`, and optional `[governance]` overrides (`status_check_endpoint`, `post_block_endpoint`, `status_cache_ttl_seconds`)
- Policy status checks reuse connections, coalesce cache lookups, and send block telemetry without blocking the caller

## [0.1.22] - 2026-06-19

### Added
- `span_scope()`, `SpanScope`, `run_with_span()`, and `run_with_span_async()` for explicit span lifecycle control, so a span can stay open across async callbacks and streaming (Node SDK parity)
- `traccia.pricing_normalizer` with `normalize()` and `diff_models()`, shared by the snapshot build and the platform pricing refresh

### Changed
- `CostAnnotatingProcessor` skips spans whose `span.type` is set to anything other than `llm` (case-insensitive). Spans without `span.type` are unaffected

### Fixed
- Redaction processor moved to the `processors/` package that existing imports expect

## [0.1.21] - 2026-06-04

### Added
- Default `GovernanceEnrichmentProcessor` adds governance event metadata and integrity hashes on export
- `traccia.governance.disclosure()` records EU AI Act Art. 50 transparency evidence on the active span
- `init(compliance={...})` / `TRACCIA_COMPLIANCE_FRAMEWORKS` stamps the EU AI Act risk tier on spans when `eu_ai_act` is listed
- `init(redact_pii=True)` / `TRACCIA_REDACT_PII` registers `RedactionSpanProcessor` (email, US phone, and SSN-like patterns on sensitive attributes)
- `redact_string()` and `apply_redaction_to_span()` for manual redaction

## [0.1.20] - 2026-04-22

### Added
- Four-level pricing resolution: bundled snapshot, local cache, `TRACCIA_PRICING_OVERRIDE_JSON`, then `init(pricing_override=...)`
- Bundled `data/pricing_snapshot.json` (2,500+ models from LiteLLM, normalized to per-1K-token rates), regenerated on every release build
- `traccia pricing status`, `traccia pricing refresh` (Traccia pricing API first with ETag caching, LiteLLM fallback, `--source upstream` to force it), and `traccia pricing clear`. Cache path overridable with `TRACCIA_PRICING_CACHE_PATH`
- Per-span provenance: `llm.pricing.generated_at`, `llm.pricing.age_days`, `llm.pricing.snapshot_version`, and `llm.pricing.model_key`
- One-time staleness log per process: INFO after 7 days, WARNING after 30 days
- Process-wide `CostResolver` so span costs and the `gen_ai.client.operation.cost` metric use the same table

### Changed
- `load_pricing_with_source()` returns `(table, source, generated_at)` instead of a 2-tuple
- `AGENT_DASHBOARD_PRICING_JSON` is deprecated in favor of `TRACCIA_PRICING_OVERRIDE_JSON` (still accepted, with a one-time warning). `DEFAULT_PRICING` remains as an alias

### Fixed
- Model prefix matching prefers the longest key, so `gpt-4o` no longer resolves to `gpt-4` pricing
- `.env` is loaded from the process working directory, not the SDK package directory
- `pricing_override` passed to `init()` now applies to metrics, not only span attributes

## [0.1.19] - 2026-04-12

### Fixed
- The `guardrails` package is included in the published wheel. In 0.1.18 it was missing from the package list

## [0.1.18] - 2026-03-29

### Added
- Guardrail detection: `GuardrailDetectorProcessor`, registered automatically by `init()`, writes findings on spans with no code changes
- Three detection tiers: explicit (`guardrail_span()` or `@observe(as_type="guardrail")`), provider-native (finish and stop reasons, safety ratings), and heuristic (denial keywords in tool errors, low confidence)
- Missing-guardrail evaluation from observed capabilities, with a `guardrail.summary` on the root span
- `@observe(as_type="guardrail")` sets `guardrail.triggered` from a `bool` return value
- `traccia.guardrail.suppress_missing` (or `suppress_missing=` on `guardrail_span`) for batch and internal agents
- `init(guardrail_heuristics=False)` / `TRACCIA_GUARDRAIL_HEURISTICS` to turn off heuristic detection
- Public `traccia.guardrails` API: `GuardrailCategory`, `GuardrailFinding`, `GuardrailSummary`, `MissingGuardrail`, `SourceType`, `Confidence`, `EnforcementMode`, and `validate_guardrail_attributes()`

### Fixed
- `openai_agents`, `crewai`, and `guardrail_heuristics` passed flat to `init()` are no longer dropped during config loading

### Known Issues
- The `guardrails` package was missing from the wheel. Fixed in 0.1.19

## [0.1.17] - 2026-03-19

### Changed
- Histogram metrics (`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`, `gen_ai.client.operation.cost`) export with DELTA temporality instead of cumulative, which stops cost and token totals from inflating downstream. Tracing is unaffected

## [0.1.16] - 2026-03-08

### Fixed
- `AgentEnrichmentProcessor` writes `agent.id`, `agent.name`, `env`, `llm.cost.usd`, and `span.type` with `span.set_attribute()`, so LLM child spans under an orchestrator are exported with the right agent instead of `unknown-agent-name`

## [0.1.15] - 2026-03-05

### Added
- `init(service_role="orchestrator")` for services that host several logical agents, so the host `service.name` is not registered as an agent. Propagated to trace and metrics resources

### Fixed
- OpenAI, Anthropic, LangChain, OpenAI Agents SDK, and CrewAI metrics stamp the per-run agent id, name, and environment from `run_identity` on every data point
- CrewAI agent-level and LLM-level metrics carry run identity, and an overly broad guard no longer skips identity stamping

## [0.1.14] - 2026-02-26

### Added
- `runtime_config.run_identity(...)` context manager for run-scoped agent identity (id, name, env, tenant, project), so several agents can run concurrently in one process without re-init
- `force_flush(flush_timeout=...)` flushes spans and metrics without shutting down the tracer provider

### Changed
- `AgentEnrichmentProcessor` falls back to run-scoped identity after span attributes

## [0.1.13] - 2026-02-22

### Fixed
- Metrics are flushed and shut down on `stop_tracing()` and at exit
- Shutdown is idempotent, and the exit handler no longer runs a second shutdown after an explicit `stop_tracing()`

## [0.1.12] - 2026-02-20

### Added
- `agent_id`, `agent_name`, and `env` in `init()` or via `TRACCIA_AGENT_ID`, `TRACCIA_AGENT_NAME`, and `TRACCIA_ENV`, sent on trace and metrics resources
- `get_agent_identity()` and the `traccia.identity.AgentIdentity` model
- Per-span `agent.id` and `agent.name` through `@observe(attributes=...)` for multi-agent processes. Without an agent id, the platform falls back to `service.name`

## [0.1.11] - 2026-02-19

### Removed
- **Breaking:** the legacy `HttpExporter`. OTLP over HTTP (`OTLPExporter`) is the only network exporter

### Changed
- With `use_otlp=False`, the console or file exporter must be enabled. Validated in the config model and in `start_tracing()`

## [0.1.10] - 2026-02-14

### Changed
- Traces and metrics go to the Traccia platform (`api.traccia.ai`, `/v2/traces` and `/v2/metrics`) by default when no endpoint is set
- `traccia config init` writes the platform endpoint by default, and `traccia check` falls back to it when none is configured

## [0.1.9] - 2026-02-09

### Added
- Metrics: token usage, cost (USD), and operation duration under `gen_ai.client.*`, through a central metrics recorder
- Metrics from OpenAI, Anthropic, and `requests` instrumentation, the LangChain callback handler, CrewAI, and the OpenAI Agents SDK processor
- Metrics options in config and in `traccia config init`

## [0.1.8] - 2026-02-06

### Added
- CrewAI instrumentation: spans for `Crew.kickoff` and `kickoff_async`, `Task.execute_sync` and `execute_async`, `Agent.execute_task`, and `crewai.llm.LLM.call`, nested under the current context

## [0.1.7] - 2026-02-02

### Fixed
- `requests` instrumentation skips `/v2/traces` and `/api/v2/traces` ingestion calls, not only `/v1`, so exporting does not trace itself
- Default tenant and project ids are `default-tenant` and `default-project`

## [0.1.6] - 2026-02-02

### Added
- OpenAI Responses API instrumentation (`llm.openai.responses` spans with model, input, output, tokens, cost, and status)
- OpenAI Agents SDK integration (`TracciaAgentsTracingProcessor`) mapping agent, generation, function, handoff, guardrail, and response spans
- Auto-enabled by `init()` when `agents` is installed. Opt out with `init(openai_agents=False)`, `TRACCIA_OPENAI_AGENTS=false`, or `openai_agents = false` under `[instrumentation]`

## [0.1.5] - 2026-02-01

### Added
- LangChain integration: `TracciaCallbackHandler` (alias `CallbackHandler`) traces LLM and chat model runs with model, prompt, usage, and cost
- `pip install traccia[langchain]` extra

## [0.1.4] - 2026-01-28

### Added
- `tags` on the `@observe` decorator

## [0.1.3] - 2026-01-26

### Fixed
- Packaging: all subpackages are listed explicitly so they ship in the wheel

## [0.1.2] - 2026-01-26

### Fixed
- Packaging: include all subpackages (`exporter`, `instrumentation`, `processors`, and others). Republished as 0.1.2 because 0.1.1 was deleted from PyPI

## [0.1.1] - 2026-01-26 [REMOVED]

Deleted from PyPI. Use 0.1.2 or later.

## [0.1.0] - 2026-01-26

### Added
- First public release on PyPI
- `init()` with `traccia.toml` discovery, and `start_tracing()` for full control
- `@observe` decorator for sync and async functions
- Auto-instrumentation for OpenAI, Anthropic, and `requests`
- Token counting and local cost estimation on LLM spans
- OTLP, console, and file exporters
- Sampling, rate limiting, and Pydantic-validated configuration
- CLI: `traccia config init`, `traccia doctor`, and `traccia check`
