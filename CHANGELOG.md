# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added issue templates (bug report, feature request).
- Added documentation templates.

### Changed
- Updated `CONTRIBUTING.md` with issue reporting guidelines and LICENSE reference.

## [0.1.22] - 2026-06-20
### Added
- `span_scope()`, `SpanScope`, `run_with_span()`, and `run_with_span_async()` for explicit span lifecycle control.
- `traccia.pricing_normalizer` module exposing `normalize()` and `diff_models()`.

### Changed
- `CostAnnotatingProcessor` now skips cost annotation for non-LLM spans.

### Fixed
- Moved `redaction_processor` package to the correct `processors/` path.
- Broken `TestNormalizer` import path in tests.

## [0.1.21] - 2026-06-04
### Added
- Governance on spans: `GovernanceEnrichmentProcessor` adds governance event metadata.
- `traccia.governance.disclosure()` to record Art. 50 transparency evidence.
- EU AI Act risk tier stamping when `eu_ai_act` is listed in compliance frameworks.
- PII redaction: `RedactionSpanProcessor` (email, US phone, SSN-like patterns) via `init(redact_pii=True)`.
- `redact_string()` and `apply_redaction_to_span()` for manual PII redaction.

## [0.1.20] - 2026-04-22
### Added
- 4-level cost calculation resolution chain (Bundled snapshot, Local cache, Env var override, Programmatic override).
- Four new per-span pricing provenance attributes for LLM spans.
- Process-level `CostResolver` singleton for thread-safe pricing updates.
- `traccia pricing status`, `refresh`, and `clear` CLI commands.
- `tests/test_pricing.py` covering pricing components.

### Changed
- `load_pricing_with_source()` returns a 3-tuple including the `generated_at` timestamp.

### Fixed
- Prefix match in `cost_engine.py` to correctly resolve overlapping model keys (e.g., `gpt-4o` vs `gpt-4`).
- Environment variable `.env` loading by passing explicit dotenv path to process CWD.

## [0.1.19] - 2026-04-12
### Fixed
- Included `guardrails` package in `pyproject.toml`.

## [0.1.18] - 2026-03-29
### Added
- Guardrail Detection: A passive, zero-overhead observability layer.
- `GuardrailDetectorProcessor` auto-registers on `traccia.init()`.
- Explicit annotation (Tier A) via `guardrail_span()` context manager and `@observe(as_type="guardrail")` decorator.
- Provider-native detection (Tier B) for Azure, Anthropic, Google GenAI, and OpenAI.
- Heuristic detection (Tier C) for denial-like tool error messages.
- Inference capabilities to evaluate missing-guardrail expected categories.
- SDK-level suppression for false positive missing warnings.
- `traccia.guardrails` public API and constants.

## [0.1.17] - 2026-03-19
### Changed
- Metrics exporter now uses DELTA temporality for histogram instruments to prevent runaway cost/token inflation.

## [0.1.16] - 2026-03-08
### Fixed
- Agent attribution fix: `AgentEnrichmentProcessor` now sets attributes via `span.set_attribute()` so they are correctly exported to OTel.

## [0.1.15] - 2026-03-05
### Added
- Identity propagation for multi-agent orchestrators via `service_role="orchestrator"` in `traccia.init()`.
- Per-run agent identity stamped on all framework metric data points.

### Fixed
- CrewAI instrumentation metric stamping and overly broad guard condition suppression.

## [0.1.14] - 2026-02-26
### Added
- `runtime_config.run_identity(...)` context manager for run-scoped agent identity.
- `force_flush(flush_timeout)` helper exposed to flush spans/metrics without shutting down the tracer provider.

### Changed
- `AgentEnrichmentProcessor` now respects run-scoped identity as a fallback.

## [0.1.13] - 2026-02-22
### Changed
- Flush and shutdown metrics on exit via `stop_tracing()` and `atexit`.
- Idempotent shutdown to prevent meter provider from double flushing or double shutting down.

## [0.1.12] - 2026-02-20
### Added
- Agent identity and environment configuration in `init()` or via environment variables (`TRACCIA_AGENT_ID`, `TRACCIA_AGENT_NAME`, `TRACCIA_ENV`).
- `get_agent_identity()` helper method.
- `AgentIdentity` canonical model.

### Changed
- Default agent identity uses `service.name` if not set explicitly.

## [0.1.11] - 2026-02-19
### Changed
- Config semantics enforce that when `use_otlp=False`, a console or file exporter must be explicitly enabled.

### Removed
- Legacy `HttpExporter` has been removed. OTLP HTTP (`OTLPExporter`) is now the only network exporter path.

## [0.1.10] - 2026-02-14
### Changed
- Send Traccia SDK traces to the Traccia platform by default.
- Refactored backend trace ingestion to use the default `api.traccia.ai` endpoint.

## [0.1.9] - 2026-02-09
### Added
- End-to-end metrics support covering token usage, operation cost, and latency.
- Global metrics recorder accessor.
- Framework metrics integrations for LangChain, CrewAI, and OpenAI Agents SDK.
- CLI config `init` support to produce a config compatible with the metrics system.

### Changed
- Core instrumentation updated to attach pricing-aware metrics.

## [0.1.8] - 2026-02-06
### Added
- CrewAI instrumentation monkey-patching `Crew.kickoff`, `Task.execute_sync`, `Agent.execute_task`, and `LLM.call`.

## [0.1.7] - 2026-02-02
### Fixed
- Skipped instrumentation for ingestion calls.
- Trace ingestion endpoint bugs and reverted base URLs.

## [0.1.6] - 2026-02-02
### Added
- OpenAI Responses API instrumentation (used by the Agents SDK).
- OpenAI Agents SDK integration to map agent handoffs, tools, and spans to Traccia.
- Auto-detection feature to silently install `agents` tracking if available.
- Configuration fields for OpenAI Agents.

## [0.1.5] - 2026-02-01
### Added
- LangChain support with `TracciaCallbackHandler`.

## [0.1.4] - 2026-01-28
### Added
- `tags` support for the `observe` decorator.

## [0.1.3] - 2026-01-26
### Fixed
- Package discovery issue by explicitly listing all subpackages for package-dir mapping.

## [0.1.2] - 2026-01-26
### Fixed
- Version bump following a PyPI deletion.

## [0.1.1] - 2026-01-26
### Fixed
- Included all subpackages in setuptools configuration.

## [0.1.0] - 2026-01-26
### Added
- Initial release of the Traccia SDK for AI agent observability.
- PyPI publishing workflow.
- Applied the Apache 2.0 License.

[Unreleased]: https://github.com/traccia-ai/traccia-py/compare/v0.1.22...HEAD
[0.1.22]: https://github.com/traccia-ai/traccia-py/compare/v0.1.21...v0.1.22
[0.1.21]: https://github.com/traccia-ai/traccia-py/compare/v0.1.20...v0.1.21
[0.1.20]: https://github.com/traccia-ai/traccia-py/compare/v0.1.19...v0.1.20
[0.1.19]: https://github.com/traccia-ai/traccia-py/compare/v0.1.18...v0.1.19
[0.1.18]: https://github.com/traccia-ai/traccia-py/compare/v0.1.17...v0.1.18
[0.1.17]: https://github.com/traccia-ai/traccia-py/compare/v0.1.16...v0.1.17
[0.1.16]: https://github.com/traccia-ai/traccia-py/compare/v0.1.15...v0.1.16
[0.1.15]: https://github.com/traccia-ai/traccia-py/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/traccia-ai/traccia-py/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/traccia-ai/traccia-py/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/traccia-ai/traccia-py/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/traccia-ai/traccia-py/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/traccia-ai/traccia-py/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/traccia-ai/traccia-py/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/traccia-ai/traccia-py/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/traccia-ai/traccia-py/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/traccia-ai/traccia-py/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/traccia-ai/traccia-py/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/traccia-ai/traccia-py/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/traccia-ai/traccia-py/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/traccia-ai/traccia-py/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/traccia-ai/traccia-py/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/traccia-ai/traccia-py/releases/tag/v0.1.0
