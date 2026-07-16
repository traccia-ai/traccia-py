# Changelog

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
