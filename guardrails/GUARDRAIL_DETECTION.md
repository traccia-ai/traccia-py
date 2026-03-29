# Guardrail Detection in Traccia

> **Internal reference.** The canonical copy of this document is at [`docs/GUARDRAIL_DETECTION.md`](../../docs/GUARDRAIL_DETECTION.md) in the monorepo root. This copy is kept alongside the source for developer convenience.

This document explains how guardrail detection works, what data is reliably capturable from traces, what the hard limits are, and how to use the helper APIs.

---

## How Detection Works

Guardrail detection is an **enrichment processor** that runs as each span ends. It inspects span attributes, classifies signals into findings, and writes results directly onto spans so they flow through the normal OTel export pipeline (file, OTLP, console).

Per-span findings are written as `guardrail.findings` (JSON) and `guardrail.finding.count` (int) on each span that produces findings. The aggregated summary is written onto the root span when it ends, since all child spans have already been processed by that point.

Detection is intentionally **passive**: it does not enforce anything at runtime. It only reads what was already emitted into trace spans.

State is scoped by `trace_id`, so concurrent agent runs in the same process are handled correctly.

### Detection Tiers

Every `GuardrailFinding` has a `source_type` and `confidence`:


| Tier | `source_type`     | `confidence`                           | How it is produced                                                                                        |
| ---- | ----------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| A    | `explicit`        | `high` (or `medium` if fields missing) | Typed guardrail span from OpenAI Agents SDK, or `@observe(as_type="guardrail")` with canonical attributes |
| B    | `provider_native` | `high` or `medium`                     | Deterministic provider response fields: `content_filter` finish reason, safety ratings, refusal status    |
| C    | `heuristic`       | always `low`                           | Pattern matching on error messages, denial keywords on tool spans                                         |


Heuristics are clearly tagged and excluded from coverage confidence unless they are the only available signal.

---

## What Can Be Accurately Captured

### Tier A — Explicit (High Confidence)

**OpenAI Agents SDK**

When you use the OpenAI Agents SDK with the Traccia processor, guardrail spans are automatically captured:

- `agent.span.type = "guardrail"`
- `agent.guardrail.name` = name of the guardrail
- `agent.guardrail.triggered` = `True` / `False`

These are written by the SDK itself, not inferred. Confidence is always `high`.

**Manual annotation with `@observe` (auto-triggered)**

If your guardrail function returns a `bool`, the decorator will automatically set `guardrail.triggered` from the return value — no manual span access needed:

```python
from traccia import observe

@observe(
    as_type="guardrail",
    attributes={
        "guardrail.name": "pii_scanner",
        "guardrail.category": "pii",
        "guardrail.enforcement_mode": "warn",
    }
)
def check_pii(text: str) -> bool:
    return has_pii(text)  # True → triggered, False → not triggered
```

The `guardrail.triggered` attribute is set automatically from the return value. If you pre-set `guardrail.triggered` in `attributes={}`, that value is preserved and the return value is not used.

For non-bool returns (e.g. dicts, strings), you must set `guardrail.triggered` manually via `get_current_span()`.

Or with the `guardrail_span` context manager (recommended for inline guardrail checks):

```python
from traccia.guardrails import guardrail_span

with guardrail_span("pii_check", category="pii", enforcement_mode="warn") as span:
    result = run_pii_check(text)
    span.set_attribute("guardrail.triggered", result.found_pii)
```

`guardrail_span` pre-fills all canonical `guardrail.*` attributes and sets `span.type = "guardrail"`, so you only need to set `guardrail.triggered` based on your check's outcome.

**Required attributes for `high` confidence:**

- `guardrail.name`
- `guardrail.category`
- `guardrail.triggered`

If any are missing, confidence is downgraded to `medium` and a warning is logged to `traccia.guardrails`.

### Tier B — Provider-Native (Medium/High Confidence)

**OpenAI / Azure OpenAI — `llm.finish_reason`**

| `finish_reason` value | Provider         | Confidence |
| --------------------- | ---------------- | ---------- |
| `content_filter`      | OpenAI           | `high`     |
| `content_filtered`    | Azure OpenAI     | `high`     |
| `SAFETY`              | Google GenAI     | `high`     |

All map to `category = moderation`, `triggered = True`, `enforcement_mode = block`.

**Anthropic — `llm.stop_reason`**

When Claude refuses a request, it sets `stop_reason` to `content_filter` or `content_filtered`. A finding is emitted only if `llm.vendor` is `anthropic` or `claude` (so other stacks reusing the same strings do not false-positive). Traccia maps matches to `moderation / high / block`.

**Anthropic — policy errors on `error.message`**

If `error.message` contains policy-violation phrases, a `moderation / medium` finding is emitted only when **both** (1) `llm.vendor` is `anthropic` or `claude`, and (2) the span looks like an LLM call (`span.type=llm` or any of `llm.model`, `llm.prompt`, `llm.completion`, `llm.finish_reason`, `llm.stop_reason` present).

**Google / LangChain — `llm.safety_ratings`**

The LangChain callback captures `safety_ratings` from response metadata as a JSON string. If any entry has `"probability": "HIGH"` or `"blocked": true`, a `moderation / medium / block` finding is emitted. Low-probability entries without `blocked=true` are ignored.

**Response status with refusal text**

When `llm.response.status` is `incomplete` or `failed` *and* the completion text matches known refusal phrases, this is classified as `provider_native / medium / moderation`.

**LangChain finish reason**

The LangChain callback extracts `finish_reason` from `generation_info` and `response_metadata`, enabling Tier B detection for LangChain-routed LLM calls.

### Tier C — Heuristic (Low Confidence, Labeled)

Tier C is **on by default**. Disable with `init(guardrail_heuristics=False)`, `traccia.toml` `[instrumentation] guardrail_heuristics = false`, or `TRACCIA_GUARDRAIL_HEURISTICS=false`. Tier A/B unchanged.

**Processor failures:** Logged at **warning** (`traccia.guardrails.processor`); export continues. **DEBUG** on that logger for tracebacks.

**Tool denial errors**

When a tool/function span errors with a message containing denial-like keywords (`permission`, `denied`, `unauthorized`, `forbidden`, `not allowed`), a heuristic finding is created:

- `category = tool_permission`
- `confidence = low`
- `source_type = heuristic`

These are always labeled as heuristic and excluded from `coverage_confidence=high` calculations.

**Known Tier C false-positive sources**

The following `error.type` values produce denial-sounding messages but are infrastructure/network failures, not policy violations. They are excluded from heuristic matching regardless of keywords in the error message:

| `error.type`                          | Typical message pattern                              |
| ------------------------------------- | ---------------------------------------------------- |
| `TimeoutError`, `asyncio.TimeoutError`| "Connection denied: timed out"                       |
| `ConnectionError`, `ConnectionRefusedError`, `ConnectionResetError` | "Connection refused: not allowed" |
| `OSError`                             | "Permission denied: cannot bind to port"             |
| `socket.timeout`                      | "Request timed out: connection not allowed"          |
| `requests.exceptions.Timeout`, `requests.exceptions.ConnectionError` | Standard requests errors |
| `httpx.TimeoutException`, `httpx.ConnectError` | httpx equivalents                         |

If your infrastructure errors still match denial keywords and cause false positives, use the [suppression mechanism](#suppression-for-false-positive-missing-warnings) to suppress the affected category from the missing list.

**Tier C findings do NOT count as coverage**

A `low`-confidence heuristic finding appears in `detected_categories` but does **not** remove `tool_permission` from `missing_categories`. The reasoning: seeing a possible tool denial is not the same as having a tool permission guardrail. Both signals are preserved — developers can see the heuristic fired and that coverage is still considered incomplete.

---

## What CANNOT Be Reliably Captured

These are technical hard limits, not gaps to be closed incrementally:


| What you might want                                        | Why it cannot be captured from traces                                                                                                                                                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prove a guardrail exists                                   | Only possible if the guardrail emits a trace signal. Out-of-band guardrails (API gateways, proxies, external validators) are invisible unless they write to the trace.                                                          |
| Verify a guardrail works correctly                         | Traces show observed outcomes, not policy correctness. A guardrail can exist and be misconfigured; traces cannot distinguish this.                                                                                              |
| Detect prompt injection defense                            | You cannot reliably infer from model output alone that a prompt injection check ran. Only an explicit span or structured SDK signal counts.                                                                                     |
| Guarantee PII protection from attribute key names          | Key-name heuristics (`api_key`, `token`) catch credential-like patterns but miss semantic PII. You need a dedicated classifier emitting a structured finding.                                                                   |
| Infer full policy semantics                                | Rule sets, legal mappings, and policy versions are not in traces unless the developer emits them via `guardrail.policy_id`.                                                                                                     |
| Classify generic errors as guardrails                      | Timeouts, auth failures, and transient errors look similar to denials. Without a denial-specific keyword, they are not classified.                                                                                              |
| Read span events post-facto                                | The current `Span.events` wrapper returns an empty list on active spans. Guardrail detection relies only on span attributes, not events.                                                                                        |
| Distinguish "guardrail passed" from "no guardrail present" | If `guardrail.triggered = False`, it means the guardrail ran and did not fire. If no guardrail span exists, there is no signal — the missing-guardrail evaluator reports this. The two cases look very different in the output. |


---

## Missing Guardrail Evaluation

The evaluator infers capabilities from all spans in a run and compares detected guardrail categories to what is expected for those capabilities.

### Capability inference rules


| Capability             | Evidence in traces                                                               |
| ---------------------- | -------------------------------------------------------------------------------- |
| `calls_llm`            | Any span with `llm.model`                                                        |
| `handles_user_text`    | Any span with `llm.prompt`                                                       |
| `produces_user_text`   | Any span with `llm.completion`                                                   |
| `uses_tools`           | Any span with `span.type=tool`, `agent.span.type=function`, or `agent.tool.name` |
| `has_external_actions` | Tool spans or handoff spans                                                      |


### Required-guardrail matrix


| Observed capabilities             | Expected guardrail categories          |
| --------------------------------- | -------------------------------------- |
| `calls_llm` + `handles_user_text` | `input_validation`, `prompt_injection` |
| `produces_user_text`              | `output_validation`, `moderation`      |
| `uses_tools`                      | `tool_permission`, `output_validation` |


Missing categories are reported as `MissingGuardrail` objects with `missing_confidence` levels:

- `high`: capability was explicit and complete (e.g. tool spans are deterministic). Used for `tool_permission`.
- `medium`: capability inferred from reliable but indirect signals. Used for `input_validation` and `prompt_injection` — `llm.prompt` is present but we cannot confirm the prompt came from user input (could be an internal batch pipeline).
- `low`: capability inferred heuristically only.

**Why `input_validation` and `prompt_injection` are `MEDIUM` confidence**

Any LLM call sets `llm.prompt`, including batch pipelines over internal documents. We cannot determine from the trace alone whether the prompt came from user input. `MEDIUM` accurately reflects: "we have a structural reason to expect this guardrail, but cannot confirm the prompt is user-provided." The `why_required` message reads: *"Agent makes LLM calls with prompt data (may be user-provided)"*.

---

## Suppression for False Positive Missing Warnings

If your agent is a batch pipeline or internal-only service where certain guardrail categories genuinely do not apply, you can suppress them from the missing list using the `traccia.guardrail.suppress_missing` span attribute:

```python
from traccia.guardrails import guardrail_span, ATTR_GUARDRAIL_SUPPRESS_MISSING
from opentelemetry import trace

# Option 1: via guardrail_span convenience parameter
with guardrail_span(
    "batch_root",
    category="unknown",
    suppress_missing=["prompt_injection", "input_validation"],
) as span:
    run_batch_pipeline()

# Option 2: set directly on any span
span = trace.get_current_span()
span.set_attribute(
    ATTR_GUARDRAIL_SUPPRESS_MISSING,
    ["prompt_injection", "input_validation"],
)
```

The attribute can be set on **any span** in the run (root or child). The evaluator collects suppression requests from all spans before computing the missing list.

Suppression only affects the **missing** list. It does not remove detected findings and does not affect `detected_categories` or `triggered_categories`.

**Valid category values:** use the string values of `GuardrailCategory` — `"input_validation"`, `"prompt_injection"`, `"pii"`, `"moderation"`, `"tool_permission"`, `"output_validation"`, `"rate_limit"`, `"unknown"`.

> **UI future note:** The Traccia platform will eventually let you permanently suppress specific missing-guardrail categories per agent and mark individual findings as false positives through the UI. These will be stored as agent-level policies in the Traccia database and applied when displaying the guardrail posture panel. The SDK suppression above is the equivalent for self-hosted or open-source deployments.

---

## Output Schema

### `GuardrailFinding`

```json
{
  "id": "a1b2c3d4e5f6a7b8",
  "category": "moderation",
  "name": "provider_content_filter",
  "source_type": "provider_native",
  "confidence": "high",
  "triggered": true,
  "enforcement_mode": "block",
  "status": "triggered",
  "evidence_ref": {
    "trace_id": "...",
    "span_id": "...",
    "integration": "openai",
    "attribute_keys": ["llm.finish_reason"]
  },
  "detection_reason": "llm_finish_reason_content_filter",
  "raw_excerpt": "content_filter"
}
```

### `GuardrailSummary`

```json
{
  "detected_categories": ["moderation"],
  "triggered_categories": ["moderation"],
  "missing_categories": [
    {
      "category": "input_validation",
      "why_required": "Agent makes LLM calls with prompt data (may be user-provided)",
      "missing_confidence": "medium",
      "evidence_ref": {"attribute_keys": ["calls_llm", "handles_user_text"]}
    }
  ],
  "coverage_confidence": "high",
  "capabilities_observed": ["calls_llm", "handles_user_text", "produces_user_text"],
  "limitations": [
    "1 expected guardrail category not detected in trace."
  ]
}
```

---

## Where the Data Appears

Guardrail data is written as span attributes and flows through every configured exporter:

**On individual spans** (when a guardrail signal is detected on that span):
- `guardrail.finding.count` (int)
- `guardrail.findings` (JSON string -- array of GuardrailFinding dicts)

**On the root span** (aggregated summary of the entire run):
- `guardrail.summary` (JSON string -- full GuardrailSummary dict)
- `guardrail.summary.detected_categories` (list of strings)
- `guardrail.summary.missing_count` (int)
- `guardrail.summary.coverage_confidence` (string: `high`, `medium`, or `low`)
- `guardrail.findings` (JSON string -- all findings from all spans in the run)
- `guardrail.finding.count` (int -- total across the run)

These attributes are visible in file exports, OTLP traces, and the console exporter.

---

## Accuracy Policy

- **Precision over recall**: when uncertain, a finding is classified as `unknown` or not emitted at all.
- **No confidence inflation**: a heuristic finding can never be upgraded to `high`.
- **Every finding is traceable**: `evidence_ref` always points back to the span/attribute that produced it.
- **Limitations are always surfaced**: `guardrail_summary.limitations` lists every uncertainty.

---

## Reserved Span Attribute Namespace


| Attribute                              | Type            | Description                                               |
| -------------------------------------- | --------------- | --------------------------------------------------------- |
| `guardrail.name`                       | `str`           | Human-readable guardrail name                             |
| `guardrail.category`                   | `str`           | Category enum value                                       |
| `guardrail.triggered`                  | `bool`          | Whether the guardrail fired                               |
| `guardrail.enforcement_mode`           | `str`           | `block`, `warn`, `log_only`, `unknown`                    |
| `guardrail.policy_id`                  | `str`           | Optional policy version reference                         |
| `guardrail.source_sdk`                 | `str`           | Which integration emitted this                            |
| `guardrail.evidence_type`              | `str`           | `span_attribute`, `provider_field`, `heuristic`           |
| `traccia.guardrail.suppress_missing`   | `list[str]`     | Categories to suppress from missing-guardrail report      |


