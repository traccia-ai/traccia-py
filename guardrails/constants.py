"""Reserved attribute namespace for guardrail detection in Traccia traces."""

# --- Span attribute keys (guardrail.* namespace) ---
ATTR_GUARDRAIL_CATEGORY = "guardrail.category"
ATTR_GUARDRAIL_NAME = "guardrail.name"
ATTR_GUARDRAIL_TRIGGERED = "guardrail.triggered"
ATTR_GUARDRAIL_ENFORCEMENT_MODE = "guardrail.enforcement_mode"
ATTR_GUARDRAIL_POLICY_ID = "guardrail.policy_id"
ATTR_GUARDRAIL_SOURCE_SDK = "guardrail.source_sdk"
ATTR_GUARDRAIL_EVIDENCE_TYPE = "guardrail.evidence_type"

# --- Existing span attributes consumed by detectors ---
ATTR_SPAN_TYPE = "span.type"
ATTR_AGENT_SPAN_TYPE = "agent.span.type"
ATTR_AGENT_GUARDRAIL_NAME = "agent.guardrail.name"
ATTR_AGENT_GUARDRAIL_TRIGGERED = "agent.guardrail.triggered"
ATTR_AGENT_TOOL_NAME = "agent.tool.name"
ATTR_LLM_MODEL = "llm.model"
ATTR_LLM_PROMPT = "llm.prompt"
ATTR_LLM_COMPLETION = "llm.completion"
ATTR_LLM_FINISH_REASON = "llm.finish_reason"
ATTR_LLM_RESPONSE_STATUS = "llm.response.status"
ATTR_LLM_VENDOR = "llm.vendor"
ATTR_LLM_STOP_REASON = "llm.stop_reason"
ATTR_LLM_SAFETY_RATINGS = "llm.safety_ratings"
ATTR_ERROR_TYPE = "error.type"
ATTR_ERROR_MESSAGE = "error.message"

# --- Suppression attribute (set by developer on any span to suppress missing warnings) ---
ATTR_GUARDRAIL_SUPPRESS_MISSING = "traccia.guardrail.suppress_missing"
