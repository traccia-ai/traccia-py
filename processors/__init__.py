"""Span processors and supporting utilities."""

from traccia.processors.batch_processor import BatchSpanProcessor
from traccia.processors.drop_policy import (
    DEFAULT_DROP_POLICY,
    DropNewestPolicy,
    DropOldestPolicy,
    DropPolicy,
)
from traccia.processors.sampler import Sampler, SamplingResult
from traccia.processors.token_counter import TokenCountingProcessor, estimate_tokens_from_text
from traccia.processors.cost_engine import compute_cost, DEFAULT_PRICING
from traccia.processors.cost_processor import CostAnnotatingProcessor
from traccia.processors.logging_processor import LoggingSpanProcessor
from traccia.processors.agent_enricher import AgentEnrichmentProcessor
from traccia.processors.rate_limiter import RateLimiter, RateLimitingSpanProcessor
from traccia.processors.guardrail_detector import GuardrailDetectorProcessor

__all__ = [
    "BatchSpanProcessor",
    "DropPolicy",
    "DropOldestPolicy",
    "DropNewestPolicy",
    "DEFAULT_DROP_POLICY",
    "Sampler",
    "SamplingResult",
    "TokenCountingProcessor",
    "estimate_tokens_from_text",
    "compute_cost",
    "DEFAULT_PRICING",
    "CostAnnotatingProcessor",
    "LoggingSpanProcessor",
    "AgentEnrichmentProcessor",
    "RateLimiter",
    "RateLimitingSpanProcessor",
    "GuardrailDetectorProcessor",
]
