"""Sampling decisions for traces."""

import random
from dataclasses import dataclass

from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.sdk.trace.sampling import Sampler as OTelSampler


@dataclass
class SamplingResult:
    sampled: bool


class Sampler:
    """Head-based sampler using a fixed probability."""

    def __init__(self, sample_rate: float = 1.0) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        self.sample_rate = sample_rate

    def should_sample(self) -> SamplingResult:
        # random() returns values in [0.0, 1.0), so a strict comparison keeps
        # rate 0 at never and rate 1 at always.
        return SamplingResult(sampled=random.random() < self.sample_rate)

    def to_otel_sampler(self) -> OTelSampler:
        """
        Build the OpenTelemetry sampler that applies this rate at span creation.

        Root spans are sampled by trace ID ratio. Child spans follow their
        parent's decision, so a trace is exported whole or not at all.
        """
        return ParentBased(root=TraceIdRatioBased(self.sample_rate))
