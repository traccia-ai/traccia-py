"""Canonical agent identity model for Traccia SDK."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AgentIdentity(BaseModel):
    """
    Single in-memory representation of agent identity.
    Used for resource attributes and consistent resolution across SDK and processors.
    """

    id: Optional[str] = Field(default=None, description="Stable agent identifier (agent.id)")
    name: Optional[str] = Field(default=None, description="Display name (agent.name)")
    type: Literal["workflow", "service", "tool"] = Field(
        default="workflow",
        description="Agent type for categorization",
    )
    env: Optional[str] = Field(default=None, description="Deployment environment (e.g. production, staging)")
    project: Optional[str] = Field(default=None, description="Project or namespace")

    def to_resource_attributes(self) -> dict:
        """Return a dict of resource attributes for OTLP export."""
        attrs = {}
        if self.id:
            attrs["agent.id"] = self.id
        if self.name:
            attrs["agent.name"] = self.name
        if self.env:
            attrs["environment"] = self.env
            attrs["env"] = self.env
        if self.project:
            attrs["project.id"] = self.project
        return attrs
