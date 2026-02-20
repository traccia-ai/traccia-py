"""Tests for agent identity model and init/start_tracing identity resolution."""

from __future__ import annotations

import os
import unittest

from traccia.identity import AgentIdentity
from traccia import runtime_config
from traccia import get_agent_identity


class TestAgentIdentity(unittest.TestCase):
    def test_agent_identity_model(self):
        identity = AgentIdentity(id="my-agent", name="My Agent", env="production", project="proj1")
        self.assertEqual(identity.id, "my-agent")
        self.assertEqual(identity.name, "My Agent")
        self.assertEqual(identity.env, "production")
        self.assertEqual(identity.project, "proj1")
        self.assertEqual(identity.type, "workflow")

    def test_to_resource_attributes(self):
        identity = AgentIdentity(id="a1", name="Agent One", env="staging")
        attrs = identity.to_resource_attributes()
        self.assertEqual(attrs.get("agent.id"), "a1")
        self.assertEqual(attrs.get("agent.name"), "Agent One")
        self.assertEqual(attrs.get("environment"), "staging")
        self.assertEqual(attrs.get("env"), "staging")

    def test_get_agent_identity_reflects_runtime_config(self):
        runtime_config.set_agent_id("test-id")
        runtime_config.set_agent_name("Test Agent")
        runtime_config.set_env("dev")
        runtime_config.set_project_id("p1")
        try:
            identity = get_agent_identity()
            self.assertEqual(identity.id, "test-id")
            self.assertEqual(identity.name, "Test Agent")
            self.assertEqual(identity.env, "dev")
            self.assertEqual(identity.project, "p1")
        finally:
            runtime_config.set_agent_id(None)
            runtime_config.set_agent_name(None)
            runtime_config.set_env(None)
            runtime_config.set_project_id(None)

    def test_get_agent_identity_defaults(self):
        runtime_config.set_agent_id(None)
        runtime_config.set_agent_name(None)
        runtime_config.set_env(None)
        runtime_config.set_project_id(None)
        identity = get_agent_identity()
        self.assertIsNone(identity.id)
        self.assertIsNone(identity.name)
        self.assertIsNone(identity.env)
        self.assertIsNone(identity.project)
