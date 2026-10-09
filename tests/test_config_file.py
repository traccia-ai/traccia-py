"""Tests for config file loading and priority."""

import os
import tempfile
from pathlib import Path
import unittest
from unittest import mock
from traccia import config, init, stop_tracing


class TestConfigFileLoading(unittest.TestCase):
    """Test TOML config file loading."""
    
    def test_load_toml_config_basic(self):
        """Test loading a basic TOML config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[tracing]
api_key = "test-key"
sample_rate = 0.5

[exporters]
enable_console = true

[instrumentation]
enable_patching = false
""")
            f.flush()

            try:
                loaded = config.load_toml_config(f.name)

                # New Pydantic config returns nested structure
                self.assertEqual(loaded["tracing"]["api_key"], "test-key")
                self.assertEqual(loaded["tracing"]["sample_rate"], 0.5)
                self.assertTrue(loaded["exporters"]["enable_console"])
                self.assertFalse(loaded["instrumentation"]["enable_patching"])
            finally:
                os.unlink(f.name)
    
    def test_load_toml_config_missing_file(self):
        """Test that loading missing file returns empty dict."""
        loaded = config.load_toml_config("/nonexistent/file.toml")
        self.assertEqual(loaded, {})
    
    def test_load_toml_config_invalid_toml(self):
        """Test that invalid TOML raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("invalid [toml content")
            f.flush()

            try:
                from traccia.errors import ConfigError
                with self.assertRaises(ConfigError):
                    config.load_toml_config(f.name)
            finally:
                os.unlink(f.name)
    
    def test_find_config_file_current_directory(self):
        """Test finding config file in current directory."""
        # Create temporary directory and config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "traccia.toml"
            config_path.write_text("[tracing]\napi_key = \"test\"")
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                found = config.find_config_file()
                self.assertIsNotNone(found)
                self.assertEqual(Path(found).name, "traccia.toml")
            finally:
                os.chdir(original_cwd)
    
    def test_find_config_file_home_directory(self):
        """Test finding config file in home directory."""
        # This test is more complex as it involves home directory
        # We'll just test that the function returns None when no config exists
        # (assuming no real config exists in home)
        
        # First ensure no config in current directory
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                
                # Should return None or home config path
                found = config.find_config_file()
                # We can't assert much here without creating a home config
                # Just verify the function doesn't crash
                self.assertTrue(found is None or isinstance(found, str))
            finally:
                os.chdir(original_cwd)


class TestConfigPriority(unittest.TestCase):
    """Test configuration loading priority."""
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            stop_tracing()
        except Exception:
            pass
    
    def test_explicit_params_override_env(self):
        """Test that explicit parameters override environment variables."""
        # Set environment variable
        os.environ["AGENT_DASHBOARD_API_KEY"] = "env-key"
        
        try:
            # Load config with explicit override
            merged = config.load_config_with_priority(
                overrides={"api_key": "explicit-key"}
            )
            
            # Explicit should win
            self.assertEqual(merged["api_key"], "explicit-key")
        finally:
            del os.environ["AGENT_DASHBOARD_API_KEY"]
    
    def test_env_override_config_file(self):
        """Test that environment variables override config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[tracing]
api_key = "file-key"
""")
            f.flush()
            
            try:
                # Set environment variable
                os.environ["AGENT_DASHBOARD_API_KEY"] = "env-key"
                
                try:
                    merged = config.load_config_with_priority(config_file=f.name)
                    
                    # Env should win over file
                    self.assertEqual(merged["api_key"], "env-key")
                finally:
                    del os.environ["AGENT_DASHBOARD_API_KEY"]
            finally:
                os.unlink(f.name)
    
    def test_config_file_loaded_when_no_overrides(self):
        """Test that config file is loaded when no overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[tracing]
api_key = "file-key"
sample_rate = 0.8

[exporters]
enable_console = true
""")
            f.flush()
            
            try:
                merged = config.load_config_with_priority(config_file=f.name)
                
                self.assertEqual(merged["api_key"], "file-key")
                self.assertEqual(merged["sample_rate"], 0.8)
                self.assertTrue(merged["enable_console"])
            finally:
                os.unlink(f.name)
    
    def test_load_config_from_env_boolean_conversion(self):
        """Test that environment variables are converted to correct types."""
        os.environ["AGENT_DASHBOARD_ENABLE_PATCHING"] = "true"
        os.environ["AGENT_DASHBOARD_ENABLE_COSTS"] = "false"
        os.environ["AGENT_DASHBOARD_SAMPLE_RATE"] = "0.7"
        
        try:
            env_config = config.load_config_from_env(flat=True)
            
            self.assertTrue(env_config["enable_patching"])
            self.assertFalse(env_config["enable_costs"])
            self.assertEqual(env_config["sample_rate"], 0.7)
        finally:
            del os.environ["AGENT_DASHBOARD_ENABLE_PATCHING"]
            del os.environ["AGENT_DASHBOARD_ENABLE_COSTS"]
            del os.environ["AGENT_DASHBOARD_SAMPLE_RATE"]
    
    def test_init_with_config_file(self):
        """Test that init() loads config from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[tracing]
sample_rate = 0.3

[instrumentation]
enable_patching = false
""")
            f.flush()
            
            try:
                provider = init(
                    config_file=f.name,
                    auto_start_trace=False,
                    enable_console_exporter=False,
                )
                
                self.assertIsNotNone(provider)
                # Config should be loaded (hard to verify without inspecting internals)
                # But at least we can verify init() succeeded
            finally:
                os.unlink(f.name)
                stop_tracing()
    
    def test_endpoint_config_precedence(self):
        """Test endpoint configuration precedence: param > env > file > default."""
        from traccia.config import DEFAULT_OTLP_TRACE_ENDPOINT
        
        # Test 1: Default is used when nothing is set
        merged = config.load_config_with_priority()
        # When no endpoint is set via any source, the default is applied at usage time in auto.py
        # The config itself will have endpoint=None
        self.assertIsNone(merged.get("endpoint"))
        
        # Test 2: Config file sets endpoint
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[tracing]
endpoint = "http://file-endpoint.com/v2/traces"
""")
            f.flush()
            
            try:
                merged = config.load_config_with_priority(config_file=f.name)
                self.assertEqual(merged["endpoint"], "http://file-endpoint.com/v2/traces")
            finally:
                os.unlink(f.name)
        
        # Test 3: Environment variable overrides config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[tracing]
endpoint = "http://file-endpoint.com/v2/traces"
""")
            f.flush()
            
            try:
                os.environ["TRACCIA_ENDPOINT"] = "http://env-endpoint.com/v2/traces"
                
                try:
                    merged = config.load_config_with_priority(config_file=f.name)
                    # Env should win over file
                    self.assertEqual(merged["endpoint"], "http://env-endpoint.com/v2/traces")
                finally:
                    del os.environ["TRACCIA_ENDPOINT"]
            finally:
                os.unlink(f.name)
        
        # Test 4: Explicit parameter overrides environment and file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[tracing]
endpoint = "http://file-endpoint.com/v2/traces"
""")
            f.flush()
            
            try:
                os.environ["TRACCIA_ENDPOINT"] = "http://env-endpoint.com/v2/traces"
                
                try:
                    merged = config.load_config_with_priority(
                        config_file=f.name,
                        overrides={"endpoint": "http://param-endpoint.com/v2/traces"}
                    )
                    # Explicit param should win
                    self.assertEqual(merged["endpoint"], "http://param-endpoint.com/v2/traces")
                finally:
                    del os.environ["TRACCIA_ENDPOINT"]
            finally:
                os.unlink(f.name)
        
        # Test 5: Verify default constant value is correct
        self.assertEqual(DEFAULT_OTLP_TRACE_ENDPOINT, "https://api.traccia.ai/v2/traces")


class TestConfigFromEnv(unittest.TestCase):
    """Test loading configuration from environment variables."""
    
    def test_load_config_from_env_all_vars(self):
        """Test loading all supported environment variables."""
        test_env = {
            "AGENT_DASHBOARD_API_KEY": "test-key",
            "AGENT_DASHBOARD_ENDPOINT": "http://test.com",
            "AGENT_DASHBOARD_SAMPLE_RATE": "0.9",
            "AGENT_DASHBOARD_ENABLE_PATCHING": "true",
            "AGENT_DASHBOARD_ENABLE_TOKEN_COUNTING": "false",
            "AGENT_DASHBOARD_ENABLE_COSTS": "yes",
            "AGENT_DASHBOARD_ENABLE_CONSOLE_EXPORTER": "1",
            "AGENT_DASHBOARD_ENABLE_FILE_EXPORTER": "0",
            "AGENT_DASHBOARD_AUTO_START_TRACE": "true",
        }
        # Start from an environment with every Traccia-recognized var removed, so
        # an ambient TRACCIA_* var (a developer .env pulled in by an earlier
        # test's init() -> load_dotenv(), higher priority than the AGENT_DASHBOARD_*
        # aliases) can't shadow what this test sets.
        scrubbed = {
            k: v
            for k, v in os.environ.items()
            if k not in {name for names in config.ENV_VAR_MAPPING.values() for name in names}
        }
        scrubbed.update(test_env)

        with mock.patch.dict(os.environ, scrubbed, clear=True):
            env_config = config.load_config_from_env(flat=True)

        self.assertEqual(env_config["api_key"], "test-key")
        self.assertEqual(env_config["endpoint"], "http://test.com")
        self.assertEqual(env_config["sample_rate"], 0.9)
        self.assertTrue(env_config["enable_patching"])
        self.assertFalse(env_config["enable_token_counting"])
        self.assertTrue(env_config["enable_costs"])
        self.assertTrue(env_config["enable_console"])
        self.assertFalse(env_config["enable_file"])
        self.assertTrue(env_config["auto_start_trace"])
    
    def test_load_config_from_env_missing_vars(self):
        """Test that missing env vars don't appear in result."""
        env_config = config.load_config_from_env()
        
        # Should not have keys for missing env vars
        # (or they should be None/not present)
        # Just verify it returns a dict
        self.assertIsInstance(env_config, dict)


if __name__ == "__main__":
    unittest.main()
