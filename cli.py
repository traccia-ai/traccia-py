"""CLI for traccia utilities."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Optional

from traccia.config import (
    validate_config,
    load_config,
    find_config_file,
    ENV_VAR_MAPPING,
    DEFAULT_OTLP_TRACE_ENDPOINT,
)
from traccia.errors import ConfigError


def _check(args) -> int:
    """Check connectivity to the configured exporter endpoint."""
    # Load config to get endpoint (same resolution as SDK: param > config > default)
    try:
        config = load_config(config_file=args.config if hasattr(args, 'config') else None)
        endpoint = args.endpoint or config.tracing.endpoint or DEFAULT_OTLP_TRACE_ENDPOINT
        
        print(f"🔍 Checking connectivity to {endpoint}...")
        sys.stdout.flush()  # Ensure output appears before any errors
        
        # Try HEAD request first
        req = urllib.request.Request(endpoint, method="HEAD")
        if args.api_key or config.tracing.api_key:
            api_key = args.api_key or config.tracing.api_key
            req.add_header("Authorization", f"Bearer {api_key}")
        
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                code = resp.getcode()
                print(f"✅ Endpoint is reachable (HTTP {code})")
                print("💡 Connectivity test successful!")
                return 0
        except urllib.error.HTTPError as e:
            # HTTP 405 (Method Not Allowed), 400 (Bad Request), or 401 (Unauthorized) 
            # means the endpoint is reachable and responding - just doesn't like our test request
            if e.code in [400, 401, 405]:
                print(f"✅ Endpoint is reachable (HTTP {e.code})")
                if e.code == 405:
                    print("💡 Endpoint only accepts specific methods (expected for OTLP endpoints)")
                elif e.code == 401:
                    print("⚠️  Authentication required - check your API key")
                elif e.code == 400:
                    print("💡 Endpoint rejected test payload (expected for OTLP endpoints)")
                print("✅ Connectivity test successful!")
                return 0
            else:
                # Other HTTP errors (404, 500, etc.) are actual failures
                print(f"❌ HTTP Error {e.code}: {e.reason}", file=sys.stderr)
                return 1
                
    except ConfigError as exc:
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"❌ Connection failed: {exc.reason}", file=sys.stderr)
        print("   Make sure the endpoint is running and accessible", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"❌ Unexpected error: {exc}", file=sys.stderr)
        return 1


def _config_init(args) -> int:
    """Initialize traccia.toml config file in current directory."""
    config_path = os.path.join(os.getcwd(), "traccia.toml")
    
    # Check if file already exists
    if os.path.exists(config_path) and not args.force:
        print(f"❌ Config file already exists at {config_path}", file=sys.stderr)
        print("   Use --force to overwrite", file=sys.stderr)
        return 1
    
    # Create config template with important parameters
    config_template = """# Traccia SDK Configuration File
# NOTE: This file only includes the most commonly used options.
# For a complete list of configuration keys (including advanced and metrics options),
# see the official docs: https://traccia.ai/docs/reference/configuration

[tracing]
# API key for authentication (required for SaaS, optional for open-source)
api_key = ""

# Endpoint URL for trace ingestion (default: Traccia platform)
# For local OTLP backends use e.g. endpoint = "http://localhost:4318/v1/traces"
endpoint = "{default_endpoint}"

# Sampling rate (0.0 to 1.0) - controls what percentage of traces are sent
sample_rate = 1.0

# Auto-start a root trace on init (default: true)
auto_start_trace = true

# Name for the auto-started root trace
auto_trace_name = "root"

# Use OTLP exporter (default: true)
# Set to false if using console or file exporter
use_otlp = true

# Service name (optional)
# service_name = "my-app"

[exporters]
# IMPORTANT: Only enable ONE exporter at a time (console, file, or OTLP via use_otlp)

# Enable console exporter for local debugging
enable_console = false

# Enable file exporter to write traces to local file
enable_file = false

# File path for file exporter (only used if enable_file = true)
file_exporter_path = "traces.jsonl"

# Reset/clear trace file on initialization
reset_trace_file = false

[instrumentation]
# Auto-patch popular libraries (OpenAI, Anthropic, requests)
enable_patching = true

# Count tokens for LLM calls
enable_token_counting = true

# Calculate costs for LLM calls
enable_costs = true

# Auto-instrument tool calls (experimental)
auto_instrument_tools = false

# Maximum number of tool spans to create
max_tool_spans = 100

# Maximum depth of nested spans
max_span_depth = 10

[rate_limiting]
# Maximum spans per second (uncomment to enable rate limiting)
# max_spans_per_second = 100.0

# Maximum queue size for buffered spans
max_queue_size = 5000

# Maximum milliseconds to block before dropping spans
max_block_ms = 100

# Maximum number of spans in a single export batch
max_export_batch_size = 512

# Delay in milliseconds between export batches
schedule_delay_millis = 5000

[metrics]
# Enable OpenTelemetry metrics emission (LLM & agent metrics)
enable_metrics = true

# Metrics endpoint URL (defaults to {traces_base}/v2/metrics). Override this to
# send metrics to a different OTLP/HTTP endpoint (e.g. OTEL Collector):
# metrics_endpoint = "http://localhost:4318/v1/metrics"

# Metrics sampling rate (0.0 to 1.0, default: 1.0 = 100%)
metrics_sample_rate = 1.0

[runtime]
# Runtime metadata (optional - can be set per-session)
# session_id = ""
# user_id = ""
# tenant_id = ""
# project_id = ""

[logging]
# Enable debug logging
debug = false

# Enable span-level logging
enable_span_logging = false

[advanced]
# Maximum length for attribute values (uncomment to set limit)
# attr_truncation_limit = 1000
"""
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_template.format(default_endpoint=DEFAULT_OTLP_TRACE_ENDPOINT))
        print(f"✅ Created config file at {config_path}")
        print("\n📝 Next steps:")
        print("   1. Edit the config file to add your API key and endpoint")
        print("   2. Run `traccia doctor` to validate your configuration")
        print("   3. Run `traccia check` to test connectivity")
        return 0
    except Exception as exc:
        print(f"❌ Failed to create config file: {exc}", file=sys.stderr)
        return 1


def _doctor(args) -> int:
    """Validate configuration and diagnose common issues."""
    print("🩺 Running Traccia configuration diagnostics...\n")
    
    issues_found = 0
    
    # 1. Check for config file
    config_file = None
    if hasattr(args, 'config') and args.config:
        config_file = args.config
        if not os.path.exists(config_file):
            print(f"❌ Specified config file not found: {config_file}")
            issues_found += 1
            return 1
    else:
        config_file = find_config_file()
        if config_file:
            print(f"✅ Found config file: {config_file}")
        else:
            print("⚠️  No config file found (checked ./traccia.toml and ~/.traccia/config.toml)")
            print("   Run `traccia config init` to create one")
            issues_found += 1
    
    # 2. Check environment variables
    print("\n📋 Environment variables:")
    found_env_vars = []
    for config_key, env_vars in ENV_VAR_MAPPING.items():
        for env_var in env_vars:
            if os.getenv(env_var):
                found_env_vars.append(env_var)
                print(f"   ✅ {env_var} is set")
    
    if not found_env_vars:
        print("   ℹ️  No Traccia environment variables set")
    
    # 3. Validate configuration
    print("\n🔍 Validating configuration...")
    is_valid, message, config = validate_config(config_file=config_file)
    
    if is_valid:
        print(f"✅ {message}")
        
        # Print configuration summary
        effective_endpoint = config.tracing.endpoint or DEFAULT_OTLP_TRACE_ENDPOINT
        endpoint_source = "config/file" if config.tracing.endpoint else "default (Traccia platform)"
        print("\n📊 Configuration summary:")
        print(f"   • API Key: {'✅ Set' if config.tracing.api_key else '❌ Not set'}")
        print(f"   • Endpoint: {effective_endpoint} ({endpoint_source})")
        print(f"   • Sample Rate: {config.tracing.sample_rate}")
        print(f"   • OTLP Exporter: {'✅ Enabled' if config.tracing.use_otlp else '❌ Disabled'}")
        print(f"   • Console Exporter: {'✅ Enabled' if config.exporters.enable_console else '❌ Disabled'}")
        print(f"   • File Exporter: {'✅ Enabled' if config.exporters.enable_file else '❌ Disabled'}")
        print(f"   • Auto-patching: {'✅ Enabled' if config.instrumentation.enable_patching else '❌ Disabled'}")
        
        # Check for potential issues (no warning when endpoint is unset — SDK uses default)
        
        if not config.tracing.use_otlp and not config.exporters.enable_console and not config.exporters.enable_file:
            print("\n❌ Error: No exporter is enabled! Traces won't be exported anywhere.")
            issues_found += 1
        
        if config.rate_limiting.max_spans_per_second:
            print(f"\n   ℹ️  Rate limiting enabled: {config.rate_limiting.max_spans_per_second} spans/sec")
    else:
        print(f"❌ {message}")
        issues_found += 1
    
    # 4. Environment variable mapping reference
    print("\n📖 Environment Variable Reference:")
    print("   Common variables:")
    print("   • TRACCIA_API_KEY or AGENT_DASHBOARD_API_KEY")
    print("   • TRACCIA_ENDPOINT or AGENT_DASHBOARD_ENDPOINT")
    print("   • TRACCIA_SAMPLE_RATE")
    print("   • TRACCIA_DEBUG")
    print("\n   For a complete list, see: ENV_VAR_MAPPING in traccia/config.py")
    
    # Summary
    print("\n" + "="*60)
    if issues_found == 0:
        print("✅ No issues found! Your configuration looks good.")
        print("\n💡 Tip: Run `traccia check` to test connectivity to your endpoint")
        return 0
    else:
        print(f"⚠️  Found {issues_found} issue(s). Please review the messages above.")
        return 1


def _fetch_from_upstream() -> Optional[dict]:
    """
    Fetch the latest pricing snapshot from the upstream pricing source.
    Returns the snapshot dict on success, or None on failure.
    Internal helper — not part of the public CLI surface.
    """
    from datetime import datetime, timezone

    url = (
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "traccia-cli/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_body = resp.read()
        raw = json.loads(raw_body)
    except Exception as exc:
        print(f"  Failed to reach upstream pricing source: {exc}", file=sys.stderr)
        return None

    # Normalize to Traccia's per-1K-token schema
    models: dict = {}
    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        input_cpt = entry.get("input_cost_per_token")
        output_cpt = entry.get("output_cost_per_token")
        if input_cpt is None and output_cpt is None:
            continue
        m: dict = {}
        if input_cpt is not None:
            m["prompt"] = round(float(input_cpt) * 1_000.0, 9)
        if output_cpt is not None:
            m["completion"] = round(float(output_cpt) * 1_000.0, 9)
        if entry.get("cache_creation_input_token_cost") is not None:
            m["cache_write"] = round(float(entry["cache_creation_input_token_cost"]) * 1_000.0, 9)
        if entry.get("cache_read_input_token_cost") is not None:
            m["cached_prompt"] = round(float(entry["cache_read_input_token_cost"]) * 1_000.0, 9)
        models[model_id] = m

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "upstream",
        "source_url": url,
        "models": models,
    }


def _load_dotenv_if_present() -> None:
    """Best-effort load of a .env file from the current working directory.

    Tries python-dotenv first (richer syntax support); falls back to a simple
    line-by-line parser.  Variables already set in the environment are never
    overwritten, preserving shell-level overrides.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        # Pass dotenv_path explicitly so python-dotenv looks in the process CWD,
        # not from the location of cli.py (which is inside the SDK package tree).
        _dotenv_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(dotenv_path=_dotenv_path, override=False)
        return
    except ImportError:
        pass

    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                # Strip optional inline comments and surrounding quotes
                val = val.split("#")[0].strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def _pricing_refresh(args) -> int:
    """Fetch the latest pricing snapshot and save to local cache.

    Default behaviour (no --source flag):
      1. Try the Traccia platform endpoint (normalised, authoritative).
      2. If that fails for any reason, automatically fall back to fetching
         directly from the upstream pricing source.

    --source upstream:
      Forces step 2 only — useful if you have no Traccia account or want to
      explicitly bypass the platform.
    """
    from traccia.pricing_config import write_local_cache, local_cache_info

    # Load .env from CWD so that TRACCIA_API_KEY (and internal overrides like
    # TRACCIA_API_URL) are available when the CLI is invoked from a project directory.
    _load_dotenv_if_present()

    source = getattr(args, "source", None)

    snapshot: Optional[dict] = None

    if source != "upstream":
        # Step 1: try Traccia platform.
        # Always use the canonical platform URL. TRACCIA_API_URL is an internal
        # escape hatch for local dev / self-hosted deployments only.
        api_base = os.getenv("TRACCIA_API_URL") or "https://api.traccia.ai"

        pricing_url = f"{api_base.rstrip('/')}/v1/pricing/latest"
        api_key = os.getenv("TRACCIA_API_KEY") or os.getenv("AGENT_DASHBOARD_API_KEY", "")

        existing = local_cache_info()
        etag = existing.get("etag") if existing else None

        print(f"Fetching pricing from Traccia platform ({pricing_url}) …")
        try:
            req = urllib.request.Request(pricing_url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            if etag:
                # ETag is stored unquoted in cache; HTTP requires quoted form
                quoted_etag = etag if etag.startswith('"') else f'"{etag}"'
                req.add_header("If-None-Match", quoted_etag)

            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.getcode() == 304:
                    print("Pricing is already up to date (not modified).")
                    return 0
                resp_body = resp.read()
                new_etag = resp.getheader("ETag")

            snapshot = json.loads(resp_body)
            if new_etag:
                snapshot["etag"] = new_etag.strip('"')
            print(f"  Platform responded OK.")

        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                # urllib raises HTTPError for 304; treat it as a cache hit.
                print("Pricing is already up to date (not modified).")
                return 0
            print(f"  Platform returned HTTP {exc.code}: {exc.reason}", file=sys.stderr)
            if source is None:
                print("  Falling back to upstream pricing source …")
        except Exception as exc:
            print(f"  Could not reach platform: {exc}", file=sys.stderr)
            if source is None:
                print("  Falling back to upstream pricing source …")

    if snapshot is None:
        # Step 2: fall back to (or force) upstream source
        snapshot = _fetch_from_upstream()
        if snapshot is None:
            print(
                "Failed to fetch pricing from both platform and upstream source. "
                "Your current cache (or bundled snapshot) will continue to be used.",
                file=sys.stderr,
            )
            return 1

    path = write_local_cache(snapshot)
    model_count = len(snapshot.get("models", {}))
    generated_at = snapshot.get("generated_at", "unknown")
    print(f"Pricing refreshed: {model_count} models, generated_at={generated_at}")
    print(f"Saved to: {path}")
    return 0


def _pricing_status(args) -> int:
    """Show the current state of the local pricing cache and bundled snapshot."""
    from traccia.pricing_config import local_cache_info, snapshot_age_days
    from traccia.processors.cost_engine import (
        BUNDLED_PRICING,
        BUNDLED_PRICING_GENERATED_AT,
    )

    # Bundled snapshot info
    bundled_age = snapshot_age_days(BUNDLED_PRICING_GENERATED_AT)
    bundled_age_str = f"{bundled_age:.0f}d" if bundled_age is not None else "unknown"
    print(f"Bundled snapshot  : {len(BUNDLED_PRICING)} models, generated_at={BUNDLED_PRICING_GENERATED_AT} (age: {bundled_age_str})")
    print(f"  (Bundled at SDK install time; refreshed on each SDK release.)")

    # Local cache info
    info = local_cache_info()
    if info:
        age = snapshot_age_days(info["generated_at"])
        age_str = f"{age:.0f}d" if age is not None else "unknown"
        print(f"\nLocal cache       : {info['model_count']} models, generated_at={info['generated_at']} (age: {age_str})")
        print(f"  path            : {info['path']}")
        print(f"  source          : {info['source']}")
        if info.get("source_url"):
            print(f"  source_url      : {info['source_url']}")

        if age is not None:
            if age > 30:
                print(f"  WARNING: cache is {age:.0f} days old. Run 'traccia pricing refresh' to update.")
            elif age > 7:
                print(f"  Note: cache is {age:.0f} days old. Consider running 'traccia pricing refresh'.")
    else:
        print(
            "\nLocal cache       : NOT PRESENT\n"
            "  The SDK is using the BUNDLED snapshot (shipped with this version).\n"
            "  Pricing may be out of date if the bundled snapshot is old.\n"
            "  Run 'traccia pricing refresh' to download the latest pricing and save it locally."
        )

    # Active resolution
    from traccia.pricing_config import load_pricing_with_source
    _, active_source, active_generated_at = load_pricing_with_source()
    print(f"\nActive source     : {active_source} (generated_at={active_generated_at})")
    if active_source == "bundled" and not info:
        print(
            "\nTip: Run 'traccia pricing refresh' to get the latest prices.\n"
            "     For authoritative recomputed costs, use the Traccia platform."
        )
    return 0


def _pricing_clear(args) -> int:
    """Delete the local pricing cache, reverting to the bundled snapshot."""
    from traccia.pricing_config import clear_local_cache, _cache_path

    if clear_local_cache():
        print(f"Local pricing cache cleared ({_cache_path()}).")
        print("SDK will use the bundled snapshot until the next refresh.")
    else:
        print("No local pricing cache found.")
    return 0


def main(argv=None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="traccia",
        description="Traccia SDK - Production-ready tracing for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  traccia config init              Create a new config file
  traccia doctor                   Validate configuration
  traccia check                    Test connectivity to exporter
  traccia check --endpoint URL     Test specific endpoint
  traccia pricing status           Show current pricing snapshot info
  traccia pricing refresh          Download latest pricing (platform → upstream fallback)
  traccia pricing refresh --source upstream  Fetch directly from upstream, skip platform
  traccia pricing clear            Remove local cache, revert to bundled snapshot

For more information, visit: https://github.com/traccia-ai/traccia
        """
    )
    
    # Global options
    parser.add_argument(
        "--config",
        help="Path to config file (default: ./traccia.toml or ~/.traccia/config.toml)"
    )
    
    sub = parser.add_subparsers(dest="command", required=True)

    # Check command
    check = sub.add_parser(
        "check",
        help="Verify connectivity to ingest endpoint",
        description="Test connectivity to the configured exporter endpoint"
    )
    check.add_argument("--endpoint", help="Override endpoint URL")
    check.add_argument("--api-key", help="API key for authentication")
    check.set_defaults(func=_check)

    # Config command
    config = sub.add_parser(
        "config",
        help="Configuration management",
        description="Manage Traccia configuration files"
    )
    config_sub = config.add_subparsers(dest="config_command", required=True)
    
    config_init = config_sub.add_parser(
        "init",
        help="Create traccia.toml config file",
        description="Initialize a new traccia.toml configuration file in the current directory"
    )
    config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config file"
    )
    config_init.set_defaults(func=_config_init)
    
    # Doctor command
    doctor = sub.add_parser(
        "doctor",
        help="Validate configuration and diagnose issues",
        description="Run diagnostics on your Traccia configuration"
    )
    doctor.set_defaults(func=_doctor)

    # Pricing command
    pricing = sub.add_parser(
        "pricing",
        help="Manage local pricing snapshot",
        description="Inspect and refresh the LLM pricing table used for cost estimation",
    )
    pricing_sub = pricing.add_subparsers(dest="pricing_command", required=True)

    pricing_refresh = pricing_sub.add_parser(
        "refresh",
        help="Download the latest pricing and save to local cache",
        description=(
            "Fetch the latest LLM pricing snapshot and write it to the local cache "
            "(~/.cache/traccia/pricing.json). Future processes will use this cache "
            "instead of the bundled snapshot.\n\n"
            "By default, the Traccia platform is tried first; if unavailable, the "
            "command automatically falls back to fetching from the upstream pricing source. "
            "Use --source upstream to skip the platform entirely."
        ),
    )
    pricing_refresh.add_argument(
        "--source",
        choices=["upstream"],
        default=None,
        help="Force fetching directly from the upstream pricing source, bypassing the "
             "Traccia platform. Useful when you have no Traccia account.",
    )
    pricing_refresh.set_defaults(func=_pricing_refresh)

    pricing_status_cmd = pricing_sub.add_parser(
        "status",
        help="Show current pricing source, age, and model count",
    )
    pricing_status_cmd.set_defaults(func=_pricing_status)

    pricing_clear_cmd = pricing_sub.add_parser(
        "clear",
        help="Delete local pricing cache (revert to bundled snapshot)",
    )
    pricing_clear_cmd.set_defaults(func=_pricing_clear)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
