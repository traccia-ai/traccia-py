# Traccia

**Production-ready distributed tracing for AI agents and LLM applications**

Traccia is a lightweight, high-performance Python SDK for observability and tracing of AI agents, LLM applications, and complex distributed systems. Built on OpenTelemetry standards with specialized instrumentation for AI workloads.

## ✨ Features

- **🔍 Automatic Instrumentation**: Auto-patch OpenAI, Anthropic, requests, and HTTP libraries
- **🤖 Framework Integrations**: Support for LangChain and OpenAI Agents SDK
- **📊 LLM-Aware Tracing**: Track tokens, costs, prompts, and completions automatically
- **⚡ Zero-Config Start**: Simple `init()` call with automatic config discovery
- **🎯 Decorator-Based**: Trace any function with `@observe` decorator
- **🔧 Multiple Exporters**: OTLP (compatible with Grafana Tempo, Jaeger, Zipkin), Console, or File
- **🛡️ Production-Ready**: Rate limiting, error handling, config validation, robust flushing
- **📝 Type-Safe**: Full Pydantic validation for configuration
- **🚀 High Performance**: Efficient batching, async support, minimal overhead
- **🔐 Secure**: No secrets in logs, configurable data truncation

---

## 🚀 Quick Start

### Installation

```bash
pip install traccia
```

### Basic Usage

```python
from traccia import init, observe

# Initialize (auto-loads from traccia.toml if present)
init()

# Trace any function
@observe()
def my_function(x, y):
    return x + y

# That's it! Traces are automatically created and exported
result = my_function(2, 3)
```

### With LLM Calls

```python
from traccia import init, observe
from openai import OpenAI

init()  # Auto-patches OpenAI

client = OpenAI()

@observe(as_type="llm")
def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Automatically tracks: model, tokens, cost, prompt, completion, latency
text = generate_text("Write a haiku about Python")
```

### LangChain

Create a callback handler and pass it to `config={"callbacks": [traccia_handler]}`. Install the optional extra: `pip install traccia[langchain]`.

```python
from traccia import init
from traccia.integrations.langchain import CallbackHandler  # or TracciaCallbackHandler
from langchain_openai import ChatOpenAI

init()

# Create Traccia handler (no args)
traccia_handler = CallbackHandler()

# Use with any LangChain runnable
llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke(
    "Tell me a joke",
    config={"callbacks": [traccia_handler]}
)
```

Spans for LLM/chat model runs are created automatically with the same attributes as direct OpenAI instrumentation (model, prompt, usage, cost).

**Note:** `pip install traccia[langchain]` installs traccia plus `langchain-core`; you need this extra to use the callback handler. If you already have `langchain-core` (e.g. from `langchain` or `langchain-openai`), base `pip install traccia` may be enough at runtime, but `traccia[langchain]` is the supported way to get a compatible dependency.

### OpenAI Agents SDK

Traccia **automatically** detects and instruments the OpenAI Agents SDK when installed. No extra code needed:

```python
from traccia import init
from agents import Agent, Runner

init()  # Automatically enables Agents SDK tracing

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)
result = Runner.run_sync(agent, "Write a haiku about recursion")
```

**Configuration**: Auto-enabled by default when `openai-agents` is installed. To disable:

```python
init(openai_agents=False)  # Explicit parameter
# OR set environment variable: TRACCIA_OPENAI_AGENTS=false
# OR in traccia.toml under [instrumentation]: openai_agents = false
```

**Compatibility**: If you have `openai-agents` installed but don't use it (e.g., using LangChain or pure OpenAI instead), the integration is registered but never invoked—no overhead or extra spans.

---

## 📖 Configuration

### Configuration File

Create a `traccia.toml` file in your project root:

```bash
traccia config init
```

This creates a template config file:

```toml
[tracing]
# API key (optional - for future Traccia UI, not needed for OTLP backends)
api_key = ""

# Endpoint URL for OTLP trace ingestion
# Works with Grafana Tempo, Jaeger, Zipkin, and other OTLP-compatible backends
endpoint = "http://localhost:4318/v1/traces"

sample_rate = 1.0           # 0.0 to 1.0
auto_start_trace = true     # Auto-start root trace on init
auto_trace_name = "root"    # Name for auto-started trace
use_otlp = true             # Use OTLP exporter
# service_name = "my-app"   # Optional service name

[exporters]
# Only enable ONE exporter at a time
enable_console = false        # Print traces to console
enable_file = false           # Write traces to file
file_exporter_path = "traces.jsonl"
reset_trace_file = false      # Reset file on initialization

[instrumentation]
enable_patching = true          # Auto-patch libraries (OpenAI, Anthropic, requests)
enable_token_counting = true    # Count tokens for LLM calls
enable_costs = true             # Calculate costs
openai_agents = true            # Auto-enable OpenAI Agents SDK integration
auto_instrument_tools = false   # Auto-instrument tool calls (experimental)
max_tool_spans = 100            # Max tool spans to create
max_span_depth = 10             # Max nested span depth

[rate_limiting]
# Optional: limit spans per second
# max_spans_per_second = 100.0
max_queue_size = 5000           # Max buffered spans
max_block_ms = 100              # Max ms to block before dropping
max_export_batch_size = 512     # Spans per export batch
schedule_delay_millis = 5000    # Delay between batches

[runtime]
# Optional runtime metadata
# session_id = ""
# user_id = ""
# tenant_id = ""
# project_id = ""

[logging]
debug = false                   # Enable debug logging
enable_span_logging = false     # Enable span-level logging

[advanced]
# attr_truncation_limit = 1000  # Max attribute value length
```

### OTLP Backend Compatibility

Traccia is fully OTLP-compatible and works with:
- **Grafana Tempo** - `http://tempo:4318/v1/traces`
- **Jaeger** - `http://jaeger:4318/v1/traces`
- **Zipkin** - Configure via OTLP endpoint
- **SigNoz** - Self-hosted observability platform
- **Traccia Cloud** - Coming soon (will require API key)

### Environment Variables

All config parameters can be set via environment variables with the `TRACCIA_` prefix:

**Tracing**: `TRACCIA_API_KEY`, `TRACCIA_ENDPOINT`, `TRACCIA_SAMPLE_RATE`, `TRACCIA_AUTO_START_TRACE`, `TRACCIA_AUTO_TRACE_NAME`, `TRACCIA_USE_OTLP`, `TRACCIA_SERVICE_NAME`

**Exporters**: `TRACCIA_ENABLE_CONSOLE`, `TRACCIA_ENABLE_FILE`, `TRACCIA_FILE_PATH`, `TRACCIA_RESET_TRACE_FILE`

**Instrumentation**: `TRACCIA_ENABLE_PATCHING`, `TRACCIA_ENABLE_TOKEN_COUNTING`, `TRACCIA_ENABLE_COSTS`, `TRACCIA_AUTO_INSTRUMENT_TOOLS`, `TRACCIA_MAX_TOOL_SPANS`, `TRACCIA_MAX_SPAN_DEPTH`

**Rate Limiting**: `TRACCIA_MAX_SPANS_PER_SECOND`, `TRACCIA_MAX_QUEUE_SIZE`, `TRACCIA_MAX_BLOCK_MS`, `TRACCIA_MAX_EXPORT_BATCH_SIZE`, `TRACCIA_SCHEDULE_DELAY_MILLIS`

**Runtime**: `TRACCIA_SESSION_ID`, `TRACCIA_USER_ID`, `TRACCIA_TENANT_ID`, `TRACCIA_PROJECT_ID`

**Logging**: `TRACCIA_DEBUG`, `TRACCIA_ENABLE_SPAN_LOGGING`

**Advanced**: `TRACCIA_ATTR_TRUNCATION_LIMIT`

**Priority**: Explicit parameters > Environment variables > Config file > Defaults

### Programmatic Configuration

```python
from traccia import init

# Override config programmatically
init(
    endpoint="http://tempo:4318/v1/traces",
    sample_rate=0.5,
    enable_costs=True,
    max_spans_per_second=100.0
)
```

---

## 🎯 Usage Guide

### The `@observe` Decorator

The `@observe` decorator is the primary way to instrument your code:

```python
from traccia import observe

# Basic usage
@observe()
def process_data(data):
    return transform(data)

# Custom span name
@observe(name="data_pipeline")
def process_data(data):
    return transform(data)

# Add custom attributes
@observe(attributes={"version": "2.0", "env": "prod"})
def process_data(data):
    return transform(data)

# Specify span type
@observe(as_type="llm")  # "span", "llm", "tool"
def call_llm():
    pass

# Skip capturing specific arguments
@observe(skip_args=["password", "secret"])
def authenticate(username, password):
    pass

# Skip capturing result (for large returns)
@observe(skip_result=True)
def fetch_large_dataset():
    return huge_data
```

**Available Parameters**:
- `name` (str, optional): Custom span name (defaults to function name)
- `attributes` (dict, optional): Initial span attributes
- `as_type` (str): Span type - `"span"`, `"llm"`, or `"tool"`
- `skip_args` (list, optional): List of argument names to skip capturing
- `skip_result` (bool): Skip capturing the return value

### Async Functions

`@observe` works seamlessly with async functions:

```python
@observe()
async def async_task(x):
    await asyncio.sleep(1)
    return x * 2

result = await async_task(5)
```

### Manual Span Creation

For more control, create spans manually:

```python
from traccia import get_tracer, span

# Using convenience function
with span("operation_name") as s:
    s.set_attribute("key", "value")
    s.add_event("checkpoint_reached")
    do_work()

# Using tracer directly
tracer = get_tracer("my_service")
with tracer.start_as_current_span("operation") as s:
    s.set_attribute("user_id", 123)
    do_work()
```

### Error Handling

Traccia automatically captures and records errors:

```python
@observe()
def failing_function():
    raise ValueError("Something went wrong")

# Span will contain:
# - error.type: "ValueError"
# - error.message: "Something went wrong"
# - error.stack_trace: (truncated stack trace)
# - span status: ERROR
```

### Nested Spans

Spans are automatically nested based on call hierarchy:

```python
@observe()
def parent_operation():
    child_operation()
    return "done"

@observe()
def child_operation():
    grandchild_operation()

@observe()
def grandchild_operation():
    pass

# Creates nested span hierarchy:
# parent_operation
#   └── child_operation
#       └── grandchild_operation
```

---

## 🛠️ CLI Tools

Traccia includes a powerful CLI for configuration and diagnostics:

### `traccia config init`

Create a new `traccia.toml` configuration file:

```bash
traccia config init
traccia config init --force  # Overwrite existing
```

### `traccia doctor`

Validate configuration and diagnose issues:

```bash
traccia doctor

# Output:
# 🩺 Running Traccia configuration diagnostics...
# 
# ✅ Found config file: ./traccia.toml
# ✅ Configuration is valid
# 
# 📊 Configuration summary:
#    • API Key: ❌ Not set (optional)
#    • Endpoint: http://localhost:4318/v1/traces
#    • Sample Rate: 1.0
#    • OTLP Exporter: ✅ Enabled
```

### `traccia check`

Test connectivity to your exporter endpoint:

```bash
traccia check
traccia check --endpoint http://tempo:4318/v1/traces
```

---

## 🎨 Advanced Features

### Rate Limiting

Protect your infrastructure with built-in rate limiting:

```toml
[rate_limiting]
max_spans_per_second = 100.0  # Limit to 100 spans/sec
max_queue_size = 5000         # Max buffered spans
max_block_ms = 100             # Block up to 100ms before dropping
```

**Behavior**:
1. Try to acquire capacity immediately
2. If unavailable, block for up to `max_block_ms`
3. If still unavailable, drop span and log warning

When spans are dropped due to rate limiting, warnings are logged to help you monitor and adjust limits.

### Sampling

Control trace volume with sampling:

```python
# Sample 10% of traces
init(sample_rate=0.1)

# Sampling is applied at trace creation time
# Traces are either fully included or fully excluded
```

### Token Counting & Cost Calculation

Automatic for supported LLM providers (OpenAI, Anthropic):

```python
@observe(as_type="llm")
def call_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Span automatically includes:
# - llm.token.prompt_tokens
# - llm.token.completion_tokens
# - llm.token.total_tokens
# - llm.cost.total (in USD)
```

---

## 🔧 Troubleshooting

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via config
init(debug=True)

# Or via env var
# TRACCIA_DEBUG=1 python your_script.py
```

### Common Issues

#### **Traces not appearing**

1. Check connectivity: `traccia check`
2. Validate config: `traccia doctor`
3. Enable debug logging
4. Verify endpoint is correct and accessible

#### **High memory usage**

- Reduce `max_queue_size` in rate limiting config
- Lower `sample_rate` to reduce volume
- Enable rate limiting with `max_spans_per_second`

#### **Spans being dropped**

- Check rate limiter logs for warnings
- Increase `max_spans_per_second` if set
- Increase `max_queue_size` if spans are queued
- Check `traccia doctor` output

---

## 📚 API Reference

### Core Functions

#### `init(**kwargs) -> TracerProvider`

Initialize the Traccia SDK.

**Parameters**:
- `endpoint` (str, optional): OTLP endpoint URL
- `api_key` (str, optional): API key (optional, for future Traccia UI)
- `sample_rate` (float, optional): Sampling rate (0.0-1.0)
- `auto_start_trace` (bool, optional): Auto-start root trace
- `config_file` (str, optional): Path to config file
- `use_otlp` (bool, optional): Use OTLP exporter
- `enable_console` (bool, optional): Enable console exporter
- `enable_file` (bool, optional): Enable file exporter
- `enable_patching` (bool, optional): Auto-patch libraries
- `enable_token_counting` (bool, optional): Count tokens
- `enable_costs` (bool, optional): Calculate costs
- `max_spans_per_second` (float, optional): Rate limit
- `**kwargs`: Any other config parameter

**Returns**: TracerProvider instance

#### `stop_tracing(flush_timeout: float = 1.0) -> None`

Stop tracing and flush pending spans.

**Parameters**:
- `flush_timeout` (float): Max seconds to wait for flush

#### `get_tracer(name: str = "default") -> Tracer`

Get a tracer instance.

**Parameters**:
- `name` (str): Tracer name (typically module/service name)

**Returns**: Tracer instance

#### `span(name: str, attributes: dict = None) -> Span`

Create a span context manager.

**Parameters**:
- `name` (str): Span name
- `attributes` (dict, optional): Initial attributes

**Returns**: Span context manager

### Decorator

#### `@observe(name=None, *, attributes=None, tags=None, as_type="span", skip_args=None, skip_result=False)`

Decorate a function to create spans automatically.

**Parameters**:
- `name` (str, optional): Span name (default: function name)
- `attributes` (dict, optional): Initial attributes
- `tags` (list[str], optional): User-defined identifiers for the observed method
- `as_type` (str): Span type (`"span"`, `"llm"`, `"tool"`)
- `skip_args` (list, optional): Arguments to skip capturing
- `skip_result` (bool): Skip capturing return value

### Configuration

#### `load_config(config_file=None, overrides=None) -> TracciaConfig`

Load and validate configuration.

**Parameters**:
- `config_file` (str, optional): Path to config file
- `overrides` (dict, optional): Override values

**Returns**: Validated TracciaConfig instance

**Raises**: `ConfigError` if invalid

#### `validate_config(config_file=None, overrides=None) -> tuple[bool, str, TracciaConfig | None]`

Validate configuration without loading.

**Returns**: Tuple of (is_valid, message, config_or_none)

---

## 🏗️ Architecture

### Data Flow

```
Application Code (@observe)
        ↓
   Span Creation
        ↓
   Processors (token counting, cost, enrichment)
        ↓
   Rate Limiter (optional)
        ↓
   Batch Processor (buffering)
        ↓
   Exporter (OTLP/Console/File)
        ↓
   Backend (Grafana Tempo / Jaeger / Zipkin / etc.)
```

### Instrumentation vs Integrations

- **`traccia.instrumentation.*`**: Infrastructure and vendor instrumentation.
  - HTTP client/server helpers (including FastAPI middleware).
  - Vendor SDK hooks and monkey patching (e.g., OpenAI, Anthropic, `requests`).
  - Decorators and utilities used for auto-instrumenting arbitrary functions.

- **`traccia.integrations.*`**: AI/agent framework integrations.
  - Adapters that plug into higher-level frameworks via their official extension points (e.g., LangChain callbacks).
  - Work at the level of chains, tools, agents, and workflows rather than raw HTTP or SDK calls.

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new features, documentation improvements, or examples - we appreciate your help.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest traccia/tests/`
5. **Lint your code**: `ruff check traccia/`
6. **Commit**: `git commit -m "Add amazing feature"`
7. **Push**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/traccia-ai/traccia.git
cd traccia

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest traccia/tests/ -v

# Run with coverage
pytest traccia/tests/ --cov=traccia --cov-report=html
```

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings for public APIs
- Write tests for new features
- Keep PRs focused and atomic

### Areas We'd Love Help With

- **Integrations**: Add support for more LLM providers (Cohere, AI21, local models)
- **Backends**: Test and document setup with different OTLP backends
- **Examples**: Real-world examples of agent instrumentation
- **Documentation**: Tutorials, guides, video walkthroughs
- **Performance**: Optimize hot paths, reduce overhead
- **Testing**: Improve test coverage, add integration tests

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for full terms and conditions.

---

## 🙏 Acknowledgments

Built with:
- [OpenTelemetry](https://opentelemetry.io/) - Vendor-neutral observability framework
- [Pydantic](https://pydantic.dev/) - Data validation
- [tiktoken](https://github.com/openai/tiktoken) - Token counting

Inspired by observability tools in the ecosystem and designed to work seamlessly with the OTLP standard.

---

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/traccia-ai/traccia/issues) - Report bugs or request features
- **Discussions**: [GitHub Discussions](https://github.com/traccia-ai/traccia/discussions) - Ask questions, share ideas

---

**Made with ❤️ for the AI agent community**
