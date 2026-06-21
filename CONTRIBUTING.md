# Contributing to Traccia SDK

Thank you for your interest in contributing to Traccia SDK! We welcome contributions from the community and are grateful for your help in making Traccia better.

This document provides guidelines and instructions for contributing to the project. Please read through this guide before submitting issues or pull requests.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Improving Documentation](#improving-documentation)
  - [Contributing Code](#contributing-code)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas We'd Love Help With](#areas-wed-love-help-with)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md) that all contributors are expected to follow. Please read it before participating in our community.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please help us fix it by reporting it. Before creating a bug report:

1. **Check existing issues** - The bug might already be reported or fixed
2. **Check the documentation** - Make sure you're using the SDK correctly
3. **Reproduce the issue** - Try to reproduce it with the latest version

When reporting a bug, please include:

- **Clear title and description** - What happened and what you expected
- **Steps to reproduce** - Minimal code example that demonstrates the issue
- **Environment details**:
  - Python version
  - Traccia SDK version
  - Operating system
  - Any relevant dependencies and versions
- **Error messages and stack traces** - Full error output if applicable
- **Minimal reproduction code** - A small code snippet that reproduces the issue

Example bug report template:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run '...'
2. Call '...'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
- Python version: 3.x.x
- Traccia SDK version: x.x.x
- OS: [e.g., macOS, Linux, Windows]

**Code snippet:**
Minimal code that reproduces the issue

**Error output:**
Paste error messages here
```

### Suggesting Enhancements

We welcome suggestions for new features and improvements! When suggesting an enhancement:

1. **Check existing issues** - Your idea might already be discussed
2. **Provide context** - Explain the use case and why it would be valuable
3. **Consider alternatives** - Have you explored workarounds?
4. **Be specific** - Describe the desired behavior clearly

Enhancement suggestions should include:

- **Clear description** - What feature or improvement you'd like
- **Use case** - Why this would be useful
- **Proposed solution** - How you envision it working (if you have ideas)
- **Alternatives considered** - Other approaches you've thought about

### Improving Documentation

Documentation improvements are always welcome! This includes:

- Fixing typos and grammatical errors
- Clarifying unclear explanations
- Adding missing examples
- Improving code comments and docstrings
- Writing tutorials or guides
- Translating documentation

### Contributing Code

We welcome code contributions! Whether it's bug fixes, new features, or performance improvements, your contributions help make Traccia better for everyone.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) A virtual environment manager (venv, conda, etc.)

### Getting Started

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/traccia.git
   cd traccia
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   # Install in editable mode with development dependencies
   pip install -e ".[dev]"
   
   # Or if dev dependencies aren't configured:
   pip install -e .
   pip install pytest pytest-cov ruff mypy black
   ```

5. **Verify installation**:
   ```bash
   python -c "import traccia; print(traccia.__version__)"
   ```

## Development Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** - Write code, add tests, update documentation

3. **Run tests** to ensure everything works:
   ```bash
   pytest traccia/tests/ -v
   ```

4. **Check code style**:
   ```bash
   ruff check traccia/
   ruff format traccia/ --check
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

   Use clear, descriptive commit messages. We follow conventional commits:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Refactor:` for code refactoring
   - `Docs:` for documentation changes
   - `Test:` for test additions or changes

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Code Style and Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return types
- Write **docstrings** for all public functions, classes, and methods
- Keep functions focused and small
- Use meaningful variable and function names

### Code Formatting

We use `ruff` for linting and formatting:

```bash
# Check for issues
ruff check traccia/

# Auto-fix issues
ruff check traccia/ --fix

# Format code
ruff format traccia/
```

### Type Hints

Use type hints throughout the codebase:

```python
from typing import Optional, Dict, Any, List

def process_data(
    data: Dict[str, Any],
    options: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process data with optional configuration."""
    ...
```

### Docstrings

Follow Google-style docstrings:

```python
def init(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    **kwargs
) -> TracerProvider:
    """
    Initialize the Traccia SDK.
    
    Args:
        api_key: API key for authentication (optional)
        endpoint: OTLP endpoint URL (optional)
        **kwargs: Additional configuration options
        
    Returns:
        TracerProvider instance
        
    Raises:
        ConfigError: If configuration is invalid
        
    Example:
        >>> from traccia import init
        >>> provider = init(api_key="your-key")
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest traccia/tests/ -v

# Run specific test file
pytest traccia/tests/test_config.py -v

# Run with coverage
pytest traccia/tests/ --cov=traccia --cov-report=html

# Run tests in watch mode (if pytest-watch installed)
ptw traccia/tests/
```

### Writing Tests

- Write tests for all new features and bug fixes
- Aim for good test coverage
- Use descriptive test names that explain what they test
- Follow the Arrange-Act-Assert pattern
- Test both success and error cases

Example:

```python
def test_init_with_config_file(tmp_path):
    """Test that init() loads configuration from file."""
    # Arrange
    config_file = tmp_path / "traccia.toml"
    config_file.write_text("[tracing]\napi_key = 'test-key'")
    
    # Act
    provider = init(config_file=str(config_file))
    
    # Assert
    assert provider is not None
    # Add more assertions
```

### Test Organization

- Unit tests go in `traccia/tests/`
- Name test files with `test_` prefix
- Group related tests in classes when appropriate
- Use fixtures for common setup

## Pull Request Process

### Before Submitting

- [ ] Code follows the project's style guidelines
- [ ] All tests pass locally
- [ ] New tests are added for new functionality
- [ ] Documentation is updated (if needed)
- [ ] Commit messages are clear and descriptive
- [ ] Code is self-reviewed

### PR Checklist

When opening a pull request, please ensure:

1. **Clear title and description** - Explain what changes you made and why
2. **Reference related issues** - Link to any related issues (e.g., "Fixes #123")
3. **Update documentation** - If you changed APIs or added features
4. **Add tests** - Include tests for new functionality
5. **Keep PRs focused** - One feature or fix per PR when possible
6. **Update CHANGELOG** (if applicable) - Document user-facing changes

### Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Be patient - reviews may take time depending on maintainer availability
- Feel free to ask questions if feedback is unclear

### After Approval

Once your PR is approved and merged:
- Your changes will be included in the next release
- Thank you for contributing! 🎉

## Areas We'd Love Help With

We especially welcome contributions in these areas:

### Integrations

- **LLM Providers**: Add support for more providers (Cohere, AI21, Hugging Face, local models, etc.)
- **Frameworks**: Instrumentation for LangChain, LlamaIndex, AutoGen, and other agent frameworks
- **HTTP Libraries**: Support for additional HTTP clients (httpx, aiohttp, etc.)

### Backend Compatibility

- **OTLP Backends**: Test and document setup with different OTLP-compatible backends
- **Exporters**: New exporter implementations (Kafka, S3, etc.)
- **Protocol Support**: Additional transport protocols

### Documentation

- **Tutorials**: Step-by-step guides for common use cases
- **Examples**: Real-world examples of agent instrumentation
- **API Documentation**: Improve inline documentation and examples
- **Video Guides**: Create video walkthroughs (if you're comfortable with that)

### Performance

- **Optimization**: Profile and optimize hot paths
- **Benchmarking**: Add performance benchmarks
- **Memory Usage**: Reduce memory footprint where possible

### Testing

- **Coverage**: Improve test coverage
- **Integration Tests**: Add end-to-end integration tests
- **Performance Tests**: Add performance regression tests
- **Compatibility Tests**: Test across Python versions and platforms

### Developer Experience

- **CLI Improvements**: Enhance the CLI tooling
- **Debugging Tools**: Better debugging and diagnostic tools
- **Error Messages**: Improve error messages and troubleshooting guides

## Questions?

If you have questions about contributing:

- Open a [GitHub Discussion](https://github.com/traccia-ai/traccia/discussions) for general questions
- Open an [Issue](https://github.com/traccia-ai/traccia/issues) for bug reports or feature requests
- Check existing documentation and issues first

## License

By contributing to Traccia SDK, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).

---

Thank you for contributing to Traccia SDK! Your efforts help make observability for AI agents better for everyone. 🙏
