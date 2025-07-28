# Contributing to AutoMCP

We're thrilled that you're interested in contributing to AutoMCP! This guide will help you get started with contributing to our intelligent API to MCP tool converter.

## 🚀 Quick Start

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/AutoMCP.git
   cd AutoMCP
   ```

2. **Install Dependencies**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync --all-extras
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   uv run pre-commit install
   ```

4. **Verify Installation**
   ```bash
   uv run pytest
   uv run automcp --help
   ```

## 🎯 How to Contribute

### 🐛 Reporting Bugs

1. **Check Existing Issues**: Search [existing issues](https://github.com/yeabwang/AutoMCP/issues) first
2. **Use Bug Report Template**: Create a new issue using our bug report template
3. **Provide Details**: Include API specifications, configuration, and error logs
4. **Minimal Reproduction**: Provide the smallest possible example that reproduces the issue

### ✨ Suggesting Features

1. **Check Roadmap**: Review our [project roadmap](https://github.com/yeabwang/AutoMCP/projects)
2. **Use Feature Request Template**: Create a new issue using our feature request template
3. **Explain Use Case**: Describe the problem you're trying to solve
4. **Provide Examples**: Show how the feature would be used

### 🔧 Code Contributions

#### Areas We Need Help With

- **🔍 Parsers**: Support for new API specification formats
- **🧠 Enrichers**: Enhanced semantic analysis and AI prompts
- **🛠️ Generators**: New output formats and MCP tool types
- **🎯 Integrations**: Support for new LLM providers and APIs
- **📚 Documentation**: Tutorials, examples, and API documentation
- **🧪 Testing**: Test coverage and integration tests

#### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Changes**
   - Follow our coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   uv run pytest
   
   # Run specific tests
   uv run pytest tests/test_parsers.py
   
   # Check test coverage
   uv run pytest --cov=src/automcp --cov-report=html
   ```

4. **Check Code Quality**
   ```bash
   # Format code
   uv run ruff format src tests
   
   # Lint code
   uv run ruff check src tests
   
   # Type checking
   uv run mypy src
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add support for GraphQL specifications"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use our PR template
   - Link related issues
   - Provide clear description of changes

## 📋 Coding Standards

### Code Style

We use modern Python best practices:

- **Formatting**: `ruff format` (replaces black)
- **Linting**: `ruff check` (replaces flake8, isort, etc.)
- **Type Hints**: Required for all public APIs
- **Documentation**: Docstrings for all public functions and classes

### Code Organization

```python
"""
Module docstring explaining the purpose.

This module provides functionality for...
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import yaml
from pydantic import BaseModel

# Local imports
from automcp.core.config import Config
from automcp.models.intent import Intent


class YourClass:
    """Class docstring with examples.
    
    Args:
        config: Configuration instance
        
    Example:
        ```python
        processor = YourClass(config)
        result = await processor.process()
        ```
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    async def your_method(self, input_data: Dict[str, Any]) -> List[Intent]:
        """Method docstring.
        
        Args:
            input_data: Input data description
            
        Returns:
            List of processed intents
            
        Raises:
            ValueError: When input_data is invalid
        """
        # Implementation here
        pass
```

### Testing Standards

- **Coverage**: Aim for >90% test coverage
- **Test Types**: Unit tests, integration tests, and end-to-end tests
- **Fixtures**: Use pytest fixtures for common test data
- **Async Testing**: Use `pytest-asyncio` for async code

```python
import pytest
from automcp import AutoMCPProcessor
from automcp.core.config import Config


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration."""
    return Config(environment="test")


@pytest.mark.asyncio
async def test_process_api_spec(sample_config, tmp_path):
    """Test API specification processing."""
    processor = AutoMCPProcessor(config=sample_config)
    
    # Test implementation
    result = await processor.process_api_spec("test.yaml")
    
    assert result["status"] == "success"
    assert len(result["tools"]) > 0
```

## 🏗️ Architecture Guide

### Core Components

```
src/automcp/
├── core/                 # Core framework components
│   ├── config.py        # Configuration management
│   ├── parsers.py       # API specification parsers
│   ├── enricher.py      # AI-powered enrichment
│   └── output_generator.py  # MCP tool generation
├── models/              # Pydantic data models
│   ├── intent.py        # Intent models
│   ├── capability.py    # Capability models
│   └── mcp_tool.py      # MCP tool models
├── utils/               # Utility functions
│   ├── logging.py       # Structured logging
│   └── validation.py    # Validation utilities
└── cli.py               # Command-line interface
```

### Adding New Features

#### Adding a New Parser

1. **Create Parser Class**
   ```python
   from automcp.core.parsers import ParserInterface
   
   class YourParser(ParserInterface):
       async def parse(self, spec_path: str) -> Dict[str, Any]:
           # Implementation
           pass
   ```

2. **Register Parser**
   ```python
   # In parsers.py
   self.parsers["your_format"] = YourParser(self.config)
   ```

3. **Add Tests**
   ```python
   def test_your_parser():
       # Test implementation
       pass
   ```

#### Adding LLM Provider Support

1. **Extend LLM Client**
   ```python
   # In async_llm_client.py
   async def _call_your_provider(self, prompt: str, **kwargs) -> str:
       # Implementation
       pass
   ```

2. **Update Configuration**
   ```yaml
   # In config files
   llm:
     provider: "your_provider"
     your_provider_settings:
       api_key: "${YOUR_API_KEY}"
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/automcp --cov-report=html

# Run specific test categories
uv run pytest -m unit        # Unit tests only
uv run pytest -m integration # Integration tests only
uv run pytest -m slow        # Slow tests (CI only)

# Run tests in parallel
uv run pytest -n auto
```

### Test Categories

- **Unit Tests**: Fast tests for individual functions/classes
- **Integration Tests**: Tests that combine multiple components
- **End-to-End Tests**: Full workflow tests with real API specs
- **Performance Tests**: Benchmarks and performance regression tests

## 📚 Documentation

### Writing Documentation

- **Docstrings**: All public APIs must have comprehensive docstrings
- **Examples**: Include usage examples in docstrings
- **Type Hints**: Use type hints for better IDE support
- **README Updates**: Update README for significant features

### Building Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Build documentation
uv run mkdocs serve
```

## 🎉 Recognition

### Contributors

All contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for their contributions
- GitHub contributors page

### Contribution Types

We recognize all types of contributions:
- 💻 Code contributions
- 📖 Documentation improvements
- 🐛 Bug reports and testing
- 💡 Feature suggestions and design
- 🎨 Design and UX improvements
- 🔧 Infrastructure and tooling
- 💬 Community management

## 📞 Getting Help

- **Discord**: Join our [Discord community](https://discord.gg/automcp)
- **GitHub Discussions**: Use [GitHub Discussions](https://github.com/yeabwang/AutoMCP/discussions) for questions
- **Issues**: Create issues for bugs and feature requests
- **Email**: Contact the maintainers at [maintainers@automcp.dev](mailto:maintainers@automcp.dev)

## 📄 License

By contributing to AutoMCP, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AutoMCP! Your contributions help make AI-API integration accessible to everyone. 🚀
