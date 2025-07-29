# 📦 AutoMCP Installation Guide

This guide will help you install and set up AutoMCP for development, production, or enterprise use.

## 🚀 Quick Installation

### Using pip (Recommended)

```bash
# Latest stable release
pip install automcp

# With development extras
pip install automcp[dev]

# With all optional dependencies
pip install automcp[all]
```

### Using uv (Fastest)

```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install AutoMCP with uv
uv pip install automcp
```

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/yeabwang/AutoMCP.git
cd AutoMCP

# Install with uv (recommended for development)
uv sync --all-extras

# Or install with pip
pip install -e ".[dev,docs,test]"
```

## ⚙️ **System Requirements**

### Python Version
- **Python 3.9+** (3.11+ recommended for best performance)
- **Operating System**: Windows, macOS, Linux

### Required Dependencies
AutoMCP automatically installs these core dependencies:
- `pydantic>=2.0.0` - Data validation and serialization
- `PyYAML>=6.0` - YAML processing
- `aiohttp>=3.8.0` - Async HTTP client
- `openai>=1.0.0` - OpenAI API client
- `groq>=0.4.0` - Groq API client
- `anthropic>=0.25.0` - Anthropic API client
- `structlog>=23.0.0` - Structured logging

### Optional Dependencies
Additional features require these optional dependencies:

```bash
# For development
pip install automcp[dev]
# Includes: pytest, black, ruff, mypy, pre-commit

# For documentation
pip install automcp[docs]  
# Includes: mkdocs, mkdocs-material, mkdocstrings

# For testing
pip install automcp[test]
# Includes: pytest-asyncio, pytest-cov, pytest-mock

# All extras
pip install automcp[all]
```

## 🔑 **API Key Configuration**

AutoMCP requires an LLM API key for semantic analysis. Choose your preferred provider:

### Option 1: Groq (Recommended - Fast & Free)

1. **Get API Key**: Visit [Groq Console](https://console.groq.com/keys)
2. **Set Environment Variable**:
   ```bash
   # Linux/macOS
   export GROQ_API_KEY="your-groq-api-key-here"
   
   # Windows PowerShell
   $env:GROQ_API_KEY="your-groq-api-key-here"
   
   # Windows Command Prompt
   set GROQ_API_KEY=your-groq-api-key-here
   ```

### Option 2: OpenAI

1. **Get API Key**: Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. **Set Environment Variable**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

### Option 3: Anthropic

1. **Get API Key**: Visit [Anthropic Console](https://console.anthropic.com/)
2. **Set Environment Variable**:
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   ```

### Generic Configuration

You can also use the generic `LLM_API_KEY` environment variable:

```bash
export LLM_API_KEY="your-api-key-here"
```

## 📁 **Directory Setup**

AutoMCP works best with a standard directory structure:

```bash
# Create project directory
mkdir my-automcp-project
cd my-automcp-project

# Create standard directories
mkdir -p {input,output,config}

# Directory structure
my-automcp-project/
├── input/          # Place your API specs here
├── output/         # Generated MCP tools will appear here  
└── config/         # Custom configuration files (optional)
```

## ✅ **Verify Installation**

Run the health check to verify everything is working:

```bash
# Basic health check
automcp health

# Check specific environment
automcp -e production health

# Show configuration
automcp config show
```

Expected output:
```
🏥 AutoMCP System Health Check
==================================================
✅ Python dependencies: All available
✅ Config directory: Found
✅ Configuration loaded: development
✅ Inputs directory: Found
✅ Output directory: Found (outputs)
✅ LLM API key: Configured
✅ LLM Provider: groq
✅ File permissions: Write access OK

📊 Health Summary:
🎉 All systems healthy!
```

## 🛠️ **Development Setup**

For contributors and advanced users who want to modify AutoMCP:

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yeabwang/AutoMCP.git
cd AutoMCP

# Install with development dependencies
uv sync --all-extras

# Or with pip
pip install -e ".[dev,docs,test]"
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 3. Run Tests

```bash
# Run all tests (requires API keys)
pytest

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest --cov=automcp --cov-report=html
```

### 4. Code Quality Checks

```bash
# Type checking
mypy src/

# Linting and formatting
ruff check src/
ruff format src/

# Security scanning
bandit -r src/
```

## 🌍 **Environment Configuration**

AutoMCP supports multiple environments with different optimizations:

### Development Environment
```bash
# Uses development.yaml configuration
automcp -e development analyze api.yaml

# Features:
# - Enhanced debugging output
# - Relaxed validation rules  
# - Smaller batch sizes for testing
# - Detailed error messages
```

### Production Environment
```bash
# Uses production.yaml configuration
automcp -e production analyze api.yaml

# Features:
# - Optimized performance settings
# - Strict validation rules
# - Larger batch sizes
# - Error handling focused on reliability
```

### Enterprise Environment
```bash
# Uses enterprise.yaml configuration  
automcp -e enterprise analyze api.yaml

# Features:
# - Maximum security settings
# - Output encryption enabled
# - Audit logging enabled
# - Compliance-focused validation
```

## 🐛 **Troubleshooting**

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'automcp'
# Solution: Reinstall with correct Python environment
pip uninstall automcp
pip install automcp
```

#### 2. API Key Issues
```bash
# Error: LLM API key not configured
# Solution: Set environment variable
export GROQ_API_KEY="your-key-here"

# Verify with:
automcp config show
```

#### 3. Permission Errors
```bash
# Error: Permission denied writing to outputs/
# Solution: Check directory permissions
chmod 755 outputs/
# Or run with different output directory:
automcp analyze api.yaml -o /tmp/automcp-output/
```

#### 4. Memory Issues
```bash
# Error: Out of memory during processing
# Solution: Adjust batch size in config
automcp -e development analyze api.yaml  # Smaller batches
```

### Getting Help

1. **Check Health**: `automcp health`
2. **Validate Config**: `automcp config validate`
3. **View Logs**: Enable verbose mode with `-v` flag
4. **GitHub Issues**: [Report issues](https://github.com/yeabwang/AutoMCP/issues)
5. **Documentation**: [Full documentation](https://automcp.readthedocs.io/)

## 📝 **Next Steps**

After installation, you're ready to:

1. **🎯 [Quick Start Tutorial](usage.md)** - Your first API transformation
2. **⚙️ [Configuration Guide](configuration.md)** - Customize for your needs  
3. **📚 [Examples](examples.md)** - Real-world usage scenarios
4. **🚀 [Production Guide](best-practices.md)** - Deploy to production

## 💡 **Pro Tips**

- **Use uv**: Faster dependency resolution and installation
- **Set API keys globally**: Add to your shell profile for persistence
- **Use environments**: Match configuration to your deployment stage
- **Enable caching**: Set `performance.caching.enabled: true` for repeated runs
- **Monitor usage**: Check API usage limits with your LLM provider

---

**Having issues?** Check our [troubleshooting guide](troubleshooting.md) or [open an issue](https://github.com/yeabwang/AutoMCP/issues).
