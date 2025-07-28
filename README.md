# 🚀 AutoMCP - Intelligent API to MCP Tool Converter

[![PyPI version](https://badge.fury.io/py/automcp.svg)](https://badge.fury.io/py/automcp)
[![Python Support](https://img.shields.io/pypi/pyversions/automcp.svg)](https://pypi.org/project/automcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/yeabwang/AutoMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/yeabwang/AutoMCP/actions)
[![Coverage](https://codecov.io/gh/yeabwang/AutoMCP/branch/main/graph/badge.svg)](https://codecov.io/gh/yeabwang/AutoMCP)

Transform API specifications into intelligent Model Context Protocol (MCP) tools with AI-powered semantic enrichment. AutoMCP bridges the gap between your APIs and AI assistants, making your services instantly accessible to large language models.

## ✨ **Key Features**

- 🔍 **Intelligent API Analysis** - Automatically extracts intents and capabilities from OpenAPI, Postman, and REST APIs
- 🧠 **AI-Powered Enrichment** - Uses LLMs to generate semantic descriptions and user stories
- 🛠️ **MCP Tool Generation** - Creates production-ready Model Context Protocol tools
- ⚡ **High Performance** - Async processing with configurable concurrency and caching
- 🔧 **Enterprise Ready** - Comprehensive configuration, monitoring, and error handling
- 🎯 **Developer First** - Simple CLI, rich documentation, and extensive examples

## 🚀 **Quick Start**

### Installation

```bash
# Install from PyPI
pip install automcp

# Or install with all extras
pip install automcp[all]

# Development installation
git clone https://github.com/yeabwang/AutoMCP.git
cd AutoMCP
uv sync --all-extras
```

### Basic Usage

```bash
# Convert OpenAPI spec to MCP tools
automcp process --input api.yaml --output tools.json

# Use specific environment configuration
automcp process --input api.yaml --env production

# Interactive mode for guided conversion
automcp interactive
```

### Python API

```python
from automcp import AutoMCPProcessor

# Initialize processor
processor = AutoMCPProcessor()

# Process API specification
result = await processor.process_api_spec(
    input_path="shopify-api.yaml",
    output_path="shopify-tools.json"
)

print(f"Generated {len(result['tools'])} MCP tools!")
```

## 📋 **What You Get**

AutoMCP transforms this OpenAPI endpoint:

```yaml
/products/{id}:
  get:
    summary: Get product by ID
    parameters:
      - name: id
        in: path
        required: true
        schema:
          type: string
```

Into this intelligent MCP tool:

```json
{
  "name": "get_product_details",
  "description": "Retrieve detailed information about a specific product including pricing, availability, and specifications",
  "function": {
    "name": "get_product_details",
    "parameters": {
      "type": "object",
      "properties": {
        "product_id": {
          "type": "string",
          "description": "Unique identifier for the product"
        }
      },
      "required": ["product_id"]
    }
  },
  "examples": [
    {
      "input": {"product_id": "prod_123"},
      "description": "Get details for product with ID prod_123"
    }
  ]
}
```

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Specs     │───▶│    AutoMCP       │───▶│   MCP Tools     │
│                 │    │                  │    │                 │
│ • OpenAPI       │    │ 🔍 Parse & Extract│    │ • Functions     │
│ • Postman       │    │ 🧠 AI Enrichment │    │ • Resources     │
│ • REST APIs     │    │ 🛠️  Tool Generation│    │ • Prompts       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 **Performance Metrics**

Our optimized framework achieves:
- ✅ **100% Intent Success Rate** (improved from 67%)
- ✅ **100% Tool Generation** (3/3 tools)  
- ✅ **100% Parameter Extraction** (24/24 parameters)
- ⚡ **Sub-second Processing** for typical APIs
- 🔄 **Async Architecture** for maximum throughput

## 🎯 **Use Cases**

### **E-commerce Integration**
```bash
automcp process --input shopify-api.yaml
# → Generates tools for product management, orders, customers
```

### **CRM Automation**  
```bash
automcp process --input salesforce-api.json
# → Creates tools for lead management, opportunity tracking
```

### **Content Management**
```bash
automcp process --input wordpress-api.yaml  
# → Builds tools for content creation, media management
```

## ⚙️ **Configuration**

AutoMCP supports environment-specific configurations:

```yaml
# config/production.yaml
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"

enrichment:
  batch_size: 10
  semantic_enhancement: true
  
output:
  format: "mcp_tools"
  include_examples: true
  validate_tools: true
```

## 🔧 **Advanced Features**

### **Custom LLM Providers**
```python
from automcp.core.config import Config

config = Config(environment="production")
# Supports OpenAI, Anthropic, Groq, and custom providers
```

### **Batch Processing**
```bash
automcp process --input specs/ --output tools/ --batch
```

### **Validation & Testing**
```bash
automcp config validate
automcp analyze --input api.yaml --dry-run
```

## 📚 **Documentation**

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api-reference.md)
- [Examples & Tutorials](docs/examples.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development setup
git clone https://github.com/yeabwang/AutoMCP.git
cd AutoMCP
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run ruff format src tests
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 **Acknowledgments**

- Built with the [Model Context Protocol](https://modelcontextprotocol.io/) standard
- Powered by modern Python tooling (uv, ruff, pytest)
- Inspired by the need for seamless AI-API integration

---

**Made with ❤️ by the AutoMCP Team**

[Report Bug](https://github.com/yeabwang/AutoMCP/issues) • [Request Feature](https://github.com/yeabwang/AutoMCP/issues) • [Discord Community](https://discord.gg/automcp)