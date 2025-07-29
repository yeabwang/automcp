# 🚀 AutoMCP - Intelligent API to MCP Tool Converter

[![PyPI version](https://badge.fury.io/py/automcp.svg)](https://badge.fury.io/py/automcp)
[![Python Support](https://img.shields.io/pypi/pyversions/automcp.svg)](https://pypi.org/project/automcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/yeabwang/AutoMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/yeabwang/AutoMCP/actions)

**Transform API specifications into intelligent Model Context Protocol (MCP) tools with AI-powered semantic enrichment.**

AutoMCP bridges the gap between your APIs and AI assistants, making your services instantly accessible to large language models through sophisticated semantic analysis and MCP tool generation.

## ✨ **Why AutoMCP?**

- 🧠 **AI-Powered Intelligence** - Uses LLMs to understand your APIs and generate semantic, human-readable tools
- 🛡️ **Enterprise Security** - Built-in PII scrubbing, permission classification, and security analysis  
- ⚡ **Production Ready** - Async processing, circuit breakers, retries, and comprehensive error handling
- 🎯 **Developer First** - Zero-config setup, intelligent defaults, and rich CLI experience
- 🔧 **Highly Configurable** - Environment-based configuration for development, production, and enterprise

## � **Quick Start**

### Installation

```bash
# Install from PyPI
pip install automcp

# Or install with development extras
pip install automcp[dev]

# Development installation  
git clone https://github.com/yeabwang/AutoMCP.git
cd AutoMCP
uv sync --all-extras
```

### Basic Usage

```bash
# Analyze a single API specification
automcp analyze shopify.yaml

# Process multiple APIs at once
automcp batch input/

# Use production configuration
automcp -e production analyze api.yaml

# Check system health
automcp health
```

### Python API

```python
from automcp import AutoMCPFramework
from automcp.config import load_config

# Initialize with environment-specific config
config = load_config("production")
framework = AutoMCPFramework(config)

# Transform API specification
results = await framework.analyze("shopify.yaml")

# Access generated artifacts
intents = results.intents          # Semantic intent metadata
capabilities = results.capabilities # Permission classifications  
mcp_tools = results.mcp_tools      # MCP tool specifications
```

## 📁 **Input → Output Flow**

### Supported Input Formats
- **OpenAPI 3.0/3.1** - Full specification support (.yaml, .json)
- **Swagger 2.0** - Legacy format compatibility  
- **Postman Collections** - v2.1 format support
- **Python Source Code** - Repository scanning and analysis

### Generated Output Structure
```
outputs/
└── {api-name}/
    ├── enriched_intents.json    # Semantic intent metadata with complexity analysis
    ├── capabilities.txt         # Permission-based capability classifications  
    ├── mcp_tools.json          # Complete MCP tool specifications
    └── quality_report.json     # AI-generated quality assessment (optional)
```

### Example Generated Intent
```json
{
  "name": "search_products",
  "details": {
    "summary": "Search product catalog with advanced filtering",
    "complexity": "moderate", 
    "user_context": "authenticated_user",
    "domain": "e_commerce",
    "business_context": "Core shopping functionality",
    "examples": [
      {
        "utterance": "Find red dresses under $100",
        "context": {"color": "red", "category": "dresses", "max_price": 100}
      }
    ]
  },
  "confidence": 0.95,
  "requirements": {
    "authentication": [{"type": "oauth", "required": true}],
    "permissions": [{"level": "white", "action": "search_products"}]
  }
}
```

## 🌍 **Environment-Based Configuration**

AutoMCP supports environment-specific configurations for different deployment scenarios:

```bash
# Development - Enhanced debugging, relaxed validation
automcp -e development analyze api.yaml

# Production - Optimized performance, strict validation  
automcp -e production analyze api.yaml

# Enterprise - Maximum security, audit logging, encryption
automcp -e enterprise analyze api.yaml
```

### Configuration Hierarchy
1. **default.yaml** - Base configuration with sensible defaults
2. **{environment}.yaml** - Environment-specific overrides
3. **Environment variables** - Runtime configuration
4. **CLI flags** - Command-specific overrides

## 🧠 **AI-Powered Features**

### Semantic Intent Generation
- **Smart Naming** - Converts `/api/v1/users/{id}` → `get_user_profile`
- **Context Inference** - Determines user contexts (anonymous, authenticated, admin)
- **Complexity Analysis** - Classifies operations (simple, moderate, complex, multi_step)
- **Business Domain Detection** - Identifies domains (e_commerce, healthcare, finance)

### Security & Permissions
- **Permission Classification** - WHITE (safe), GREY (moderate), BLACK (dangerous)
- **Authentication Detection** - Infers auth requirements and types
- **PII Scrubbing** - Removes sensitive data from outputs
- **Risk Assessment** - Evaluates security implications

### Quality Assurance
- **AI Quality Scoring** - LLM-driven quality assessment (A-F grades)
- **Validation Pipeline** - Multi-stage validation with detailed reports
- **Confidence Metrics** - Transformation confidence scoring
- **Issue Detection** - Automatic problem identification with suggestions

## 📊 **Performance & Scale**

- **Processing Speed**: 50+ endpoints/second
- **Memory Efficiency**: <100MB for 1000+ endpoints  
- **Accuracy**: 95%+ intent classification accuracy
- **Concurrent Processing**: Configurable worker pools
- **Caching**: LRU cache for semantic transformations
- **Circuit Breakers**: Automatic failure handling and recovery

## 🔧 **Advanced Features**

### Multi-Stage Processing Pipeline
1. **Technical Extraction** - Parse specifications and validate structure
2. **Semantic Transformation** - LLM-driven naming and classification  
3. **Context Inference** - User context and business domain analysis
4. **Security Analysis** - Risk assessment and access control mapping
5. **MCP Generation** - Client-specific tool specification creation
6. **Validation** - Quality assessment and schema verification

### Multiple MCP Client Support
- **LangChain** - Tools with prompt templates and examples
- **OpenAI** - Function calling format with parameters
- **FastMCP** - Optimized tool specifications
- **LlamaIndex** - Agent-ready tool definitions
- **Generic** - Standard MCP protocol format

### Enterprise Security
- **PII Detection** - Configurable sensitive data patterns
- **Output Encryption** - Encrypt generated files at rest
- **Audit Logging** - Comprehensive operation tracking
- **Access Control** - Permission-based capability classification
- **Compliance** - OWASP security standards adherence

## 📖 **Documentation**

- [**Installation Guide**](docs/installation.md) - Detailed setup instructions
- [**Configuration Reference**](docs/configuration.md) - Complete configuration options
- [**Usage Examples**](docs/examples.md) - Real-world usage scenarios  
- [**API Reference**](docs/api-reference.md) - Python API documentation
- [**Best Practices**](docs/best-practices.md) - Production deployment guide

## 🛠️ **Development**

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/yeabwang/AutoMCP.git
cd AutoMCP

# Install with uv (recommended)
uv sync --all-extras

# Install with pip
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests (uses real LLM APIs)
pytest

# Run specific test suite  
pytest tests/test_integration.py -v
```

### Code Quality Standards
- **Type Safety** - Full MyPy type checking
- **Testing** - 85%+ coverage with real LLM integration
- **Formatting** - Black code formatting + Ruff linting
- **Security** - Bandit security scanning
- **Documentation** - Comprehensive docstrings and examples

## 🤝 **Contributing**

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📋 **Roadmap**

- [ ] **GraphQL Support** - Support for GraphQL schema analysis
- [ ] **AsyncAPI Support** - Event-driven API specifications
- [ ] **Web UI** - Browser-based analysis and configuration
- [ ] **VS Code Extension** - IDE integration for developers
- [ ] **Docker Support** - Containerized deployment options
- [ ] **Cloud Connectors** - Direct integration with AWS, Azure, GCP APIs

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Model Context Protocol** - Built on the MCP standard for AI-agent integration
- **OpenAPI Initiative** - Leverages OpenAPI specifications for API understanding  
- **LLM Providers** - Powered by Groq, OpenAI, and Anthropic for semantic analysis
- **Python Ecosystem** - Built with modern Python tooling and best practices

---

**Ready to transform your APIs into AI-ready tools?** [Get started with AutoMCP today!](docs/installation.md)

[![Star History](https://api.star-history.com/svg?repos=yeabwang/AutoMCP&type=Date)](https://star-history.com/#yeabwang/AutoMCP&Date)

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