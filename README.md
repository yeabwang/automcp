# 🚀 AutoMCP - Intelligent API to MCP Tool Converter

**Transform API specifications into intelligent Model Context Protocol (MCP) tools with AI-powered semantic enrichment.**

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



**Made with ❤️ 

[Report Bug](https://github.com/yeabwang/AutoMCP/issues) • [Request Feature](https://github.com/yeabwang/AutoMCP/issues) • [Discord Community](https://discord.gg/automcp)
