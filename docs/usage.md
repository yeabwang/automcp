# 🎯 AutoMCP Usage Guide

This guide covers common usage patterns and provides real-world examples of transforming API specifications into MCP tools.

## 🚀 Basic Usage

### Analyze a Single API Specification

```bash
# Basic analysis with default settings
automcp analyze shopify.yaml

# Specify custom output directory
automcp analyze api.yaml -o custom_output/

# Use production configuration
automcp -e production analyze api.yaml

# Preview without generating files
automcp analyze api.yaml --dry-run
```

### Batch Process Multiple APIs

```bash
# Process all specs in input directory
automcp batch input/

# Process with custom pattern
automcp batch apis/ -p "*.yaml"

# Use multiple concurrent workers
automcp batch specs/ -c 8

# Continue processing if individual files fail
automcp batch input/ --continue-on-error
```

## 📂 Typical Workflow

### 1. Prepare Your API Specifications

Create an input directory and add your API specifications:

```bash
mkdir input
# Add your API specification files
cp shopify-api.yaml input/
cp stripe-api.json input/
cp internal-api.yaml input/
```

### 2. Run Analysis

```bash
# Analyze all APIs
automcp batch input/

# Or analyze individually
automcp analyze input/shopify-api.yaml
automcp analyze input/stripe-api.json
```

### 3. Review Generated Outputs

```
outputs/
├── shopify-api/
│   ├── enriched_intents.json     # 🧠 Semantic intent metadata
│   ├── capabilities.txt          # 🔐 Permission classifications
│   ├── mcp_tools.json           # 🛠️ MCP tool specifications
│   └── quality_report.json      # 📊 Quality assessment
├── stripe-api/
│   ├── enriched_intents.json
│   ├── capabilities.txt
│   ├── mcp_tools.json
│   └── quality_report.json
└── internal-api/
    ├── enriched_intents.json
    ├── capabilities.txt
    ├── mcp_tools.json
    └── quality_report.json
```

## 📋 Real-World Examples

### Example 1: E-commerce API (Shopify)

**Input**: Shopify Products API specification

```yaml
# shopify-products.yaml
openapi: 3.0.0
info:
  title: Shopify Products API
  version: 1.0.0
paths:
  /admin/api/2023-01/products.json:
    get:
      summary: Retrieve products
      parameters:
        - name: limit
          in: query
          type: integer
        - name: since_id
          in: query
          type: integer
```

**Command**:
```bash
automcp analyze shopify-products.yaml -e production
```

**Generated Intent**:
```json
{
  "name": "list_products",
  "details": {
    "summary": "Retrieve a list of products from the store catalog",
    "complexity": "simple",
    "user_context": "authenticated_user",
    "domain": "e_commerce",
    "business_context": "Core inventory management functionality",
    "description": "Allows authenticated users to retrieve products with pagination support",
    "examples": [
      {
        "utterance": "Get the first 50 products",
        "context": {"limit": 50}
      },
      {
        "utterance": "Get products added after product ID 12345",
        "context": {"since_id": 12345}
      }
    ]
  },
  "confidence": 0.95,
  "paths": [
    {
      "method": "GET",
      "endpoint": "/admin/api/2023-01/products.json",
      "parameters": [
        {
          "name": "limit",
          "description": "Number of products to retrieve",
          "required": false,
          "type": "integer"
        }
      ]
    }
  ],
  "requirements": {
    "authentication": [
      {
        "type": "api_key",
        "required": true,
        "description": "Shopify API key required for admin access"
      }
    ],
    "permissions": [
      {
        "level": "white",
        "action": "list_products",
        "description": "Safe read operation for product data"
      }
    ]
  }
}
```

**Generated Capability Classification**:
```
WHITE-LISTED:
list_products, get_product_details, search_products

GREY-LISTED:
create_product, update_product, update_inventory

BLACK-LISTED:
delete_product, bulk_delete_products, admin_override
```

**Generated MCP Tool**:
```json
{
  "name": "shopify_list_products",
  "description": "Retrieve a list of products from the Shopify store catalog with pagination support",
  "method": "GET",
  "path": "/admin/api/2023-01/products.json",
  "safety_level": "safe",
  "input_schema": {
    "type": "object",
    "properties": {
      "limit": {
        "type": "integer",
        "description": "Number of products to retrieve (max 250)",
        "minimum": 1,
        "maximum": 250,
        "default": 50
      },
      "since_id": {
        "type": "integer", 
        "description": "Retrieve products after this ID for pagination"
      }
    }
  },
  "metadata": {
    "complexity": "simple",
    "confidence": 0.95,
    "parameter_count": 2,
    "domain": "e_commerce"
  }
}
```

### Example 2: Payment API (Stripe)

**Input**: Stripe Charges API

```bash
automcp analyze stripe-charges.yaml -e enterprise
```

**Key Features**:
- Detects financial domain context
- Classifies payment operations as high-risk (BLACK-LISTED)
- Generates enterprise security requirements
- Includes compliance-focused validation

### Example 3: Healthcare API

**Input**: Healthcare Patient API

```bash
automcp -e enterprise analyze patient-api.yaml
```

**Enterprise Features**:
- HIPAA compliance considerations
- Enhanced PII scrubbing for medical data
- Audit logging for all operations
- Encrypted output files

## ⚙️ Environment-Specific Usage

### Development Environment

Perfect for API exploration and testing:

```bash
# Development mode - relaxed validation, verbose output
automcp -e development analyze api.yaml -v

# Features enabled:
# ✅ Enhanced debugging output
# ✅ Relaxed validation rules
# ✅ Detailed error messages
# ✅ Smaller batch sizes for testing
```

### Production Environment

Optimized for performance and reliability:

```bash
# Production mode - optimized performance, strict validation
automcp -e production batch input/ -c 16

# Features enabled:
# ✅ Optimized batch sizes (16 concurrent)
# ✅ Strict validation rules
# ✅ Performance-focused error handling
# ✅ Comprehensive quality assessment
```

### Enterprise Environment

Maximum security and compliance:

```bash
# Enterprise mode - security focused, audit logging
automcp -e enterprise analyze sensitive-api.yaml

# Features enabled:
# ✅ Enhanced PII scrubbing
# ✅ Output encryption
# ✅ Audit logging
# ✅ Compliance validation
# ✅ Advanced security analysis
```

## 🔧 Advanced Usage Patterns

### Custom Configuration

Create custom configuration for specific use cases:

```bash
# Create custom config
cp config/production.yaml config/my-custom.yaml

# Edit my-custom.yaml with your settings
# Then use it:
automcp -e my-custom analyze api.yaml
```

### Programmatic Usage

Use AutoMCP in Python scripts:

```python
import asyncio
from automcp import AutoMCPFramework
from automcp.config import load_config

async def analyze_api():
    # Load production configuration
    config = load_config("production")
    
    # Initialize framework
    framework = AutoMCPFramework(config)
    
    # Analyze API specification
    results = await framework.analyze("shopify.yaml")
    
    # Access results
    print(f"Generated {len(results.intents)} intents")
    print(f"Generated {len(results.mcp_tools)} MCP tools")
    print(f"Quality grade: {results.quality_grade}")
    
    return results

# Run analysis
results = asyncio.run(analyze_api())
```

### Integration with CI/CD

Add AutoMCP to your CI/CD pipeline:

```yaml
# .github/workflows/api-analysis.yml
name: API Analysis
on:
  push:
    paths: ['api-specs/**']

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install AutoMCP
        run: pip install automcp
        
      - name: Analyze APIs
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          automcp batch api-specs/ -e production
          
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: mcp-tools
          path: outputs/
```

## 📊 Quality Assessment

AutoMCP provides detailed quality assessment for generated outputs:

### Understanding Quality Grades

- **A (90-100%)** - Excellent quality, ready for production
- **B (80-89%)** - Good quality, minor improvements possible  
- **C (70-79%)** - Acceptable quality, some issues to address
- **D (60-69%)** - Below average, significant improvements needed
- **F (<60%)** - Poor quality, major issues require attention

### Quality Report Example

```json
{
  "overall": {
    "grade": "A",
    "score": 92,
    "assessment": "High-quality semantic transformation with clear intent mapping"
  },
  "intents": {
    "total": 12,
    "high_confidence": 10,
    "medium_confidence": 2,
    "low_confidence": 0,
    "issues": []
  },
  "mcp_tools": {
    "total": 12,
    "valid": 12,
    "invalid": 0,
    "safety_distribution": {
      "safe": 8,
      "moderate": 3,
      "dangerous": 1
    }
  },
  "recommendations": [
    "Consider adding more detailed parameter descriptions",
    "Review authentication requirements for admin endpoints"
  ]
}
```

## 🎯 Best Practices

### 1. **Use Appropriate Environments**
- `development` for exploration and testing
- `production` for regular use
- `enterprise` for sensitive/regulated APIs

### 2. **Organize Your Inputs**
```bash
input/
├── public-apis/        # Public API specifications
├── internal-apis/      # Internal API specifications  
├── third-party/        # Partner/vendor APIs
└── archive/           # Old versions
```

### 3. **Review Generated Outputs**
- Check quality reports for issues
- Validate permission classifications
- Review security assessments
- Test generated MCP tools

### 4. **Version Control Integration**
```bash
# Add to .gitignore
outputs/
*.log

# But track quality reports
!outputs/*/quality_report.json
```

### 5. **Monitor API Usage**
- Track LLM API usage costs
- Use caching for repeated analyses
- Consider batch processing for efficiency

## 🚨 Common Issues and Solutions

### Issue: Low Quality Scores

**Problem**: Generated intents have low confidence scores

**Solutions**:
```bash
# Use more detailed API specifications
# Add better descriptions and examples to your OpenAPI spec
# Try different LLM providers:
automcp -e production analyze api.yaml  # Uses optimized settings
```

### Issue: Permission Misclassification

**Problem**: Safe operations classified as dangerous

**Solution**: Review and customize permission patterns in configuration

### Issue: API Rate Limits

**Problem**: LLM API rate limit exceeded

**Solutions**:
```bash
# Use development environment (slower, smaller batches)
automcp -e development batch input/

# Or adjust rate limits in config
# Edit config/production.yaml:
# llm_client:
#   provider_settings:
#     groq:
#       rate_limit_requests_per_minute: 30
```

## 📝 Next Steps

- **[Configuration Guide](configuration.md)** - Customize AutoMCP for your needs
- **[API Reference](api-reference.md)** - Detailed API documentation  
- **[Best Practices](best-practices.md)** - Production deployment guide
- **[Examples Repository](https://github.com/yeabwang/AutoMCP-Examples)** - More real-world examples

---

**Need help?** Check our [troubleshooting guide](troubleshooting.md) or [open an issue](https://github.com/yeabwang/AutoMCP/issues)
