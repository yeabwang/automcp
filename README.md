# MCP Tool Generation Framework

The **MCP Tool Generation Framework** is an enterprise-grade system designed to transform raw API specifications (e.g., OpenAPI, Swagger, Postman collections) into semantically rich, AI-consumable metadata using the **Model Context Protocol (MCP)** standard.

This framework automates the end-to-end process of **parsing**, **enriching**, and **generating structured output files**—empowering AI agents to safely and intelligently interact with APIs.

---

## 🚀 Core Transformation Workflow

### 🔹 Input
- Raw API specifications:
  - OpenAPI (e.g., `shopify_products.yaml`)
  - Swagger
  - Postman collections

### 🔹 Output
Generates **AI-ready metadata** in the following structured files:

- `intents.json`  
  > Defines high-level user intents (e.g., `product_search`, `create_order`) enriched with:
  - Summaries
  - Complexity levels
  - User context
  - Authentication requirements
  - Permissions
  - Confidence scores

- `capabilities.txt`  
  > Classifies operations based on **risk & access** levels:
  - `white`: Safe / public
  - `grey`: Authenticated / cautious
  - `black`: Restricted / high-risk

- `mcp_tools.json`  
  > Provides detailed tool definitions for LLM orchestration, including:
  - Input/output schemas
  - Descriptions
  - Prompt templates
  - Usage examples
  - Security headers

---

## 🧠 Pipeline Architecture

1. **Parser**
   - Extracts raw elements from API specifications:
     - Endpoints
     - HTTP methods
     - Parameters
     - Request/response schemas

2. **Enricher**
   - Uses LLMs to:
     - Infer semantic context
     - Estimate task complexity
     - Map user context and roles
     - Determine authentication & permission levels

3. **Output Generator**
   - Validates and generates production-grade artifacts:
     - Ensures schema correctness
     - Applies version control
     - Annotates with rich metadata

---

## ✅ Key Principles

- **Semantic intelligence** for better LLM orchestration
- **Security-aware generation** for safe tool exposure
- **Scalability** for large and diverse API ecosystems
- **Standards-compliant** with OpenAPI, Swagger, and Postman

---

## 📁 Example Files




mkdir spec-analyzer-mcp
cd spec-analyzer-mcp


uv init
python -m venv .venv

source .venv/Scripts/activate

uv add -r requirements.txt

# Production environment configuration
export LLM_API_KEY="your-production-api-key"
export CONFIG_PATH="/etc/mcp-analyzer/config.yaml"
export LOG_LEVEL="info"
export ENVIRONMENT="production"

# Security hardening
export SECURITY_KEY="your-encryption-key"
export RATE_LIMIT_ENABLED="true"

# production-config.yaml
profile: "production"
enrichment:
  batch_size: 16
  max_batch_size: 32
  retry_attempts: 5
llm_client:
  timeout: 45
  provider: "groq"  # or your preferred provider
monitoring:
  metrics_enabled: true
  health_check_port: 8080
security:
  encrypt_outputs: true
  rate_limit_by_key: true
  input_sanitization: true


# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["./scripts/entrypoint.sh"]


# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-analyzer
  template:
    metadata:
      labels:
        app: mcp-analyzer
    spec:
      containers:
      - name: mcp-analyzer
        image: mcp-analyzer:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-credentials
              key: api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10

# Optimize batch sizes by provider
PROVIDER_OPTIMIZATIONS = {
    "groq": {"optimal_batch_size": 12, "max_tokens": 4096},
    "openai": {"optimal_batch_size": 10, "max_tokens": 4096},
    "anthropic": {"optimal_batch_size": 6, "max_tokens": 4096}
}

# Implement request caching
@lru_cache(maxsize=1000)
def cached_semantic_naming(method: str, path: str, summary: str) -> str:
    return generate_semantic_name(method, path, summary)   

# Stream large OpenAPI specs
def parse_large_spec_streaming(spec_path: str):
    """Parse large specs without loading entire spec into memory."""
    # Implementation with streaming JSON parser
    pass

# Batch processing with memory management
def process_with_memory_management(intents: List[Dict], batch_size: int = 50):
    """Process intents in batches to manage memory usage."""
    for i in range(0, len(intents), batch_size):
        batch = intents[i:i + batch_size]
        yield process_batch(batch)
        # Force garbage collection between batches
        gc.collect()

# Optimize concurrent processing
async def process_intents_concurrent(intents: List[Dict]) -> List[Dict]:
    """Process intents with optimal concurrency."""
    semaphore = asyncio.Semaphore(10)  # Limit concurrent LLM calls
    
    async def process_single(intent: Dict) -> Dict:
        async with semaphore:
            return await transform_intent(intent)
    
    tasks = [process_single(intent) for intent in intents]
    return await asyncio.gather(*tasks)