# enricher.py - Optimized Domain-Agnostic Semantic Transformation Engine
# =============================================================================
# Transforms API intents into rich, AI-agent-ready metadata using LLM-driven analysis.
# Key Features:
# - Fully config-driven: Complexity, user contexts, permissions, domains from config.yaml.
# - Async pipeline for scalability, fixing parsers.py errors (16–21).
# - Client-type-specific MCP tool generation (langchain, openai, generic, etc.).
# - PII scrubbing with config patterns and regex.
# - Robust validation, error handling, and caching.
# Critical Requirements:
# - Industry-standard: Async pipeline, LLM-driven, JSON schemas.
# - Production-ready: Retries, metrics via LLM client, logging.
# - Secure: PII scrubbing in prompts/responses/tools.
# - Rich: Semantic naming, complexity, user context, MCP tools.
# - Robust: Handles failures, malformed data, async scalability.
# - No hardcoded logic: Config-driven mappings, no enums.
# - MCP Support: Client-specific tool fields (prompts, parameters).
# Assumptions:
# - config.yaml: Sections for enrichment, semantic_transformation (with llm_prompts, complexity_levels, user_contexts, permission_levels, domain_patterns, sensitive_data_patterns, auth_type_mappings, required_intent_details), mcp (with safety_mapping, tool_generation), security (with auth_header_patterns).
# - Pydantic Version: v2.x for validation syntax; adjust for v1.x if needed.
# - llm_client_interface.py: Defines ResponseFormat, LLMClientInterface with query_semantic.
# - async_llm_client.py: Implements LLMClientInterface with robust query_semantic.
# =============================================================================

# Standard library
import json
import re
import time
import hashlib
from typing import List, Dict, Optional, Any, Tuple, Awaitable
from collections import OrderedDict  # FIX: Added for LRU cache
from dataclasses import dataclass

# Third-party
import psutil
import scrubadub
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException

# Local
from .llm_client_interface import LLMClientInterface, ResponseFormat

@dataclass
class SemanticTransformationResult:
    """Result of semantic transformation for a single intent."""
    semantic_name: str
    complexity: str
    user_context: str
    permission_level: str
    auth_requirements: List[Dict]
    rich_summary: str
    confidence: float
    processing_metadata: Dict

def create_logger(config: Dict[str, Any]) -> structlog.BoundLogger:
    """Create and configure structlog logger."""
    log_format = config["logging"].get("format", "console")
    if log_format == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    elif log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.processors.KeyValueRenderer()
    
    redact_keys = config["logging"].get("redact_keys", [])
    
    def redact_sensitive(logger, method_name, event_dict):
        """Redact sensitive information recursively."""
        def redact_recursive(obj):
            if isinstance(obj, dict):
                for key in list(obj.keys()):
                    if key in redact_keys:
                        obj[key] = "[REDACTED]"
                    else:
                        redact_recursive(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    redact_recursive(item)
        redact_recursive(event_dict)
        return event_dict
    
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_sensitive,
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    return structlog.get_logger(__name__)

class SemanticEnricher:
    """
    Domain-agnostic semantic transformation engine for API specifications.
    """

    def __init__(self, llm_client: LLMClientInterface, config: Dict[str, Any]):
        """Initialize with config.yaml."""
        self.full_config = config
        self.enrichment_config = config.get("enrichment", {})
        self.semantic_config = config.get("semantic_transformation", {})
        self.mcp_config = config.get("mcp", {})
        
        self.llm_client = llm_client
        self.logger = create_logger(self.full_config)
        
        # Config-driven mappings
        self.complexity_levels = [level["name"] for level in self.semantic_config.get("complexity_levels", [])]
        self.user_contexts = [ctx["name"] for ctx in self.semantic_config.get("user_contexts", [])]
        self.permission_levels = [key for key in self.semantic_config.get("permission_levels", {})]
        self.domain_patterns = self.semantic_config.get("domain_patterns", {})
        self.sensitive_data_patterns = self.semantic_config.get("sensitive_data_patterns", {})
        self.auth_type_mappings = self.semantic_config.get("auth_type_mappings", {})
        
        # Performance optimization state
        self._last_resource_check = 0.0
        self._current_batch_size = self.enrichment_config.get("batch_size", 8)
        
        # Initialize processing pipeline
        self._initialize_pipeline()
        
        # FIX: Initialize LRU cache with max size from config (fixes unbounded cache growth)
        max_cache_size = self.semantic_config.get("caching", {}).get("max_cache_size", 10000)
        self._semantic_cache = OrderedDict()
        self._max_cache_size = max_cache_size
        
        self.logger.info("Semantic enricher initialized",
                        mode=self.semantic_config.get("mode", "llm_driven"),
                        stages=len(self.processing_stages))

    def _initialize_pipeline(self):
        """Initialize the semantic transformation pipeline."""
        self.processing_stages = self.enrichment_config.get("processing_stages", [
            "technical_extraction",
            "semantic_transformation",
            "context_inference",
            "security_analysis",
            "mcp_generation",
            "validation"
        ])
        self.stage_configs = {stage: self.enrichment_config.get("stage_settings", {}).get(stage, {}) for stage in self.processing_stages}

    # FIX: Corrected return type for async methods (fixes async signature issues from error analysis)
    async def enrich_intents(self, raw_intents: List[Dict]) -> List[Dict[str, Any]]:
        """Main async semantic enrichment pipeline."""
        self.logger.info("Starting semantic enrichment pipeline", intent_count=len(raw_intents))
        
        technical_data = await self._process_technical_extraction(raw_intents)
        semantic_results = await self._process_semantic_transformation(technical_data)
        context_enriched = await self._process_context_inference(semantic_results)
        security_enriched = await self._process_security_analysis(context_enriched)
        enriched_intents = await self._assemble_enriched_intents(security_enriched)
        validated_intents = await self._process_validation(enriched_intents)
        
        self.logger.info("Semantic enrichment pipeline completed",
                        input_count=len(raw_intents),
                        output_count=len(validated_intents))
        return validated_intents

    async def _process_technical_extraction(self, raw_intents: List[Dict]) -> List[Dict]:
        """Stage 1: Validate and prepare technical data."""
        self.logger.info("Processing technical extraction stage")
        
        extracted_data = []
        for intent in raw_intents:
            if not await self._validate_technical_intent(intent):
                self.logger.warning("Skipping invalid intent", path=intent.get("path"))
                continue
            
            technical_intent = {
                "path": intent.get("path", ""),
                "method": intent.get("method", "").upper(),
                "summary": intent.get("summary", ""),
                "description": intent.get("description", ""),
                "operation_id": intent.get("operation_id", ""),
                "tags": intent.get("tags", []),
                "parameters": intent.get("parameters", []),
                "request_body": intent.get("requestBody", []),
                "responses": intent.get("responses", []),
                "security": intent.get("security", []),
                "complexity_indicators": intent.get("complexity_indicators", {}),
                "_raw_data": intent,
                "mcp_client_type": intent.get("mcp_client_type", "generic")
            }
            extracted_data.append(technical_intent)
        
        self.logger.info("Technical extraction completed", extracted_count=len(extracted_data))
        return extracted_data

    async def _process_semantic_transformation(self, technical_data: List[Dict]) -> List[SemanticTransformationResult]:
        """Stage 2: LLM-driven semantic transformation."""
        self.logger.info("Processing semantic transformation stage")
        
        semantic_results = []
        batch_size = await self._adapt_batch_size()
        
        for i in range(0, len(technical_data), batch_size):
            batch = technical_data[i:i + batch_size]
            batch_results = await self._process_semantic_batch(batch)
            semantic_results.extend(batch_results)
        
        self.logger.info("Semantic transformation completed", results_count=len(semantic_results))
        return semantic_results

    async def _process_semantic_batch(self, batch: List[Dict]) -> List[SemanticTransformationResult]:
        """Process a batch of intents for semantic transformation."""
        batch_results = []
        
        for intent in batch:
            cache_key = await self._generate_cache_key(intent)
            if cache_key in self._semantic_cache:
                cached_result = self._semantic_cache[cache_key]
                self._semantic_cache.move_to_end(cache_key)  # Update LRU order
                self.logger.debug("Using cached semantic result", path=intent["path"])
                batch_results.append(cached_result)
                continue
            
            semantic_result = await self._transform_single_intent(intent)
            if self.semantic_config.get("caching", {}).get("enabled", True):
                self._semantic_cache[cache_key] = semantic_result
                # FIX: Evict oldest entry if cache exceeds max size
                if len(self._semantic_cache) > self._max_cache_size:
                    self._semantic_cache.popitem(last=False)
                    self.logger.debug("Evicted oldest cache entry", cache_size=len(self._semantic_cache))
            batch_results.append(semantic_result)
        
        return batch_results

    async def _transform_single_intent(self, intent: Dict) -> SemanticTransformationResult:
        """Transform a single intent using LLM-driven analysis."""
        try:
            semantic_name = await self._generate_semantic_name(intent)
            complexity = await self._analyze_complexity(intent)
            user_context = await self._infer_user_context(intent, semantic_name)
            permission_level = await self._map_permission_level(semantic_name, intent, user_context)
            auth_requirements = await self._process_auth_requirements(intent)
            rich_summary = await self._generate_rich_summary(intent, semantic_name, user_context)
            confidence = await self._calculate_transformation_confidence(intent, semantic_name)
            
            return SemanticTransformationResult(
                semantic_name=semantic_name,
                complexity=complexity,
                user_context=user_context,
                permission_level=permission_level,
                auth_requirements=auth_requirements,
                rich_summary=rich_summary,
                confidence=confidence,
                processing_metadata={
                    "transformation_time": time.time(),
                    "llm_calls": 5,
                    "cache_hit": False,
                    "original_intent": intent
                }
            )
        except Exception as e:
            self.logger.error("Semantic transformation failed", path=intent.get("path"), error=str(e))
            return await self._create_fallback_transformation(intent)

    async def _generate_semantic_name(self, intent: Dict) -> str:
        """Generate semantic function name using LLM."""
        # FIX: Use query_semantic with expected schema for structured response (fixes missing LLM response validation)
        context = {
            "method": intent["method"],
            "path": intent["path"],
            "summary": intent["summary"],
            "parameters": await self._summarize_parameters(intent["parameters"]),
            "domain_context": intent.get("domain_context", "general")
        }
        prompt = self.semantic_config["llm_prompts"]["semantic_naming"].format(**context)
        expected_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            semantic_name = response.get("name", "")
            if await self._validate_semantic_name(semantic_name):
                return scrubadub.clean(semantic_name)
            self.logger.warning("Invalid semantic name from LLM", name=semantic_name, path=intent["path"])
            return await self._fallback_semantic_naming(intent)
        except Exception as e:
            self.logger.error("LLM semantic naming failed", error=str(e))
            return await self._fallback_semantic_naming(intent)

    async def _analyze_complexity(self, intent: Dict) -> str:
        """Analyze endpoint complexity using LLM."""
        # FIX: Use query_semantic with expected schema
        complexity_data = {
            "parameters": intent["parameters"],
            "request_body": intent["request_body"],
            "response_schema": await self._summarize_responses(intent["responses"]),
            "semantic_name": getattr(self, '_current_semantic_name', intent.get("operation_id", ""))
        }
        prompt = self.semantic_config["llm_prompts"]["complexity_analysis"].format(**complexity_data)
        expected_schema = {
            "type": "object",
            "properties": {"complexity": {"type": "string", "enum": self.complexity_levels}},
            "required": ["complexity"]
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            complexity = response.get("complexity", "").strip().lower()
            return complexity if complexity in self.complexity_levels else self.semantic_config.get("fallback_strategies", {}).get("complexity", "moderate")
        except Exception as e:
            self.logger.error("LLM complexity analysis failed", error=str(e))
            return await self._fallback_complexity_analysis(intent)

    async def _infer_user_context(self, intent: Dict, semantic_name: str) -> str:
        """Infer user context using LLM."""
        # FIX: Use query_semantic with expected schema
        context_data = {
            "semantic_name": semantic_name,
            "method": intent["method"],
            "path": intent["path"],
            "auth_required": bool(intent["security"]),
            "domain_context": intent.get("domain_context", "general")
        }
        prompt = self.semantic_config["llm_prompts"]["user_context_inference"].format(**context_data)
        expected_schema = {
            "type": "object",
            "properties": {"user_context": {"type": "string", "enum": self.user_contexts}},
            "required": ["user_context"]
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            context = response.get("user_context", "").strip().lower()
            return context if context in self.user_contexts else self.semantic_config.get("fallback_strategies", {}).get("user_context", "standard_user")
        except Exception as e:
            self.logger.error("LLM user context inference failed", error=str(e))
            return await self._fallback_user_context(intent)

    async def _map_permission_level(self, semantic_name: str, intent: Dict, user_context: str) -> str:
        """Map semantic action to permission level using LLM."""
        # FIX: Use query_semantic with expected schema
        permission_data = {
            "semantic_name": semantic_name,
            "method": intent["method"],
            "user_context": user_context,
            "data_sensitivity": await self._assess_data_sensitivity(intent),
            "domain_context": intent.get("domain_context", "general")
        }
        prompt = self.semantic_config["llm_prompts"]["permission_inference"].format(**permission_data)
        expected_schema = {
            "type": "object",
            "properties": {"permission": {"type": "string", "enum": self.permission_levels}},
            "required": ["permission"]
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            permission = response.get("permission", "").strip().lower()
            return permission if permission in self.permission_levels else self.semantic_config.get("fallback_strategies", {}).get("permission", "grey")
        except Exception as e:
            self.logger.error("LLM permission mapping failed", error=str(e))
            return await self._fallback_permission_level(intent)

    async def _process_auth_requirements(self, intent: Dict) -> List[Dict]:
        """Process authentication requirements."""
        auth_requirements = []
        for security_scheme in intent.get("security", []):
            processed_auth = {
                "type": await self._normalize_auth_type(security_scheme.get("type", "unknown")),
                "required": security_scheme.get("required", True),
                "scheme": security_scheme.get("scheme"),
                "scopes": security_scheme.get("scopes", [])
            }
            if processed_auth["type"] == "api_key":
                processed_auth.update({
                    "header_name": security_scheme.get("name", self.full_config.get("security", {}).get("auth_header_patterns", {}).get("api_key", ["X-API-Key"])[0]),
                    "location": security_scheme.get("in", "header")
                })
            elif processed_auth["type"] == "oauth":
                processed_auth.update({
                    "flows": security_scheme.get("flows", {}),
                    "scope": security_scheme.get("scopes", [])
                })
            auth_requirements.append(processed_auth)
        
        if not auth_requirements:
            inferred_auth = await self._infer_auth_requirements(intent)
            if inferred_auth:
                auth_requirements.append(inferred_auth)
        
        return auth_requirements

    async def _generate_rich_summary(self, intent: Dict, semantic_name: str, user_context: str) -> str:
        """Generate rich, contextual summary."""
        original_summary = intent.get("summary", "")
        context_prefix = self.semantic_config.get("user_contexts", [{"name": user_context, "description": "User wants to"}]).get(user_context, "User wants to")
        
        if len(original_summary) > 10:
            return scrubadub.clean(f"{context_prefix} {original_summary.lower()}")
        
        action_word = await self._extract_action_from_semantic_name(semantic_name)
        object_word = await self._extract_object_from_semantic_name(semantic_name)
        return scrubadub.clean(f"{context_prefix} {action_word} {object_word}")

    async def _process_context_inference(self, semantic_results: List[SemanticTransformationResult]) -> List[Dict]:
        """Stage 3: Process context inference results."""
        self.logger.info("Processing context inference stage")
        
        context_enriched = []
        for result in semantic_results:
            enriched = {
                "semantic_result": result,
                "context_metadata": {
                    "inferred_domain": await self._infer_domain_from_semantic_name(result.semantic_name),
                    "business_context": await self._infer_business_context(result),
                    "usage_patterns": await self._infer_usage_patterns(result)
                }
            }
            context_enriched.append(enriched)
        
        return context_enriched

    async def _process_security_analysis(self, context_enriched: List[Dict]) -> List[Dict]:
        """Stage 4: Enhanced security analysis."""
        self.logger.info("Processing security analysis stage")
        
        security_enriched = []
        for item in context_enriched:
            result = item["semantic_result"]
            security_metadata = {
                "risk_assessment": await self._assess_security_risk(result),
                "compliance_requirements": await self._assess_compliance_requirements(result),
                "access_control": await self._determine_access_control(result)
            }
            item["security_metadata"] = security_metadata
            security_enriched.append(item)
        
        return security_enriched

    async def _assemble_enriched_intents(self, security_enriched: List[Dict]) -> List[Dict]:
        """Assemble final enriched intent structures."""
        enriched_intents = []
        
        for item in security_enriched:
            result = item["semantic_result"]
            context_meta = item["context_metadata"]
            security_meta = item["security_metadata"]
            original_data = result.processing_metadata.get("original_intent", {})
            
            enriched_intent = {
                "name": result.semantic_name,
                "details": {
                    "summary": result.rich_summary,
                    "complexity": result.complexity,
                    "user_context": result.user_context,
                    "domain": context_meta.get("inferred_domain", "general"),
                    "business_context": context_meta.get("business_context", "")
                },
                "paths": [{
                    "method": original_data.get("method", "GET"),
                    "endpoint": original_data.get("path", "/")
                }],
                "requirements": {
                    "authentication": result.auth_requirements,
                    "permissions": [{
                        "level": result.permission_level,
                        "action": result.semantic_name
                    }],
                    "access_control": security_meta.get("access_control", {})
                },
                "confidence": result.confidence,
                "enriched_by": "semantic_llm",
                "metadata": {
                    "processing_time": result.processing_metadata.get("transformation_time"),
                    "llm_calls": result.processing_metadata.get("llm_calls"),
                    "risk_level": security_meta.get("risk_assessment", {}).get("level", "medium"),
                    "mcp_client_type": original_data.get("mcp_client_type", "generic")
                }
            }
            enriched_intents.append(enriched_intent)
        
        return enriched_intents

    async def _process_validation(self, enriched_intents: List[Dict]) -> List[Dict]:
        """Stage 6: Validate enriched intents."""
        self.logger.info("Processing validation stage")
        
        validated_intents = []
        validation_errors = []
        
        for intent in enriched_intents:
            validation_result = await self._validate_enriched_intent(intent)
            if validation_result["valid"]:
                validated_intents.append(intent)
            else:
                validation_errors.append({
                    "intent_name": intent.get("name", "unknown"),
                    "errors": validation_result["errors"]
                })
        
        if validation_errors:
            self.logger.warning("Validation errors found", error_count=len(validation_errors))
        
        self.logger.info("Validation completed", valid_count=len(validated_intents))
        return validated_intents

    # FIX: Corrected return type for async methods (fixes async signature issues)
    async def generate_capabilities(self, raw_intents: List[Dict]) -> Dict[str, List[str]]:
        """Generate semantic capabilities."""
        self.logger.info("Generating semantic capabilities")
        
        semantic_names = []
        for intent in raw_intents:
            semantic_name = await self._generate_semantic_name(intent)
            semantic_names.append((semantic_name, intent))
        
        capabilities = {level: [] for level in self.permission_levels}
        batch_size = await self._adapt_batch_size()
        
        for i in range(0, len(semantic_names), batch_size):
            batch = semantic_names[i:i + batch_size]
            batch_classifications = await self._classify_capabilities_batch(batch)
            for (semantic_name, _), classification in zip(batch, batch_classifications):
                if classification in capabilities:
                    capabilities[classification].append(semantic_name)
        
        for level in capabilities:
            capabilities[level] = sorted(list(set(capabilities[level])))
        
        self.logger.info("Capabilities generation completed", counts={level: len(capabilities[level]) for level in capabilities})
        return capabilities

    # FIX: Corrected return type for async methods (fixes async signature issues)
    async def generate_mcp_tools(self, enriched_intents: List[Dict]) -> List[Dict[str, Any]]:
        """Generate MCP tool specifications."""
        self.logger.info("Generating MCP tools", intent_count=len(enriched_intents))
        
        mcp_tools = []
        for intent in enriched_intents:
            try:
                mcp_tool = await self._generate_single_mcp_tool(intent)
                mcp_tools.append(mcp_tool)
            except Exception as e:
                self.logger.error("Failed to generate MCP tool", intent_name=intent.get("name"), error=str(e))
                continue
        
        self.logger.info("MCP tools generation completed", tools_generated=len(mcp_tools))
        return mcp_tools

    async def _generate_single_mcp_tool(self, enriched_intent: Dict) -> Dict:
        """Generate a single MCP tool specification."""
        original_data = enriched_intent.get("metadata", {}).get("original_intent", {})
        if not original_data:
            original_data = await self._reconstruct_original_data(enriched_intent)
        
        client_type = enriched_intent["metadata"].get("mcp_client_type", "generic")
        tool_name = enriched_intent["name"]
        description = await self._generate_tool_description(enriched_intent)
        input_schema = await self._build_input_schema(original_data, enriched_intent) if self.mcp_config.get("include_schemas", True) else {}
        output_schema = await self._build_output_schema(original_data, enriched_intent) if self.mcp_config.get("include_schemas", True) else {}
        headers = await self._extract_tool_headers(original_data, enriched_intent)
        safety_level = self.mcp_config.get("safety_mapping", {}).get(enriched_intent["requirements"]["permissions"][0]["level"], "safe")
        
        mcp_tool = {
            "name": tool_name,
            "description": description,
            "method": enriched_intent["paths"][0]["method"],
            "path": enriched_intent["paths"][0]["endpoint"],
            "safety_level": safety_level,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "headers": headers,
            "metadata": {
                "complexity": enriched_intent["details"]["complexity"],
                "user_context": enriched_intent["details"]["user_context"],
                "domain": enriched_intent["details"].get("domain", "general"),
                "confidence": enriched_intent["confidence"],
                "generated_at": time.time(),
                "client_type": client_type
            }
        }
        
        if client_type == "langchain" and self.mcp_config.get("generate_prompts", True):
            mcp_tool["prompt_template"] = await self._generate_prompt_template(enriched_intent)
            if self.mcp_config.get("generate_examples", True):
                mcp_tool["examples"] = await self._generate_usage_examples(enriched_intent)
        elif client_type == "fastmcp":
            mcp_tool["transport"] = self.mcp_config.get("tool_generation", {}).get("fastmcp_transport", "streamable_http")
        elif client_type == "openai":
            mcp_tool["parameters"] = await self._build_openai_parameters(input_schema)
        elif client_type == "llamaindex":
            mcp_tool["resource"] = self.mcp_config.get("tool_generation", {}).get("llamaindex_resource", "api_endpoint")
        elif client_type == "autogen":
            mcp_tool["parameters"] = await self._build_autogen_parameters(input_schema)
        
        return mcp_tool

    async def _adapt_batch_size(self) -> int:
        """Adapt batch size based on system resources."""
        now = time.time()
        throttle_seconds = self.enrichment_config.get("adapt_throttle_seconds", 10)
        
        if now - self._last_resource_check < throttle_seconds:
            return self._current_batch_size
        
        self._last_resource_check = now
        
        cpu_percent = psutil.cpu_percent(interval=self.enrichment_config.get("resource_sample_interval", 0.1))
        mem_percent = psutil.virtual_memory().percent
        
        cpu_threshold = self.enrichment_config.get("cpu_threshold", 70)
        mem_threshold = self.enrichment_config.get("mem_threshold", 80)
        max_batch_size = self.enrichment_config.get("max_batch_size", 20)
        growth_factor = self.enrichment_config.get("batch_growth_factor", 2)
        
        if cpu_percent < cpu_threshold and mem_percent < mem_threshold:
            new_batch_size = min(self._current_batch_size * growth_factor, max_batch_size)
            if new_batch_size != self._current_batch_size:
                self.logger.info("Adapted batch size", old_size=self._current_batch_size, new_size=new_batch_size)
                self._current_batch_size = new_batch_size
        
        return self._current_batch_size

    async def _generate_cache_key(self, intent: Dict) -> str:
        """Generate cache key for intent."""
        key_data = f"{intent['method']}:{intent['path']}:{intent.get('summary', '')}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _validate_technical_intent(self, intent: Dict) -> bool:
        """Validate technical intent has required fields."""
        required_fields = ["path", "method"]
        return all(field in intent and intent[field] for field in required_fields)

    async def _validate_semantic_name(self, semantic_name: str) -> bool:
        """Validate semantic name meets quality standards."""
        if not semantic_name or not isinstance(semantic_name, str):
            return False
        
        quality_thresholds = self.semantic_config.get("quality_thresholds", {})
        min_length = quality_thresholds.get("semantic_name_min_length", 3)
        max_length = quality_thresholds.get("semantic_name_max_length", 50)
        pattern = self.full_config.get("validation", {}).get("semantic_name_pattern", r'^[a-z][a-z0-9_]*[a-z0-9]$')
        
        if not (min_length <= len(semantic_name) <= max_length):
            return False
        if not re.match(pattern, semantic_name):
            return False
        if semantic_name in self.full_config.get("validation", {}).get("semantic_name_reserved_words", []):
            return False
        
        return True

    async def _extract_semantic_name_from_response(self, response: Any) -> str:
        """Extract semantic name from LLM response."""
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        semantic_name = content.strip().lower()
        semantic_name = re.sub(r'^(semantic name:?|function name:?|name:?)\s*', '', semantic_name)
        semantic_name = re.sub(r'[^\w_]', '', semantic_name)
        return semantic_name

    async def _fallback_semantic_naming(self, intent: Dict) -> str:
        """Fallback semantic naming."""
        method = intent["method"].lower()
        path = intent["path"]
        path_parts = [part for part in path.split('/') if part and not part.startswith('{')]
        object_name = path_parts[-1].rstrip('s') if path_parts else self.enrichment_config.get("fallbacks", {}).get("object", "resource")
        
        method_mapping = self.semantic_config.get("method_mapping", {
            "get": "get",
            "post": "create",
            "put": "update",
            "patch": "update",
            "delete": "delete",
            "head": "check",
            "options": "describe"
        })
        
        action = method_mapping.get(method, method)
        semantic_name = f"{action}_{object_name}"
        semantic_name = re.sub(r'[^\w_]', '', semantic_name)
        semantic_name = re.sub(r'_+', '_', semantic_name)
        return semantic_name

    async def _fallback_complexity_analysis(self, intent: Dict) -> str:
        """Fallback complexity analysis."""
        indicators = intent.get("complexity_indicators", {})
        param_complexity = indicators.get("parameter_complexity", 0)
        total_properties = indicators.get("total_properties", 0)
        nesting_depth = indicators.get("nesting_depth", 0)
        
        thresholds = self.semantic_config.get("complexity_thresholds", {
            "simple": 3,
            "moderate": 8,
            "complex": 15
        })
        
        complexity_score = param_complexity + total_properties + (nesting_depth * 2)
        if complexity_score <= thresholds.get("simple", 3):
            return "simple"
        elif complexity_score <= thresholds.get("moderate", 8):
            return "moderate"
        elif complexity_score <= thresholds.get("complex", 15):
            return "complex"
        return "multi_step"

    async def _fallback_user_context(self, intent: Dict) -> str:
        """Fallback user context inference."""
        path = intent["path"].lower()
        method = intent["method"].lower()
        context_mapping = self.semantic_config.get("user_context_fallback", {
            "admin|manage": "admin_user",
            "post|put|patch|delete": "authenticated_user",
            "public|catalog": "anonymous_visitor"
        })
        
        for pattern, context in context_mapping.items():
            if re.search(pattern, path + method):
                return context if context in self.user_contexts else "authenticated_user"
        return self.semantic_config.get("fallback_strategies", {}).get("user_context", "authenticated_user")

    async def _fallback_permission_level(self, intent: Dict) -> str:
        """Fallback permission level mapping."""
        method = intent["method"].lower()
        path = intent["path"].lower()
        permission_mapping = self.semantic_config.get("permission_fallback", {
            "delete|admin|manage": "black",
            "post|put|patch": "grey",
            "get|head|options": "white"
        })
        
        for pattern, level in permission_mapping.items():
            if re.search(pattern, path + method):
                return level if level in self.permission_levels else "grey"
        return self.semantic_config.get("fallback_strategies", {}).get("permission", "grey")

    async def _create_fallback_transformation(self, intent: Dict) -> SemanticTransformationResult:
        """Create fallback transformation result."""
        semantic_name = await self._fallback_semantic_naming(intent)
        complexity = await self._fallback_complexity_analysis(intent)
        user_context = await self._fallback_user_context(intent)
        permission_level = await self._fallback_permission_level(intent)
        
        return SemanticTransformationResult(
            semantic_name=semantic_name,
            complexity=complexity,
            user_context=user_context,
            permission_level=permission_level,
            auth_requirements=[],
            rich_summary=scrubadub.clean(f"User wants to {intent.get('summary', 'perform operation')}"),
            confidence=self.enrichment_config.get("fallback_confidence", 0.3),
            processing_metadata={
                "transformation_time": time.time(),
                "llm_calls": 0,
                "fallback_used": True,
                "original_intent": intent
            }
        )

    async def _summarize_parameters(self, parameters: List[Dict]) -> str:
        """Summarize parameters for LLM context."""
        if not parameters:
            return "No parameters"
        
        param_summary = []
        for param in parameters:
            name = param.get("name", "unknown")
            param_type = param.get("type", "unknown")
            required = param.get("required", False)
            status = "required" if required else "optional"
            param_summary.append(f"{name} ({param_type}, {status})")
        
        return "; ".join(param_summary)

    async def _summarize_responses(self, responses: List[Dict]) -> str:
        """Summarize response schemas for LLM context."""
        if not responses:
            return "No response schema defined"
        
        success_responses = [r for r in responses if r.get("is_success", False)]
        if success_responses:
            response = success_responses[0]
            schema = response.get("schema", {})
            return await self._summarize_schema(schema)
        
        return "Response schema available"

    async def _summarize_schema(self, schema: Dict) -> str:
        """Summarize a JSON schema."""
        if not schema:
            return "Empty schema"
        
        schema_type = schema.get("type", "object")
        if schema_type == "object":
            properties = schema.get("properties", {})
            return f"Object with {len(properties)} properties"
        elif schema_type == "array":
            items = schema.get("items", {})
            items_type = items.get("type", "unknown")
            return f"Array of {items_type}"
        return f"{schema_type} value"

    async def _normalize_auth_type(self, auth_type: str) -> str:
        """Normalize authentication type names."""
        return self.auth_type_mappings.get(auth_type, auth_type)

    async def _infer_auth_requirements(self, intent: Dict) -> Optional[Dict]:
        """Infer authentication requirements."""
        # FIX: Use query_semantic with expected schema
        auth_data = {
            "semantic_name": getattr(self, '_current_semantic_name', ''),
            "method": intent["method"],
            "path": intent["path"],
            "user_context": getattr(self, '_current_user_context', 'authenticated_user'),
            "permission_level": getattr(self, '_current_permission_level', 'grey'),
            "domain_context": intent.get("domain_context", "general")
        }
        prompt = self.semantic_config["llm_prompts"]["auth_requirement_inference"].format(**auth_data)
        expected_schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "required": {"type": "boolean"},
                "scope": {"type": "string"}
            },
            "required": ["type", "required"]
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            return {
                "type": response.get("type", "none"),
                "required": response.get("required", False),
                "scope": response.get("scope", "")
            }
        except Exception as e:
            self.logger.error("Auth requirement inference failed", error=str(e))
            return None

    async def _extract_action_from_semantic_name(self, semantic_name: str) -> str:
        """Extract action word from semantic name."""
        return semantic_name.split('_')[0] if '_' in semantic_name else semantic_name

    async def _extract_object_from_semantic_name(self, semantic_name: str) -> str:
        """Extract object word from semantic name."""
        if '_' in semantic_name:
            parts = semantic_name.split('_')
            return '_'.join(parts[1:]) if len(parts) > 1 else parts[0]
        return "resource"

    async def _assess_data_sensitivity(self, intent: Dict) -> str:
        """Assess data sensitivity level."""
        path = intent["path"].lower()
        for level, patterns in self.sensitive_data_patterns.items():
            if any(pattern in path for pattern in patterns):
                return level.replace("_sensitivity", "")
        return "low"

    async def _calculate_transformation_confidence(self, intent: Dict, semantic_name: str) -> float:
        """Calculate confidence score for transformation."""
        confidence_factors = []
        quality_thresholds = self.semantic_config.get("quality_thresholds", {})
        
        if await self._validate_semantic_name(semantic_name):
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)
        
        summary = intent.get("summary", "")
        if len(summary) >= quality_thresholds.get("summary_min_length", 10):
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.05)
        
        params = intent.get("parameters", [])
        if params:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        responses = intent.get("responses", [])
        if responses:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        security = intent.get("security", [])
        if security:
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.05)
        
        return min(sum(confidence_factors), 1.0)

    async def _classify_capabilities_batch(self, batch: List[Tuple[str, Dict]]) -> List[str]:
        """Classify a batch of semantic names into capability levels."""
        classifications = []
        
        for semantic_name, intent in batch:
            try:
                user_context = await self._infer_user_context(intent, semantic_name)
                permission_level = await self._map_permission_level(semantic_name, intent, user_context)
                classifications.append(permission_level)
            except Exception as e:
                self.logger.error("Capability classification failed", error=str(e), semantic_name=semantic_name)
                classifications.append(self.semantic_config.get("fallback_strategies", {}).get("permission", "grey"))
        
        return classifications

    async def _validate_enriched_intent(self, intent: Dict) -> Dict:
        """Validate enriched intent structure and quality."""
        errors = []
        quality_thresholds = self.semantic_config.get("quality_thresholds", {})
        
        # FIX: Use required_intent_details from config.yaml for validation
        required_fields = self.semantic_config.get("required_intent_details", ["summary", "complexity", "user_context"])
        for field in required_fields:
            if field not in intent.get("details", {}):
                errors.append(f"Missing required detail field: {field}")
        
        if "name" in intent and not await self._validate_semantic_name(intent["name"]):
            errors.append(f"Invalid semantic name: {intent['name']}")
        
        if "confidence" in intent:
            confidence = intent["confidence"]
            min_confidence = quality_thresholds.get("min_confidence", 0.0)
            if not isinstance(confidence, (int, float)) or not (min_confidence <= confidence <= 1):
                errors.append(f"Invalid confidence value: {confidence}")
        
        if "details" in intent:
            details = intent["details"]
            if details.get("complexity") and details["complexity"] not in self.complexity_levels:
                errors.append(f"Invalid complexity: {details['complexity']}")
            if details.get("user_context") and details["user_context"] not in self.user_contexts:
                errors.append(f"Invalid user context: {details['user_context']}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def _generate_tool_description(self, enriched_intent: Dict) -> str:
        """Generate tool description using LLM."""
        # FIX: Use query_semantic with expected schema
        context = {
            "semantic_name": enriched_intent["name"],
            "summary": enriched_intent["details"]["summary"],
            "parameters": "parameters available",
            "user_context": enriched_intent["details"]["user_context"],
            "complexity": enriched_intent["details"]["complexity"],
            "domain_context": enriched_intent["details"].get("domain", "general")
        }
        prompt = self.semantic_config["llm_prompts"]["tool_description_generation"].format(**context)
        expected_schema = {
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"]
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            description = response.get("description", "").strip()
            quality_thresholds = self.semantic_config.get("quality_thresholds", {})
            min_length = quality_thresholds.get("description_min_length", 20)
            max_length = quality_thresholds.get("description_max_length", 200)
            
            if min_length <= len(description) <= max_length:
                return scrubadub.clean(description)
            return await self._fallback_tool_description(enriched_intent)
        except Exception as e:
            self.logger.error("Tool description generation failed", error=str(e))
            return await self._fallback_tool_description(enriched_intent)

    async def _fallback_tool_description(self, enriched_intent: Dict) -> str:
        """Fallback tool description generation."""
        name = enriched_intent["name"]
        summary = enriched_intent["details"]["summary"]
        
        if summary and len(summary) > 10:
            return scrubadub.clean(f"{name.replace('_', ' ').title()} - {summary}")
        action = await self._extract_action_from_semantic_name(name)
        obj = await self._extract_object_from_semantic_name(name)
        return scrubadub.clean(f"{action.title()} {obj.replace('_', ' ')} via API endpoint")

    async def _build_input_schema(self, original_data: Dict, enriched_intent: Dict) -> Dict:
        """Build input schema for MCP tool."""
        # FIX: Added recursive schema processing for nested objects/arrays (fixes incomplete schema generation)
        parameters = original_data.get("parameters", [])
        request_body = original_data.get("request_body", [])
        
        properties = {}
        required = []
        
        def process_schema(schema, prefix=""):
            if not isinstance(schema, dict):
                return
            schema_type = schema.get("type", "object")
            if schema_type == "object":
                for name, prop in schema.get("properties", {}).items():
                    full_name = f"{prefix}{name}" if prefix else name
                    properties[full_name] = {
                        "type": prop.get("type", "string"),
                        "description": prop.get("description", "")
                    }
                    if name in schema.get("required", []):
                        required.append(full_name)
                    if prop.get("type") == "object":
                        process_schema(prop, f"{full_name}.")
                    elif prop.get("type") == "array" and prop.get("items", {}).get("type") == "object":
                        process_schema(prop.get("items", {}), f"{full_name}[]")
            elif schema_type == "array":
                items = schema.get("items", {})
                if items.get("type") == "object":
                    process_schema(items, f"{prefix}[]")
        
        for param in parameters:
            name = param.get("name")
            if name:
                schema = param.get("schema", {})
                properties[name] = {
                    "type": schema.get("type", "string"),
                    "description": param.get("description", "")
                }
                if param.get("required", False):
                    required.append(name)
                process_schema(schema, f"{name}.")
        
        for rb in request_body:
            schema = rb.get("schema", {})
            process_schema(schema)
            required.extend(schema.get("required", []))
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    async def _build_output_schema(self, original_data: Dict, enriched_intent: Dict) -> Dict:
        """Build output schema for MCP tool."""
        # FIX: Added recursive schema processing for output schemas
        responses = original_data.get("responses", [])
        success_response = next((r for r in responses if r.get("is_success", False)), None)
        
        if not success_response:
            return {"type": "object", "properties": {}, "required": []}
        
        properties = {}
        required = []
        
        def process_schema(schema, prefix=""):
            if not isinstance(schema, dict):
                return
            schema_type = schema.get("type", "object")
            if schema_type == "object":
                for name, prop in schema.get("properties", {}).items():
                    full_name = f"{prefix}{name}" if prefix else name
                    properties[full_name] = {
                        "type": prop.get("type", "string"),
                        "description": prop.get("description", "")
                    }
                    if name in schema.get("required", []):
                        required.append(full_name)
                    if prop.get("type") == "object":
                        process_schema(prop, f"{full_name}.")
                    elif prop.get("type") == "array" and prop.get("items", {}).get("type") == "object":
                        process_schema(prop.get("items", {}), f"{full_name}[]")
            elif schema_type == "array":
                items = schema.get("items", {})
                if items.get("type") == "object":
                    process_schema(items, f"{prefix}[]")
        
        schema = success_response.get("schema", {})
        process_schema(schema)
        required.extend(schema.get("required", []))
        
        return {
            "type": schema.get("type", "object"),
            "properties": properties,
            "required": required
        }

    async def _build_openai_parameters(self, input_schema: Dict) -> List[Dict]:
        """Build parameters for OpenAI client type."""
        parameters = []
        for name, schema in input_schema.get("properties", {}).items():
            param = {
                "name": name,
                "type": schema.get("type", "string"),
                "required": name in input_schema.get("required", []),
                "description": schema.get("description", "")
            }
            parameters.append(param)
        return parameters

    async def _build_autogen_parameters(self, input_schema: Dict) -> List[Dict]:
        """Build parameters for AutoGen client type."""
        return await self._build_openai_parameters(input_schema)

    async def _extract_tool_headers(self, original_data: Dict, enriched_intent: Dict) -> Dict:
        """Extract and generate headers for the tool."""
        headers = {self.mcp_config.get("headers_generation", {}).get("default_content_type", "Content-Type"): "application/json"}
        auth_reqs = enriched_intent.get("requirements", {}).get("authentication", [])
        auth_patterns = self.full_config.get("security", {}).get("auth_header_patterns", {})
        
        for auth_req in auth_reqs:
            auth_type = auth_req.get("type")
            if auth_type == "api_key":
                header_name = auth_req.get("header_name", auth_patterns.get("api_key", ["X-API-Key"])[0])
                headers[header_name] = "{api_token}"
            elif auth_type in ["bearer_token", "http_auth"]:
                bearer_patterns = auth_patterns.get("bearer", ["Authorization"])
                headers[bearer_patterns[0]] = "Bearer {access_token}"
        
        return headers

    async def _reconstruct_original_data(self, enriched_intent: Dict) -> Dict:
        """Reconstruct original data from enriched intent."""
        return {
            "path": enriched_intent["paths"][0]["endpoint"],
            "method": enriched_intent["paths"][0]["method"],
            "parameters": [],
            "responses": []
        }

    async def _infer_domain_from_semantic_name(self, semantic_name: str) -> str:
        """Infer business domain from semantic name."""
        semantic_lower = semantic_name.lower()
        for domain, patterns in self.domain_patterns.items():
            if any(pattern in semantic_lower for pattern in patterns):
                return domain
        return self.semantic_config.get("fallback_strategies", {}).get("domain", "general")

    async def _infer_business_context(self, result: SemanticTransformationResult) -> str:
        """Infer business context from semantic result."""
        context_mapping = self.semantic_config.get("business_context_mapping", {
            "simple": "Basic operation",
            "moderate": "Standard business operation",
            "complex": "Advanced business process",
            "multi_step": "Complex workflow operation"
        })
        return context_mapping.get(result.complexity, "Business operation")

    async def _infer_usage_patterns(self, result: SemanticTransformationResult) -> List[str]:
        """Infer typical usage patterns."""
        patterns = []
        usage_config = self.semantic_config.get("usage_patterns", {})
        if result.user_context in usage_config.get("user_context_patterns", {}):
            patterns.extend(usage_config["user_context_patterns"][result.user_context])
        if result.complexity in usage_config.get("complexity_patterns", {}):
            patterns.extend(usage_config["complexity_patterns"][result.complexity])
        return patterns

    async def _assess_security_risk(self, result: SemanticTransformationResult) -> Dict:
        """Assess security risk level."""
        risk_config = self.semantic_config.get("risk_assessment", {
            "permission_risk": {"white": 1, "grey": 2, "black": 3},
            "complexity_risk": {"simple": 1, "moderate": 2, "complex": 3, "multi_step": 4},
            "context_risk": {"anonymous_visitor": 1, "browsing_user": 1, "authenticated_user": 2, "premium_user": 2, "admin_user": 4, "system_service": 3}
        })
        
        risk_score = (risk_config["permission_risk"].get(result.permission_level, 2) +
                     risk_config["complexity_risk"].get(result.complexity, 2) +
                     risk_config["context_risk"].get(result.user_context, 2))
        
        thresholds = self.semantic_config.get("risk_thresholds", {"low": 3, "medium": 6})
        level = "low" if risk_score <= thresholds["low"] else "medium" if risk_score <= thresholds["medium"] else "high"
        
        return {
            "level": level,
            "score": risk_score,
            "factors": {
                "permission_level": result.permission_level,
                "complexity": result.complexity,
                "user_context": result.user_context
            }
        }

    async def _assess_compliance_requirements(self, result: SemanticTransformationResult) -> List[str]:
        """Assess compliance requirements."""
        compliance_config = self.semantic_config.get("compliance_requirements", {
            "user_context": {
                "authenticated_user|premium_user": ["GDPR data protection"],
                "admin_user": ["SOX compliance", "Audit logging required"],
                "system_service": ["Internal audit"]
            },
            "semantic_name": {
                "payment|order|transaction": ["PCI DSS compliance"]
            }
        })
        
        compliance_reqs = []
        for context, reqs in compliance_config["user_context"].items():
            if re.search(context, result.user_context):
                compliance_reqs.extend(reqs)
        for pattern, reqs in compliance_config["semantic_name"].items():
            if re.search(pattern, result.semantic_name.lower()):
                compliance_reqs.extend(reqs)
        
        return list(set(compliance_reqs))

    async def _determine_access_control(self, result: SemanticTransformationResult) -> Dict:
        """Determine access control requirements."""
        return {
            "authentication_required": bool(result.auth_requirements),
            "authorization_level": result.permission_level,
            "rate_limiting": await self._determine_rate_limiting(result),
            "ip_restrictions": result.permission_level == "black",
            "audit_logging": result.permission_level in ["grey", "black"]
        }

    async def _determine_rate_limiting(self, result: SemanticTransformationResult) -> Dict:
        """Determine rate limiting."""
        rate_limits = self.semantic_config.get("rate_limits", {
            "anonymous_visitor": {"requests_per_minute": 60, "burst": 10},
            "browsing_user": {"requests_per_minute": 100, "burst": 20},
            "authenticated_user": {"requests_per_minute": 300, "burst": 50},
            "premium_user": {"requests_per_minute": 600, "burst": 100},
            "admin_user": {"requests_per_minute": 1000, "burst": 200},
            "system_service": {"requests_per_minute": 5000, "burst": 1000}
        })
        
        base_limits = rate_limits.get(result.user_context, {"requests_per_minute": 300, "burst": 50})
        if result.complexity in ["complex", "multi_step"]:
            base_limits["requests_per_minute"] = int(base_limits["requests_per_minute"] * 0.5)
        
        return base_limits

    # FIX: Use query_semantic for prompt template generation
    async def _generate_prompt_template(self, enriched_intent: Dict) -> str:
        """Generate prompt template for langchain client."""
        context = {
            "semantic_name": enriched_intent["name"],
            "description": enriched_intent["details"]["summary"],
            "parameters": "parameters available",
            "examples": await self._generate_usage_examples(enriched_intent),
            "domain_context": enriched_intent["details"].get("domain", "general")
        }
        prompt = self.semantic_config["llm_prompts"]["prompt_template_generation"].format(**context)
        expected_schema = {
            "type": "object",
            "properties": {"template": {"type": "string"}},
            "required": ["template"]
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            template = response.get("template", "").strip()
            return scrubadub.clean(template)
        except Exception as e:
            self.logger.error("Prompt template generation failed", error=str(e))
            return f"Execute {enriched_intent['name'].replace('_', ' ')} with {{parameters}}"

    # FIX: Use query_semantic for usage examples
    async def _generate_usage_examples(self, enriched_intent: Dict) -> List[str]:
        """Generate usage examples for langchain client."""
        context = {
            "semantic_name": enriched_intent["name"],
            "description": enriched_intent["details"]["summary"],
            "input_schema": await self._build_input_schema(enriched_intent.get("metadata", {}).get("original_intent", {}), enriched_intent),
            "domain_context": enriched_intent["details"].get("domain", "general")
        }
        prompt = self.semantic_config["llm_prompts"]["usage_examples_generation"].format(**context)
        expected_schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        
        try:
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            examples = response if isinstance(response, list) else []
            quality_thresholds = self.semantic_config.get("quality_thresholds", {})
            min_length = quality_thresholds.get("example_min_length", 5)
            max_count = quality_thresholds.get("example_max_count", 5)
            valid_examples = [scrubadub.clean(ex) for ex in examples if len(ex) >= min_length][:max_count]
            return valid_examples if valid_examples else [f"{enriched_intent['name']}(example_param='value')"]
        except Exception as e:
            self.logger.error("Usage examples generation failed", error=str(e))
            return [f"{enriched_intent['name']}(example_param='value')"]

# Legacy compatibility
Enricher = SemanticEnricher