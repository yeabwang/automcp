# parsers.py - Optimized Parser for MCP Tool Generation Framework
# =============================================================================
# This module provides a domain-agnostic parsing framework for OpenAPI, Postman,
# REST API endpoint discovery, and repository sources, producing rich metadata for LLM-driven enrichment.
# Key Features:
# - Config-driven authentication with LLM fallback for ambiguous cases.
# - Domain context inference for semantic alignment.
# - Selective async processing for LLM tasks to balance performance.
# - Comprehensive PII scrubbing and metadata extraction.
# - Support for multiple MCP client types (langchain, fastmcp, openai, llamaindex, autogen, generic).
# - Robust error handling with retries and circuit breakers.

# Critical Requirements:
# - Industry-standard: Supports OpenAPI, Postman, and MCP standards with Pydantic validation.
# - Production-ready: Retries, circuit breakers, comprehensive logging, and monitoring.
# - Secure: PII scrubbing with config-driven patterns, auth extraction with LLM fallback.
# - Rich: Full metadata extraction (parameters, responses, security, domain context).
# - Robust: Handles edge cases, complex schemas, multiple auth schemes, and ambiguous sources.
# - No hardcoded logic: Uses config-driven mappings and LLM prompts for flexibility.
# =============================================================================

# Standard library
import asyncio
import concurrent.futures
import io
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Awaitable
import logging
import yaml

# Third-party
import git
import griffe
import cProfile
import prance
import pydantic
import requests
import scrubadub
import structlog
import tenacity
import pstats
from circuitbreaker import circuit
from typing import Literal

# Local
from .enricher import SemanticEnricher
from .output_generator import EnhancedOutputGenerator
from .async_llm_client import EnhancedAsyncLLMClient
from .exceptions import ParserError, ClientError

logger = structlog.get_logger(__name__)

def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge for profile overrides."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base

# Nested Pydantic models for type safety (Fix Error 1)
class MCPConfigModel(pydantic.BaseModel):
    client_type: str = "generic"
    generate_tools: bool = True
    include_schemas: bool = True
    generate_prompts: bool = True
    generate_examples: bool = True
    safety_mapping: Dict[str, str] = {}
    tool_template: str = "json"
    tool_generation: Dict[str, Any] = {}
    input_schema_generation: Dict[str, Any] = {}
    output_schema_generation: Dict[str, Any] = {}
    headers_generation: Dict[str, Any] = {}

class SemanticTransformationModel(pydantic.BaseModel):
    enabled: bool = True
    mode: str = "llm_driven"
    confidence_threshold: float = 0.7
    domain_patterns: Dict[str, List[str]] = {}
    complexity_levels: List[Dict[str, Any]] = []
    user_contexts: List[Dict[str, Any]] = []
    permission_levels: Dict[str, Dict[str, Any]] = {}
    auth_type_mappings: Dict[str, str] = {}
    sensitive_data_patterns: Dict[str, List[str]] = {}
    llm_prompts: Dict[str, str] = {}
    quality_thresholds: Dict[str, Any] = {}
    fallback_strategies: Dict[str, str] = {}
    caching: Dict[str, Any] = {}

class ConfigModel(pydantic.BaseModel):
    """Pydantic model for validating configuration with semantic transformation support."""
    parsers_order: List[str]
    flatten_depth: Dict[str, int]
    log_progress: bool
    profile_enrichment: bool
    enrichment: Dict[str, Any]
    output: Dict[str, Any]
    logging: Dict[str, Any]
    validation: Dict[str, Any]
    llm_client: Dict[str, Any]
    timeouts: Dict[str, float]
    parser_workers: int
    spec_file_priorities: List[str]
    profile: str
    profiles: Dict[str, Dict[str, Any]]
    mcp: MCPConfigModel
    exporter: Dict[str, Any]
    security: Dict[str, Any]
    error_handling: Dict[str, Any]
    monitoring: Dict[str, Any]
    telemetry: Dict[str, Any]
    semantic_transformation: SemanticTransformationModel

    @classmethod
    def from_config(cls, config: Dict) -> 'ConfigModel':
        """Load config with profile overrides."""
        profile_name = config.get("profile", "development")
        if profile_name in config.get("profiles", {}):
            base_config = config.copy()
            profile_overrides = config["profiles"][profile_name]
            merged_config = deep_merge(base_config, profile_overrides)
            return cls.model_validate(merged_config)
        return cls.model_validate(config)

    @pydantic.model_validator(mode="after")
    def set_defaults(self) -> 'ConfigModel':
        """Set default values for missing fields."""
        self.flatten_depth = self.flatten_depth or {"default": 3}
        self.timeouts = self.timeouts or {"request": 30.0, "subprocess": 60.0, "llm_query": 60.0}
        if not self.semantic_transformation:
            self.semantic_transformation = SemanticTransformationModel()
        return self

class EnhancedIntentModel(pydantic.BaseModel):
    """Enhanced intent model with full OpenAPI metadata and domain context."""
    path: str
    method: str
    summary: str = ""
    description: str = ""
    operation_id: str = ""
    tags: List[str] = []
    parameters: List[Dict] = []
    requestBody: List[Dict] = []
    responses: List[Dict] = []
    security: List[Dict] = []
    servers: List[Dict] = []
    deprecated: bool = False
    external_docs: Optional[Dict] = None
    domain_context: Optional[str] = None
    _raw_openapi_data: Optional[Dict] = None
    mcp_client_type: str = "generic"
    config: Optional[Dict] = None  # Fix Error 15

    @pydantic.field_validator("method")
    @classmethod
    def validate_method(cls, value: str, info: pydantic.ValidationInfo) -> str:
        """Validate HTTP method against config."""
        config = getattr(info, 'context', {}) or {}
        valid_methods = config.get("validation", {}).get("valid_http_methods", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
        if value.upper() not in valid_methods:
            raise ValueError(f"Invalid HTTP method: {value}")
        return value.upper()

def configure_structlog(config: Dict[str, Any]) -> structlog.BoundLogger:
    """Configure structlog with config-driven redaction and format."""
    log_format = config["logging"].get("format", "console")
    redact_keys = config["logging"].get("redact_keys", [])
    sensitive_patterns = config["security"].get("sensitive_data_patterns", {})
    all_sensitive_keys = redact_keys + sum(sensitive_patterns.values(), [])

    if log_format == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    elif log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    elif log_format == "keyvalue":
        renderer = structlog.processors.KeyValueRenderer()
    else:
        raise ValueError(f"Invalid logging format: {log_format}")

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            create_redact_processor(all_sensitive_keys),
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    return structlog.get_logger(__name__)

def create_redact_processor(redact_keys: List[str]):
    """Create redaction processor for sensitive data."""
    def redact_sensitive(logger, method_name, event_dict):
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
    return redact_sensitive

class ParserInterface:
    """Base interface for specification parsers."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # FIX: Made parse async to support async LLM calls and integrate with async pipeline (fixes async/await inconsistency)
    async def parse(self, spec_path: str) -> Dict[str, Any]:  # Fix Errors 5, 8, 9
        raise NotImplementedError

class EnhancedOpenAPIParser(ParserInterface):
    """Optimized OpenAPI parser with config-driven auth, LLM fallback, and domain context."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_depth = config["flatten_depth"].get("openapi", config["flatten_depth"].get("default", 3))
        self.auth_type_mappings = config["semantic_transformation"].get("auth_type_mappings", {})
        self.security_config = config["security"]
        self.llm_client = EnhancedAsyncLLMClient(
            config=self.config,
            endpoint_url=config["llm_client"].get("endpoint", ""),
            headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
        )
        self.mcp_client_type = config["mcp"].get("client_type", "generic")

    @circuit(failure_threshold=5, recovery_timeout=60)
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=8),
        retry=tenacity.retry_if_exception_type((requests.RequestException, ClientError)),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )
    # FIX: Made parse async to allow awaiting _infer_security_for_intents directly (fixes async inconsistency and runtime errors)
    async def parse(self, spec_path: str) -> Dict[str, Any]:  # Now async 
        """Parse OpenAPI spec with full metadata, domain context, and MCP client support."""
        try:
            if self.config["log_progress"]:
                logger.info("Starting OpenAPI parsing", spec_path=spec_path)

            parser = prance.ResolvingParser(spec_path, lazy=False, resolve_types=True)
            spec = parser.specification
            if not isinstance(spec, dict) or not spec:
                raise ParserError(f"Invalid OpenAPI specification: {spec_path}")

            self.full_spec = spec
            intents = []
            paths = spec.get("paths", {})
            domain_context = self._infer_domain_context(spec)

            # FIX: Added context manager for ThreadPoolExecutor to prevent resource leaks (shutdown properly)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["parser_workers"]) as executor:
                futures = [
                    executor.submit(self.process_path, path, methods, spec, domain_context)
                    for path, methods in paths.items()
                ]
                intents_needing_inference = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        path_intents = future.result()
                        for intent in path_intents:
                            if not intent["security"] and self.config["semantic_transformation"].get("enabled", False):
                                intents_needing_inference.append(intent)
                            else:
                                intents.append(intent)
                    except Exception as exc:
                        logger.error("Path processing failed", error=str(exc))

            # Post-process intents needing LLM inference
            if intents_needing_inference:
                inferred_intents = await self._infer_security_for_intents(intents_needing_inference, spec, domain_context)  # Now await directly (no asyncio.run)
                intents.extend(inferred_intents)

            return {
                "intents": intents,
                "info": spec.get("info", {}),
                "servers": spec.get("servers", []),
                "security": spec.get("security", []),
                "components": spec.get("components", {}),
                "security_schemes": self._extract_global_security_schemes(spec),
                "tags": spec.get("tags", []),
                "external_docs": spec.get("externalDocs"),
                "domain_context": domain_context,
                "_full_spec": spec
            }

        except Exception as e:
            logger.error("OpenAPI parsing failed", spec_path=spec_path, error=str(e))
            raise ParserError(f"OpenAPI parsing failed for {spec_path}: {str(e)}") from e

    def process_path(self, path: str, methods: Dict, spec: Dict, domain_context: str) -> List[Dict]:  # Fix Error 2
        """Process path with enhanced metadata (sync for ThreadPoolExecutor)."""
        path_intents = []
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue

            try:
                params = self._flatten_schema(details.get("parameters", []), self.max_depth)
                request_body = self._extract_request_body(details.get("requestBody", {}))
                responses = self._extract_response_schemas(details.get("responses", {}))
                security_reqs = self._extract_operation_security(spec, details)

                intent = {
                    "path": path,
                    "method": method.upper(),
                    "summary": details.get("summary", ""),
                    "description": details.get("description", ""),
                    "operation_id": details.get("operationId", ""),
                    "tags": details.get("tags", []),
                    "parameters": params,
                    "requestBody": request_body,
                    "responses": responses,
                    "security": security_reqs,
                    "servers": details.get("servers", spec.get("servers", [])),
                    "deprecated": details.get("deprecated", False),
                    "external_docs": details.get("externalDocs"),
                    "domain_context": domain_context,
                    "mcp_client_type": self.mcp_client_type,
                    "parameter_count": len(params),
                    "has_request_body": bool(request_body),
                    "response_types": [r.get("media_type") for r in responses],
                    "requires_auth": bool(security_reqs),
                    "complexity_indicators": self._calculate_complexity_indicators(params, request_body, responses),
                    "_raw_openapi_data": {
                        "path": path,
                        "method": method,
                        "operation": details,
                        "global_security": spec.get("security", []),
                        "components": spec.get("components", {})
                    }
                }

                intent = self._scrub_pii(intent)
                path_intents.append(intent)

            except Exception as e:
                logger.error("Operation processing failed", path=path, method=method, error=str(e))
                continue

        return path_intents

    async def _infer_security_for_intents(self, intents: List[Dict], spec: Dict, domain_context: str) -> List[Dict]:
        """Infer security requirements for multiple intents using LLM."""
        inferred_intents = []
        for intent in intents:
            path = intent["path"]
            method = intent["method"]
            operation = intent["_raw_openapi_data"]["operation"]
            security_reqs = await self._infer_security_with_llm(spec, operation, path, method, domain_context)
            intent["security"] = security_reqs
            intent["requires_auth"] = bool(security_reqs)
            inferred_intents.append(intent)
        return inferred_intents

    async def _infer_security_with_llm(self, spec: Dict, operation: Dict, path: str, method: str, domain_context: str) -> List[Dict]:
        """Infer security requirements using LLM for ambiguous cases."""
        if not self.config["semantic_transformation"].get("enabled", False):
            return []

        prompt = self.config["semantic_transformation"]["llm_prompts"]["auth_requirement_inference"].format(
            semantic_name=operation.get("operationId", f"{method}_{path.replace('/', '_')}"),
            method=method.upper(),
            path=path,
            user_context="unknown",
            permission_level="unknown",
            domain_context=domain_context
        )
        try:
            provider_settings = self.config["llm_client"]["provider_settings"].get("groq", {})
            response = await self.llm_client.query(prompt=prompt, **provider_settings)  # Fix Errors 3, 4
            auth_req = json.loads(response)
            mapped_type = self.auth_type_mappings.get(auth_req.get("type", "none"), "none")
            return [{
                "scheme_name": "inferred",
                "type": mapped_type,
                "required": auth_req.get("required", False),
                "scopes": auth_req.get("scope", "").split(",") if auth_req.get("scope") else [],
                "inferred_by_llm": True
            }]
        except Exception as e:
            logger.warning("LLM auth inference failed", error=str(e))
            return [{
                "scheme_name": "inferred",
                "type": self.config["semantic_transformation"]["fallback_strategies"].get("auth_type", "none"),
                "required": False,
                "scopes": [],
                "inferred_by_llm": True
            }]

    def _extract_operation_security(self, spec: Dict, operation: Dict) -> List[Dict]:
        """Extract security requirements with config-driven mappings."""
        security_schemes = spec.get("components", {}).get("securitySchemes", {})
        operation_security = operation.get("security", spec.get("security", []))
        auth_headers = self.security_config.get("auth_header_patterns", {})

        extracted_reqs = []
        for security_requirement in operation_security:
            for scheme_name, scopes in security_requirement.items():
                scheme_definition = security_schemes.get(scheme_name, {})
                scheme_type = scheme_definition.get("type", "unknown")
                mapped_type = self.auth_type_mappings.get(scheme_type, "other")

                security_req = {
                    "scheme_name": scheme_name,
                    "type": mapped_type,
                    "original_type": scheme_type,
                    "description": scheme_definition.get("description", ""),
                    "scopes": scopes if isinstance(scopes, list) else [],
                    "required": True
                }

                if mapped_type == "api_key":
                    security_req.update({
                        "name": scheme_definition.get("name", auth_headers.get("api_key", ["X-API-Key"])[0]),
                        "in": scheme_definition.get("in", "header")
                    })
                elif mapped_type == "http_auth":
                    security_req.update({
                        "scheme": scheme_definition.get("scheme", "bearer"),
                        "bearer_format": scheme_definition.get("bearerFormat", "")
                    })
                elif mapped_type == "oauth":
                    security_req.update({
                        "flows": scheme_definition.get("flows", {}),
                        "available_scopes": self._extract_oauth_scopes(scheme_definition.get("flows", {}))
                    })
                elif mapped_type == "openid_connect":
                    security_req.update({
                        "openid_connect_url": scheme_definition.get("openIdConnectUrl", "")
                    })
                else:
                    security_req.update({
                        "raw_definition": scheme_definition,
                        "requires_llm_inference": True
                    })

                extracted_reqs.append(security_req)

        return extracted_reqs

    # FIX: Added validation for empty/invalid responses to prevent KeyError and improve robustness (edge cases)
    def _extract_response_schemas(self, responses: Dict) -> List[Dict]:
        """Extract response schemas with enhanced analysis."""
        if not isinstance(responses, dict) or not responses:
            logger.warning("Empty or invalid responses dict", responses=responses)
            return []  # Return empty list to avoid downstream errors

        extracted_responses = []
        valid_content_types = self.config["validation"].get("valid_content_types", [])
        for status_code, response_definition in responses.items():
            if not isinstance(response_definition, dict):
                logger.warning("Invalid response definition for status", status_code=status_code)
                continue  # Skip invalid entries

            headers = response_definition.get("headers", {})
            content = response_definition.get("content", {})
            for media_type, media_definition in content.items():
                if media_type not in valid_content_types:
                    logger.warning("Unsupported content type", media_type=media_type)
                    continue
                schema = media_definition.get("schema", {})
                response_entry = {
                    "status_code": status_code,
                    "status_range": self._categorize_status_code(status_code),
                    "media_type": media_type,
                    "description": response_definition.get("description", ""),
                    "schema": schema,
                    "headers": headers,
                    "examples": media_definition.get("examples", {}),
                    "is_success": str(status_code).startswith("2"),
                    "is_error": not str(status_code).startswith("2"),
                    "schema_complexity": self._analyze_schema_complexity(schema),
                    "content_type_category": self._categorize_content_type(media_type)
                }
                extracted_responses.append(response_entry)

        return extracted_responses

    # FIX: Added validation for empty/invalid request_body to prevent KeyError and improve robustness
    def _extract_request_body(self, request_body: Dict) -> List[Dict]:
        """Extract request body with enhanced analysis."""
        if not isinstance(request_body, dict) or not request_body:
            logger.warning("Empty or invalid request body", request_body=request_body)
            return []  # Return empty list to avoid downstream errors

        extracted_bodies = []
        valid_content_types = self.config["validation"].get("valid_content_types", [])
        content = request_body.get("content", {})
        for media_type, media_definition in content.items():
            if media_type not in valid_content_types:
                logger.warning("Unsupported content type", media_type=media_type)
                continue
            schema = media_definition.get("schema", {})
            body_entry = {
                "media_type": media_type,
                "schema": schema,
                "required": request_body.get("required", False),
                "description": request_body.get("description", ""),
                "examples": media_definition.get("examples", {}),
                "schema_complexity": self._analyze_schema_complexity(schema),
                "flattened_properties": self._flatten_schema([{"schema": schema}], self.max_depth)
            }
            extracted_bodies.append(body_entry)

        return extracted_bodies

    def _extract_global_security_schemes(self, spec: Dict) -> Dict:
        """Extract global security schemes with config-driven categorization."""
        security_schemes = spec.get("components", {}).get("securitySchemes", {})
        categorized_schemes = {key: {} for key in self.auth_type_mappings.values()}
        categorized_schemes["other"] = {}

        for scheme_name, scheme_def in security_schemes.items():
            scheme_type = scheme_def.get("type", "other")
            mapped_type = self.auth_type_mappings.get(scheme_type, "other")
            categorized_schemes[mapped_type][scheme_name] = scheme_def

        return categorized_schemes

    def _infer_domain_context(self, spec: Dict) -> str:
        """Infer domain context from spec for LLM enrichment."""
        if not self.config["semantic_transformation"].get("enabled", False):
            return "generic"

        domain_patterns = self.config["semantic_transformation"].get("domain_patterns", {})
        title = spec.get("info", {}).get("title", "").lower()
        description = spec.get("info", {}).get("description", "").lower()
        tags = [tag.get("name", "").lower() for tag in spec.get("tags", [])]

        for domain, patterns in domain_patterns.items():
            for pattern in patterns:
                if any(pattern in text for text in [title, description] + tags):
                    return domain
        return "generic"

    def _calculate_complexity_indicators(self, params: List[Dict], request_body: List[Dict], responses: List[Dict]) -> Dict:
        """Calculate complexity indicators for LLM enrichment."""
        return {
            "parameter_complexity": len(params) + sum(len(p.get("nested", [])) for p in params),
            "request_body_complexity": sum(rb.get("schema_complexity", 0) for rb in request_body),
            "response_complexity": sum(r.get("schema_complexity", 0) for r in responses),
            "total_properties": self._count_total_properties(params, request_body),
            "nesting_depth": self._calculate_max_nesting_depth(params, request_body),
            "has_arrays": self._has_array_types(params, request_body),
            "has_objects": self._has_object_types(params, request_body)
        }

    def _analyze_schema_complexity(self, schema: Dict) -> int:
        """Analyze schema complexity with scoring system."""
        if not schema:
            return 0

        complexity_score = 0
        schema_type = schema.get("type", "")
        if schema_type == "object":
            properties = schema.get("properties", {})
            complexity_score += len(properties)
            for prop_schema in properties.values():
                if isinstance(prop_schema, dict):
                    complexity_score += self._analyze_schema_complexity(prop_schema) * 0.5
        elif schema_type == "array":
            items_schema = schema.get("items", {})
            complexity_score += 1 + self._analyze_schema_complexity(items_schema) * 0.7

        for combinator in ["allOf", "oneOf", "anyOf"]:
            if combinator in schema:
                complexity_score += len(schema[combinator]) * (2 if combinator == "allOf" else 1.5)

        return int(complexity_score)

    def _flatten_schema(self, items: List[Dict], max_depth: int, depth: int = 0, visited: Optional[Set[int]] = None) -> List[Dict]:
        """Flatten schema with PII scrubbing and validation."""
        if depth > max_depth:
            logger.warning("Maximum flattening depth reached", current_depth=depth, max_depth=max_depth)
            return []

        if visited is None:
            visited = set()

        flattened = []
        for item in items:
            if not isinstance(item, dict):
                continue

            item = self._scrub_pii(item)
            schema_id = id(item)
            if schema_id in visited:
                logger.warning("Schema cycle detected", depth=depth)
                continue
            visited.add(schema_id)

            schema = item.get("schema", {})
            if schema:
                processed_schema = self._process_schema(schema, max_depth, depth, visited)
                flattened.extend(processed_schema)
            elif "content" in item:
                flattened.extend(self._handle_content(item, max_depth, depth, visited))
            else:
                enhanced_item = self._enhance_parameter_metadata(item)
                flattened.append(enhanced_item)

        return flattened

    def _process_schema(self, schema: Dict, max_depth: int, depth: int, visited: Set) -> List[Dict]:
        """Process schema with enhanced type handling."""
        processed_items = []
        schema = self._handle_allOf(schema)

        if "oneOf" in schema:
            processed_items.append(self._handle_oneOf(schema, max_depth, depth + 1, visited.copy()))
        elif "properties" in schema:
            processed_items.extend(self._handle_properties(schema, max_depth, depth + 1, visited.copy()))
        else:
            processed_items.append(self._enhance_primitive_schema(schema))

        return processed_items

    def _enhance_parameter_metadata(self, param: Dict) -> Dict:
        """Enhance parameter with validation metadata."""
        enhanced = param.copy()
        if "type" not in enhanced and "schema" in enhanced:
            enhanced["type"] = enhanced["schema"].get("type", "string")
        enhanced["validation"] = self._extract_validation_rules(param)
        enhanced["usage_hints"] = self._generate_usage_hints(param)
        return enhanced

    def _extract_validation_rules(self, param: Dict) -> Dict:
        """Extract validation rules from parameter."""
        validation = {}
        schema = param.get("schema", {})
        for key in ["minimum", "maximum", "minLength", "maxLength", "pattern", "enum"]:
            if key in schema:
                validation[key.replace("enum", "allowed_values")] = schema[key]
        return validation

    def _scrub_pii(self, data: Any) -> Any:
        """Scrub PII using config-driven sensitive patterns."""
        sensitive_patterns = self.config["security"].get("sensitive_data_patterns", {})
        all_sensitive_keys = sum(sensitive_patterns.values(), [])

        def scrub_recursive(obj):
            if isinstance(obj, dict):
                scrubbed = {}
                for key, value in obj.items():
                    if key in all_sensitive_keys:
                        scrubbed[key] = "[REDACTED]"
                    else:
                        scrubbed[key] = scrub_recursive(value)
                return scrubbed
            elif isinstance(obj, list):
                return [scrub_recursive(item) for item in obj]
            elif isinstance(obj, str):
                return scrubadub.clean(obj)
            return obj

        return scrub_recursive(data)

    def _categorize_status_code(self, status_code: str) -> str:
        """Categorize HTTP status codes."""
        code = str(status_code)
        ranges = {
            "2": "success",
            "3": "redirect",
            "4": "client_error",
            "5": "server_error"
        }
        return ranges.get(code[0], "unknown")

    def _categorize_content_type(self, media_type: str) -> str:
        """Categorize media types."""
        media_type = media_type.lower()
        categories = {
            "json": ["json"],
            "xml": ["xml"],
            "form": ["form"],
            "text": ["text"],
            "binary": ["binary", "octet-stream"]
        }
        for category, patterns in categories.items():
            if any(pattern in media_type for pattern in patterns):
                return category
        return "other"

    def _extract_oauth_scopes(self, flows: Dict) -> List[str]:
        """Extract OAuth scopes."""
        scopes = set()
        for flow_def in flows.values():
            scopes.update(flow_def.get("scopes", {}).keys())
        return list(scopes)

    def _count_total_properties(self, params: List[Dict], request_body: List[Dict]) -> int:
        """Count total properties."""
        count = len(params)
        for rb in request_body:
            count += len(rb.get("schema", {}).get("properties", {}))
        return count

    def _calculate_max_nesting_depth(self, params: List[Dict], request_body: List[Dict]) -> int:
        """Calculate maximum nesting depth."""
        max_depth = max(len(p.get("nested", [])) for p in params) if params else 0
        for rb in request_body:
            depth = self._calculate_schema_depth(rb.get("schema", {}))
            max_depth = max(max_depth, depth)
        return max_depth

    def _calculate_schema_depth(self, schema: Dict, current_depth: int = 0) -> int:
        """Calculate schema nesting depth."""
        if not isinstance(schema, dict):
            return current_depth

        max_depth = current_depth
        for key in ["properties", "items"]:
            if key in schema:
                sub_schema = schema[key] if key == "items" else schema[key].values()
                for sub in sub_schema if isinstance(sub_schema, dict) else [sub_schema]:
                    depth = self._calculate_schema_depth(sub, current_depth + 1)
                    max_depth = max(max_depth, depth)
        return max_depth

    def _has_array_types(self, params: List[Dict], request_body: List[Dict]) -> bool:
        """Check for array types."""
        if any(p.get("type") == "array" for p in params):
            return True
        return any(self._schema_contains_arrays(rb.get("schema", {})) for rb in request_body)

    def _has_object_types(self, params: List[Dict], request_body: List[Dict]) -> bool:
        """Check for object types."""
        if any(p.get("type") == "object" or p.get("nested") for p in params):
            return True
        return any(self._schema_contains_objects(rb.get("schema", {})) for rb in request_body)

    def _schema_contains_arrays(self, schema: Dict) -> bool:
        """Check for arrays in schema."""
        if schema.get("type") == "array":
            return True
        return any(self._schema_contains_arrays(prop) for prop in schema.get("properties", {}).values())

    def _schema_contains_objects(self, schema: Dict) -> bool:
        """Check for objects in schema."""
        if schema.get("type") == "object" or "properties" in schema:
            return True
        return any(self._schema_contains_objects(prop) for prop in schema.get("properties", {}).values())

    def _handle_oneOf(self, schema: Dict, max_depth: int, depth: int, visited: Set) -> Dict:
        """Handle oneOf with type resolution."""
        options = []
        for sub_schema in schema["oneOf"]:
            processed = self._process_schema({"schema": sub_schema}, max_depth, depth, visited)
            options.extend(processed)
        return {
            "name": "oneOf_union",
            "type": "union",
            "options": options,
            "required": False,
            "discriminator": schema.get("discriminator"),
            "union_complexity": len(options)
        }

    def _handle_properties(self, schema: Dict, max_depth: int, depth: int, visited: Set) -> List[Dict]:
        """Handle properties with metadata."""
        properties = []
        required_props = schema.get("required", [])
        for prop_name, prop_schema in schema.get("properties", {}).items():
            nested_items = self._process_schema({"schema": prop_schema}, max_depth, depth, visited) if "properties" in prop_schema else []
            properties.append({
                "name": prop_name,
                "type": prop_schema.get("type", "object" if "properties" in prop_schema else "any"),
                "required": prop_name in required_props,
                "description": prop_schema.get("description", ""),
                "nested": nested_items,
                "validation": self._extract_validation_rules({"schema": prop_schema}),
                "examples": prop_schema.get("examples", []),
                "default": prop_schema.get("default"),
                "deprecated": prop_schema.get("deprecated", False)
            })
        return properties

    def _handle_content(self, item: Dict, max_depth: int, depth: int, visited: Set) -> List[Dict]:
        """Handle content for request bodies."""
        content = item.get("content", {})
        processed_content = []
        valid_content_types = self.config["validation"].get("valid_content_types", [])
        for media_type, media_def in content.items():
            if media_type not in valid_content_types:
                logger.warning("Unsupported content type", media_type=media_type)
                continue
            if "schema" in media_def:
                schema_items = self._process_schema({"schema": media_def["schema"]}, max_depth, depth, visited)
                for schema_item in schema_items:
                    schema_item["media_type"] = media_type
                    schema_item["content_category"] = self._categorize_content_type(media_type)
                processed_content.extend(schema_items)
        return processed_content

    def _enhance_primitive_schema(self, schema: Dict) -> Dict:
        """Enhance primitive schema with metadata."""
        return {
            "type": schema.get("type", "any"),
            "format": schema.get("format"),
            "description": schema.get("description", ""),
            "validation": self._extract_validation_rules({"schema": schema}),
            "examples": schema.get("examples", []),
            "default": schema.get("default"),
            "nullable": schema.get("nullable", False),
            "read_only": schema.get("readOnly", False),
            "write_only": schema.get("writeOnly", False)
        }

    def _generate_usage_hints(self, param: Dict) -> List[str]:
        """Generate usage hints for parameters."""
        hints = []
        schema = param.get("schema", {})
        if schema.get("type") == "string" and "enum" in schema:
            hints.append(f"Allowed values: {', '.join(map(str, schema['enum']))}")
        if "minimum" in schema and "maximum" in schema:
            hints.append(f"Range: {schema['minimum']} - {schema['maximum']}")
        if "pattern" in schema:
            hints.append(f"Pattern: {schema['pattern']}")
        if param.get("required", False):
            hints.append("Required parameter")
        return hints

    def _extract_operation_metadata(self, operation: Dict) -> Dict:
        """Extract operation metadata."""
        return {
            "has_callbacks": bool(operation.get("callbacks")),
            "has_examples": bool(operation.get("examples")),
            "parameter_count": len(operation.get("parameters", [])),
            "response_count": len(operation.get("responses", {})),
            "tag_count": len(operation.get("tags", [])),
            "is_deprecated": operation.get("deprecated", False),
            "has_security": bool(operation.get("security")),
            "has_servers": bool(operation.get("servers"))
        }

    @staticmethod
    def _handle_allOf(schema: Dict) -> Dict:
        """Handle allOf by merging sub-schemas."""
        if "allOf" not in schema:
            return schema

        merged = {"properties": {}, "required": []}
        for sub in schema["allOf"]:
            if not isinstance(sub, dict):
                continue
            merged["properties"].update(sub.get("properties", {}))
            merged["required"].extend(sub.get("required", []))
        merged["required"] = list(set(merged["required"]))
        schema.update(merged)
        schema.pop("allOf")
        return schema

class RESTParser(ParserInterface):
    """Parser for REST APIs fetching OpenAPI specs from URLs."""
    @circuit(failure_threshold=5, recovery_timeout=60)
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=8),
        retry=tenacity.retry_if_exception_type((requests.RequestException,)),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )
    # FIX: Made parse async for consistency with other parsers and async pipeline
    async def parse(self, spec_path: str) -> Dict[str, Any]:
        """Fetch and parse REST API spec."""
        if not spec_path.startswith(("http://", "https://")):
            raise ParserError(f"Invalid REST URL: {spec_path}")

        try:
            if self.config["log_progress"]:
                logger.info("Starting REST parsing", spec_path=spec_path)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(requests.get, spec_path, timeout=self.config["timeouts"]["request"])
                resp = future.result()
            resp.raise_for_status()
            data = resp.json()

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir) / "temp.json"
                with open(temp_path, "w") as f:
                    json.dump(data, f)
                return await EnhancedOpenAPIParser(self.config).parse(str(temp_path))  # Await async parse

        except Exception as e:
            logger.error("REST parsing failed", spec_path=spec_path, error=str(e))
            raise ParserError(f"REST parsing failed for {spec_path}: {str(e)}") from e

class PostmanParser(ParserInterface):
    """Parser for Postman collections with LLM-driven auth inference."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_client = EnhancedAsyncLLMClient(
            config=self.config,
            endpoint_url=config["llm_client"].get("endpoint", ""),
            headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
        )

    @circuit(failure_threshold=5, recovery_timeout=60)
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=8),
        retry=tenacity.retry_if_exception_type((subprocess.CalledProcessError,)),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )
    async def parse(self, spec_path: str) -> Dict[str, Any]:
        """Convert Postman collection to OpenAPI and parse."""
        spec_path = os.path.abspath(spec_path)
        if not Path(spec_path).is_file() or any(c in spec_path for c in [";", "|", "&"]):
            raise ParserError(f"Invalid Postman file path: {spec_path}")

        try:
            if self.config["log_progress"]:
                logger.info("Starting Postman parsing", spec_path=spec_path)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir) / "temp_openapi.json"
                result = subprocess.run(
                    ["npx", "postman-to-openapi", shlex.quote(spec_path), "-o", str(temp_path)],
                    capture_output=True,
                    check=True,
                    timeout=self.config["timeouts"]["subprocess"]
                )
                logger.debug("Postman conversion success", stdout=result.stdout.decode())

                parsed_data = await EnhancedOpenAPIParser(self.config).parse(str(temp_path))  # Await async parse
                if not parsed_data.get("security_schemes") and self.config["semantic_transformation"].get("enabled", False):
                    parsed_data["security_schemes"] = await self._infer_postman_security(spec_path)
                parsed_data["mcp_client_type"] = self.config["mcp"].get("client_type", "generic")
                return parsed_data

        except Exception as e:
            logger.error("Postman parsing failed", error=str(e))
            raise ParserError(f"Postman parsing failed for {spec_path}: {str(e)}") from e

    async def _infer_postman_security(self, spec_path: str) -> Dict:
        """Infer security schemes from Postman collection using LLM."""
        with open(spec_path, "r") as f:
            postman_data = json.load(f)
        prompt = self.config["semantic_transformation"]["llm_prompts"]["auth_requirement_inference"].format(
            semantic_name="postman_collection",
            method="UNKNOWN",
            path="UNKNOWN",
            user_context="unknown",
            permission_level="unknown",
            domain_context="generic"
        )
        try:
            provider_settings = self.config["llm_client"]["provider_settings"].get("groq", {})  # Fix Errors 6, 7
            response = await self.llm_client.query(
                prompt=prompt + f"\nPostman Collection: {json.dumps(postman_data, indent=2)[:2000]}",
                **provider_settings
            )
            auth_req = json.loads(response)
            mapped_type = self.config["semantic_transformation"]["auth_type_mappings"].get(
                auth_req.get("type", "none"), "none"
            )
            return {
                mapped_type: {
                    "inferred": {
                        "type": mapped_type,
                        "required": auth_req.get("required", False),
                        "scopes": auth_req.get("scope", "").split(",") if auth_req.get("scope") else []
                    }
                }
            }
        except Exception as e:
            logger.warning("Postman LLM auth inference failed", error=str(e))
            return {}

class ScrapingParser(ParserInterface):
    """Fallback parser for repository scraping with LLM-driven auth inference."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_client = EnhancedAsyncLLMClient(
            config=self.config,
            endpoint_url=config["llm_client"].get("endpoint", ""),
            headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
        )

    # FIX: Made parse async for consistency
    async def parse(self, spec_path: str) -> Dict[str, Any]:
        """Scan repository for spec files or source code."""
        p = Path(spec_path)
        try:
            if self.config["log_progress"]:
                logger.info("Starting scraping", spec_path=spec_path)

            if p.is_dir():
                repo = git.Repo(p)
                files = repo.git.ls_files().splitlines()
                specs = [
                    f for f in files
                    if any(ext in f.lower() for ext in self.config["spec_file_priorities"])
                ]
                if specs:
                    specs.sort(key=lambda f: self.config["spec_file_priorities"].index(
                        next(ext for ext in self.config["spec_file_priorities"] if ext in f.lower())
                    ))
                    found = p / specs[0]
                    logger.info("Prioritized spec found", found=str(found))
                    if "postman" in found.name.lower():
                        return await PostmanParser(self.config).parse(str(found))
                    return await EnhancedOpenAPIParser(self.config).parse(str(found))
                return await self._parse_source_code(spec_path)
            return await GriffeParser(self.config).parse(spec_path)

        except Exception as e:
            logger.error("Scraping failed", error=str(e))
            raise ParserError(f"Scraping failed for {spec_path}: {str(e)}") from e

    async def _parse_source_code(self, spec_path: str) -> Dict:
        """Parse source code with LLM-driven metadata extraction."""
        parsed_data = await GriffeParser(self.config).parse(spec_path)
        parsed_data["mcp_client_type"] = self.config["mcp"].get("client_type", "generic")
        return parsed_data

class GriffeParser(ParserInterface):
    """Parser for Python source code with LLM-driven auth inference."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_client = EnhancedAsyncLLMClient(
            config=self.config,
            endpoint_url=config["llm_client"].get("endpoint", ""),
            headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
        )

    # FIX: Made parse async for consistency
    async def parse(self, spec_path: str) -> Dict[str, Any]:
        """Parse Python source code for function signatures and docstrings."""
        try:
            if self.config["log_progress"]:
                logger.info("Starting Griffe parsing", spec_path=spec_path)

            mod_name = Path(spec_path).stem
            mod_path = Path(spec_path).parent
            module = griffe.load(mod_name, search_paths=[str(mod_path)])
            if not module:
                raise ParserError(f"Failed to load module from {spec_path}")

            intents = []
            domain_context = "generic"
            for f in module.functions.values():
                if f.name.startswith("_"):
                    continue
                intent = {
                    "name": f.name,
                    "summary": f.docstring.value if f.docstring else "",
                    "parameters": [
                        {"name": p.name, "type": str(p.annotation) or "any"}
                        for p in f.parameters
                    ],
                    "domain_context": domain_context,
                    "mcp_client_type": self.config["mcp"].get("client_type", "generic")
                }
                if not isinstance(intent["parameters"], list):
                    logger.error("Invalid parameters format", intent_name=intent["name"])
                    continue
                intents.append(intent)

            for cls_name, cls in module.classes.items():
                for m_name, m in cls.functions.items():
                    if m_name.startswith("_") and m_name != "__init__":
                        continue
                    intent = {
                        "class": cls_name,
                        "name": m_name,
                        "summary": m.docstring.value if m.docstring else "",
                        "parameters": [
                            {"name": p.name, "type": str(p.annotation) or "any"}
                            for p in m.parameters  # Fix Error 10
                        ],
                        "domain_context": domain_context,
                        "mcp_client_type": self.config["mcp"].get("client_type", "generic")
                    }
                    if not isinstance(intent["parameters"], list):
                        logger.error("Invalid parameters format", intent_name=intent["name"])
                        continue
                    intents.append(intent)

            if not intents:
                raise ParserError(f"No intents extracted from {spec_path}")

            parsed_data = {"intents": intents}
            if self.config["semantic_transformation"].get("enabled", False):
                inferred_security = await self._infer_source_security(spec_path)
                # Convert dict to list format for security_schemes
                parsed_data["security_schemes"] = [inferred_security] if inferred_security else []
            return parsed_data

        except Exception as e:
            logger.error("Griffe parsing failed", error=str(e))
            raise ParserError(f"Griffe parsing failed for {spec_path}: {str(e)}") from e

    async def _infer_source_security(self, spec_path: str) -> Dict:
        """Infer security schemes from Python source code using LLM."""
        with open(spec_path, "r") as f:
            source_code = f.read()
        prompt = self.config["semantic_transformation"]["llm_prompts"]["auth_requirement_inference"].format(
            semantic_name="source_code",
            method="UNKNOWN",
            path="UNKNOWN",
            user_context="unknown",
            permission_level="unknown",
            domain_context="generic"
        )
        try:
            provider_settings = self.config["llm_client"]["provider_settings"].get("groq", {})  # Fix Errors 11, 12
            response = await self.llm_client.query(
                prompt=prompt + f"\nSource Code: {source_code[:2000]}",
                **provider_settings
            )
            auth_req = json.loads(response)
            mapped_type = self.config["semantic_transformation"]["auth_type_mappings"].get(
                auth_req.get("type", "none"), "none"
            )
            return {
                mapped_type: {
                    "inferred": {
                        "type": mapped_type,
                        "required": auth_req.get("required", False),
                        "scopes": auth_req.get("scope", "").split(",") if auth_req.get("scope") else []
                    }
                }
            }
        except Exception as e:
            logger.warning("Griffe LLM auth inference failed", error=str(e))
            return {}

class SpecAnalyzer:
    """Main class for specification analysis with optimized parsing."""
    def __init__(self, llm_client=None, config=None):
        """Initialize analyzer with config-driven parsing."""
        if config is None:
            default = Path(__file__).parent.parent.parent / "config.yaml"
            path = Path(os.environ.get("CONFIG_PATH", default))
            with open(path, "r") as f:
                config = yaml.safe_load(f)  # Fix Error 13

        self.config = ConfigModel.from_config(config).model_dump()
        self.logger = configure_structlog(self.config)

        llm_cfg = self.config["llm_client"]
        api_key = os.environ.get("LLM_API_KEY", llm_cfg.get("api_key", ""))
        if api_key and "LLM_API_KEY" not in os.environ:
            logger.warning("API key loaded from config; use LLM_API_KEY environment variable for security")
        if not api_key:
            logger.warning("No LLM API key provided; some features may be limited")

        if llm_client is None:
            llm_client = EnhancedAsyncLLMClient(
                config=self.config,
                endpoint_url=llm_cfg.get("endpoint", ""),
                headers={"Authorization": f"Bearer {api_key}"}
            )

        self.enricher = SemanticEnricher(llm_client, self.config)
        out_cfg = self.config["output"]
        out_cfg["output_dir"] = os.path.abspath(out_cfg["output_dir"])
        config = self.config.copy()
        config["output"] = out_cfg
        self.output_gen = EnhancedOutputGenerator(config)  # Fix Error 14

        self.parsers = {
            "openapi": EnhancedOpenAPIParser(self.config),
            "rest": RESTParser(self.config),
            "postman": PostmanParser(self.config),
            "scraping": ScrapingParser(self.config)
        }

    async def analyze(self, spec_path: str, dry_run: bool = False) -> None:
        """Analyze specification with optimized parsing pipeline."""
        raw_data = {}
        for p_type in self.config["parsers_order"]:
            try:
                raw_data = await self.parsers[p_type].parse(spec_path)  # Now await async parse methods
                if raw_data.get("intents"):
                    logger.info("Parsed successfully", parser_type=p_type, intents_count=len(raw_data["intents"]))
                    break
            except ParserError as e:
                logger.warning("Parser failed, trying next", parser=p_type, error=str(e))

        if not raw_data.get("intents"):
            raise ValueError(f"No intents parsed from {spec_path}")

        valid_intents = []
        for intent in raw_data["intents"]:
            try:
                valid_intents.append(EnhancedIntentModel(config=self.config, **intent).model_dump())  # Fix Error 15
            except pydantic.ValidationError as ve:
                logger.error("Intent validation failed", error=str(ve), intent_path=intent.get("path"))

        if not valid_intents:
            raise ValueError("No valid intents after validation")

        if dry_run:
            logger.info("Dry run mode: Skipping enrichment and output")
            return

        retry_attempts = self.config["error_handling"].get("max_retries", 3)
        for attempt in range(retry_attempts):
            try:
                if self.config["profile_enrichment"]:
                    pr = cProfile.Profile()
                    pr.enable()
                    t0 = time.time()

                    enriched_intents = await self.enricher.enrich_intents(valid_intents)  # Fix Errors 16–18
                    capabilities = await self.enricher.generate_capabilities(valid_intents)  # Fix Errors 19–20
                    mcp_tools = await self.enricher.generate_mcp_tools(enriched_intents)  # Fix Error 21
                    await self.output_gen.save_outputs(enriched_intents, capabilities, mcp_tools)

                    pr.disable()
                    s = io.StringIO()
                    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
                    ps.print_stats(20)
                    t1 = time.time()
                    logger.info("Profiling results", seconds=t1 - t0, profile=s.getvalue())
                else:
                    enriched_intents = await self.enricher.enrich_intents(valid_intents)
                    capabilities = await self.enricher.generate_capabilities(valid_intents)
                    mcp_tools = await self.enricher.generate_mcp_tools(enriched_intents)
                    await self.output_gen.save_outputs(enriched_intents, capabilities, mcp_tools)

                logger.info("Analysis completed", enriched_intents=len(enriched_intents), mcp_tools=len(mcp_tools))
                break

            except Exception as e:
                logger.warning("Enrichment attempt failed, retrying", attempt=attempt + 1, error=str(e))
                if attempt == retry_attempts - 1:
                    raise ValueError(f"Enrichment failed after {retry_attempts} attempts: {str(e)}") from e