# output_generator.py - Optimized Domain-Agnostic Output Generator
# =============================================================================
# Saves enriched intents, capabilities, and MCP tools with validation and LLM-driven quality assessment.
# Key Features:
# - Fully config-driven: All validation rules, thresholds, and patterns from config.yaml.
# - Async pipeline for scalability, fixing parsers.py Errors 22–24.
# - LLM-driven quality assessment with structured explanations (score, grade, issues, reasoning).
# - Client-type-specific validation for MCP clients (langchain, openai, etc.).
# - Comprehensive PII scrubbing with scrubadub and regex.
# - Robust error handling, versioning, and quality reporting.
# Critical Requirements:
# - Industry-standard: Pydantic validation, JSON/YAML output, async operations.
# - Production-ready: Async methods, metrics via LLM client, detailed logging.
# - Secure: PII scrubbing with config patterns.
# - Rich: LLM quality assessment, client-specific validation, quality reports.
# - Robust: Handles validation errors, empty inputs, async scalability.
# - No hardcoded logic: Config-driven rules and thresholds.
# - MCP Support: Validates client-specific fields (prompts, parameters).
# Assumptions:
# - config.yaml: Sections for output (output_dir, save_format, versioning, strict_validation, quality_assessment with sampling_rate), semantic_transformation (llm_prompts, quality_thresholds with capability_name_min_length, required_intent_details, complexity_levels, user_contexts, permission_levels, sensitive_data_patterns with high/medium/low_sensitivity), mcp (tool_template, safety_mapping, headers_generation), validation (semantic_name_pattern, valid_http_methods).
# - Pydantic Version: v2.x (field_validator, ValidationInfo). For v1.x, adjust to @validator.
# - LLM Client: Optional llm_client for quality assessment; falls back to rule-based if None.
# =============================================================================

# Standard library
import json
import copy
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Awaitable
import re
import random

# Third-party
import pydantic
import scrubadub
import structlog
import yaml
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# Local
from .llm_client_interface import LLMClientInterface, ResponseFormat

def create_logger(config: Dict[str, Any]) -> structlog.BoundLogger:
    """Create and configure structlog logger."""
    log_format = config["logging"].get("format", "console")
    renderer = {
        "console": structlog.dev.ConsoleRenderer(colors=True),
        "json": structlog.processors.JSONRenderer(),
        "key_value": structlog.processors.KeyValueRenderer()
    }.get(log_format, structlog.dev.ConsoleRenderer(colors=True))

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

class EnhancedIntentModel(BaseModel):
    """Pydantic model for validating enriched semantic intents."""
    name: str = Field(...)
    details: Dict[str, Any] = Field(...)
    paths: List[Dict[str, str]] = Field(..., min_length=1)
    requirements: Dict[str, Any] = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    enriched_by: Optional[str] = Field(default="semantic_llm")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def with_config(cls, config: Dict[str, Any]):
        """Create model with configuration context."""
        cls._config = config
        return cls

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        """Validate name using config."""
        config = info.context.get("config", {}) if info.context else {}
        pattern = config.get("validation", {}).get("semantic_name_pattern", r'^[a-z][a-z0-9_]*[a-z0-9]$')
        min_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("semantic_name_min_length", 3)
        max_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("semantic_name_max_length", 100)
        
        if not re.match(pattern, v):
            raise ValueError(f"Name does not match pattern: {pattern}")
        if not (min_length <= len(v) <= max_length):
            raise ValueError(f"Name length must be {min_length}-{max_length} characters")
        if v in config.get("validation", {}).get("semantic_name_reserved_words", []):
            raise ValueError(f"Name is a reserved word: {v}")
        return v

    @field_validator("details")
    @classmethod
    def validate_details(cls, v: Dict, info: ValidationInfo) -> Dict:
        """Validate details structure."""
        config = info.context.get("config", {}) if info.context else {}
        required_keys = config.get("semantic_transformation", {}).get("required_intent_details", ["summary", "complexity", "user_context"])
        valid_complexity = [level["name"] for level in config.get("semantic_transformation", {}).get("complexity_levels", [])]
        valid_contexts = [ctx["name"] for ctx in config.get("semantic_transformation", {}).get("user_contexts", [])]
        
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required detail: {key}")
        
        if valid_complexity and v.get("complexity") not in valid_complexity:
            raise ValueError(f"Invalid complexity: {v.get('complexity')}. Must be one of {valid_complexity}")
        
        if valid_contexts and v.get("user_context") not in valid_contexts:
            raise ValueError(f"Invalid user_context: {v.get('user_context')}. Must be one of {valid_contexts}")
        
        summary = v.get("summary", "")
        min_summary_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("summary_min_length", 10)
        if not isinstance(summary, str) or len(summary) < min_summary_length:
            raise ValueError(f"Summary must be at least {min_summary_length} characters")
        
        return v

    @field_validator("requirements")
    @classmethod
    def validate_requirements(cls, v: Dict, info: ValidationInfo) -> Dict:
        """Validate requirements structure."""
        config = info.context.get("config", {}) if info.context else {}
        valid_levels = list(config.get("semantic_transformation", {}).get("permission_levels", {}).keys())
        
        if "authentication" in v:
            if not isinstance(v["authentication"], list):
                raise ValueError("Authentication must be a list")
            for auth_req in v["authentication"]:
                if not isinstance(auth_req, dict) or "type" not in auth_req:
                    raise ValueError("Authentication requirement must have 'type' field")
        
        if "permissions" in v:
            if not isinstance(v["permissions"], list):
                raise ValueError("Permissions must be a list")
            for perm in v["permissions"]:
                if not isinstance(perm, dict) or "level" not in perm or "action" not in perm:
                    raise ValueError("Permission must have 'level' and 'action' fields")
                if valid_levels and perm.get("level") not in valid_levels:
                    raise ValueError(f"Invalid permission level: {perm.get('level')}. Must be one of {valid_levels}")
        
        return v

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, v: List[Dict], info: ValidationInfo) -> List[Dict]:
        """Validate paths structure."""
        config = info.context.get("config", {}) if info.context else {}
        valid_methods = config.get("validation", {}).get("valid_http_methods", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
        
        for path in v:
            if not isinstance(path, dict) or "method" not in path or "endpoint" not in path:
                raise ValueError("Path must have 'method' and 'endpoint' fields")
            if path.get("method", "").upper() not in valid_methods:
                raise ValueError(f"Invalid HTTP method: {path.get('method')}. Must be one of {valid_methods}")
            if not path.get("endpoint", "").startswith("/"):
                raise ValueError("Endpoint must start with '/'")
        
        return v

class EnhancedCapabilitiesModel(BaseModel):
    """Pydantic model for validating capabilities."""
    class Config:
        extra = "allow"  # Allow dynamic permission levels

    @classmethod
    def with_config(cls, config: Dict[str, Any]):
        """Create model with configuration context."""
        cls._config = config
        return cls

    @pydantic.model_validator(mode='after')
    def validate_capabilities(self):
        """Validate capability names."""
        config = getattr(self.__class__, '_config', {})
        min_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("capability_name_min_length", 3)
        max_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("capability_name_max_length", 50)
        pattern = config.get("validation", {}).get("semantic_name_pattern", r'^[a-z][a-z0-9_]*[a-z0-9]$')
        
        all_capabilities = set()
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, list):
                for name in field_value:
                    if not isinstance(name, str):
                        raise ValueError(f"Capability names must be strings in {field_name}")
                    if not re.match(pattern, name):
                        raise ValueError(f"Invalid capability name format: {name}. Must match pattern: {pattern}")
                    if not (min_length <= len(name) <= max_length):
                        raise ValueError(f"Capability name length must be {min_length}-{max_length} characters: {name}")
                    if name in all_capabilities:
                        raise ValueError(f"Duplicate capability across categories: {name}")
                    all_capabilities.add(name)
        return self

class EnhancedMCPToolModel(BaseModel):
    """Pydantic model for validating MCP tool metadata."""
    name: str = Field(...)
    description: str = Field(...)
    method: str = Field(...)
    path: str = Field(..., pattern=r'^/')
    safety_level: str = Field(...)
    input_schema: Dict[str, Any] = Field(...)
    # FIX: Restored correct output_schema definition (fixes masked_field=True typo)
    output_schema: Dict[str, Any] = Field(..., exclude=True)  # Exclude for serialization to mask sensitive data
    prompt_template: Optional[str] = Field(default=None)
    examples: Optional[List[str]] = Field(default_factory=list)
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    transport: Optional[str] = Field(default=None)  # For fastmcp
    parameters: Optional[List[Dict]] = Field(default=None)  # For openai, autogen
    resource: Optional[str] = Field(default=None)  # For llamaindex

    @classmethod
    def with_config(cls, config: Dict[str, Any]):
        """Create model with configuration context."""
        cls._config = config
        return cls

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        """Validate name."""
        config = info.context.get("config", {}) if info.context else {}
        pattern = config.get("validation", {}).get("semantic_name_pattern", r'^[a-z][a-z0-9_]*[a-z0-9]$')
        min_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("semantic_name_min_length", 3)
        max_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("semantic_name_max_length", 100)
        
        if not re.match(pattern, v):
            raise ValueError(f"Name does not match pattern: {pattern}")
        if not (min_length <= len(v) <= max_length):
            raise ValueError(f"Name length must be {min_length}-{max_length} characters")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str, info: ValidationInfo) -> str:
        """Validate description."""
        config = info.context.get("config", {}) if info.context else {}
        min_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("description_min_length", 20)
        max_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("description_max_length", 500)
        
        if not (min_length <= len(v) <= max_length):
            raise ValueError(f"Description length must be between {min_length} and {max_length} characters")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str, info: ValidationInfo) -> str:
        """Validate HTTP method."""
        config = info.context.get("config", {}) if info.context else {}
        valid_methods = config.get("validation", {}).get("valid_http_methods", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
        v = v.upper()
        if v not in valid_methods:
            raise ValueError(f"Invalid HTTP method: {v}. Must be one of {valid_methods}")
        return v

    @field_validator("safety_level")
    @classmethod
    def validate_safety_level(cls, v: str, info: ValidationInfo) -> str:
        """Validate safety level."""
        config = info.context.get("config", {}) if info.context else {}
        valid_levels = list(config.get("mcp", {}).get("safety_mapping", {}).values()) + list(config.get("semantic_transformation", {}).get("permission_levels", {}).keys())
        if v not in valid_levels:
            raise ValueError(f"Invalid safety level: {v}. Must be one of {valid_levels}")
        return v

    @field_validator("input_schema", "output_schema")
    @classmethod
    def validate_json_schema(cls, v: Dict) -> Dict:
        """Validate JSON Schema structures."""
        if not isinstance(v, dict) or "type" not in v:
            raise ValueError("Schema must be a dictionary with 'type' field")
        valid_types = ["object", "array", "string", "number", "integer", "boolean", "null"]
        if v.get("type") not in valid_types:
            raise ValueError(f"Invalid schema type: {v.get('type')}. Must be one of {valid_types}")
        if v.get("type") == "object" and "properties" not in v:
            raise ValueError("Object schemas should have 'properties' field")
        return v

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, v: Optional[List[str]], info: ValidationInfo) -> List[str]:
        """Validate examples."""
        config = info.context.get("config", {}) if info.context else {}
        max_examples = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("example_max_count", 5)
        min_length = config.get("semantic_transformation", {}).get("quality_thresholds", {}).get("example_min_length", 5)
        
        if v is None:
            return []
        for i, example in enumerate(v):
            if not isinstance(example, str):
                raise ValueError(f"Example {i} must be a string")
            if len(example) < min_length:
                raise ValueError(f"Example {i} too short: {example}")
            if not re.search(r'[a-z_]+\(.*\)', example):
                raise ValueError(f"Example {i} should be a function call format: {example}")
        return v[:max_examples]

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, v: Optional[Dict], info: ValidationInfo) -> Dict:
        """Validate headers."""
        config = info.context.get("config", {}) if info.context else {}
        default_content_type = config.get("mcp", {}).get("headers_generation", {}).get("default_content_type", "application/json")
        
        if v is None:
            return {"Content-Type": default_content_type}
        if "Content-Type" not in v:
            v = v.copy()
            v["Content-Type"] = default_content_type
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Headers must be string key-value pairs")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Optional[List[Dict]], info: ValidationInfo) -> Optional[List[Dict]]:
        """Validate parameters for openai/autogen."""
        if v is None:
            return None
        for i, param in enumerate(v):
            if not isinstance(param, dict) or "name" not in param or "type" not in param:
                raise ValueError(f"Parameter {i} must be a dict with 'name' and 'type' fields")
        return v

class QualityAssessment:
    """LLM-driven quality assessment for outputs."""
    def __init__(self, config: Dict[str, Any], llm_client: Optional[LLMClientInterface] = None):
        self.config = config
        self.llm_client = llm_client
        self.quality_config = config.get("output", {}).get("quality_assessment", {})
        self.semantic_config = config.get("semantic_transformation", {})
        self.logger = create_logger(config)

    async def assess_intent_quality(self, intent: Dict) -> Dict:
        """Assess intent quality using LLM or fallback rules."""
        if self.llm_client and self.quality_config.get("use_llm_assessment", False):
            return await self._llm_assess_intent_quality(intent)
        return await self._rule_based_intent_quality(intent)

    async def _llm_assess_intent_quality(self, intent: Dict) -> Dict:
        """Use LLM to assess intent quality."""
        prompt = self.semantic_config.get("llm_prompts", {}).get("intent_quality_assessment", """
        Assess the quality of this enriched intent based on these requirements:
        
        Requirements:
        - Semantic name: Descriptive, snake_case, {min_name_length}-100 chars
        - Summary: Clear, at least {min_summary_length} chars
        - Confidence: At least {min_confidence}
        - Details: Must include {required_details}
        - Permissions: Valid levels in {permission_levels}
        
        Intent:
        {intent_json}
        
        Return a JSON object:
        {{
            "score": float (0-100),
            "grade": str (A/B/C/D/F),
            "issues": list[str],
            "reasoning": {{
                "semantic_name": str,
                "summary": str,
                "confidence": str,
                "completeness": str,
                "context": str
            }},
            "acceptable": bool
        }}
        """).format(
            min_name_length=self.quality_config.get("semantic_name_min_length", 3),
            min_summary_length=self.quality_config.get("summary_min_length", 10),
            min_confidence=self.quality_config.get("min_confidence", 0.6),
            required_details=', '.join(self.semantic_config.get("required_intent_details", ["summary", "complexity", "user_context"])),
            permission_levels=', '.join(self.semantic_config.get("permission_levels", {}).keys()),
            intent_json=json.dumps(intent, indent=2)
        )
        expected_schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "grade": {"type": "string", "enum": ["A", "B", "C", "D", "F"]},
                "issues": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "object"},
                "acceptable": {"type": "boolean"}
            },
            "required": ["score", "grade", "issues", "reasoning", "acceptable"]
        }
        
        try:
            if self.llm_client is None:
                self.logger.warning("LLM client not available, falling back to rule-based assessment")
                return await self._rule_based_intent_quality(intent)
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            return {
                "score": float(response["score"]),
                "grade": response["grade"],
                "issues": response["issues"],
                "reasoning": response["reasoning"],
                "acceptable": response["acceptable"]
            }
        except Exception as e:
            self.logger.error("LLM intent quality assessment failed", error=str(e))
            return await self._rule_based_intent_quality(intent)

    async def _rule_based_intent_quality(self, intent: Dict) -> Dict:
        """Rule-based intent quality assessment."""
        quality_score = 0.0
        issues = []
        reasoning = {}
        quality_thresholds = self.semantic_config.get("quality_thresholds", {})
        
        # Semantic name (25 points)
        name = intent.get("name", "")
        pattern = self.config.get("validation", {}).get("semantic_name_pattern", r'^[a-z][a-z0-9_]*[a-z0-9]$')
        min_name_length = quality_thresholds.get("semantic_name_min_length", 3)
        
        if len(name) >= min_name_length and re.match(pattern, name):
            quality_score += 25
            reasoning["semantic_name"] = "Descriptive and follows snake_case"
        else:
            issues.append("Poor semantic name quality")
            reasoning["semantic_name"] = "Name does not follow conventions"
        
        # Summary (25 points)
        summary = intent.get("details", {}).get("summary", "")
        min_summary_length = quality_thresholds.get("summary_min_length", 10)
        
        if len(summary) >= min_summary_length * 2:
            quality_score += 25
            reasoning["summary"] = "Comprehensive and descriptive summary"
        elif len(summary) >= min_summary_length:
            quality_score += 15
            issues.append("Summary could be more descriptive")
            reasoning["summary"] = "Basic summary provided"
        else:
            issues.append("Poor summary quality")
            reasoning["summary"] = "Summary too short or missing"
        
        # Confidence (20 points)
        confidence = intent.get("confidence", 0)
        min_confidence = quality_thresholds.get("min_confidence", 0.6)
        
        if confidence >= 0.8:
            quality_score += 20
            reasoning["confidence"] = "High confidence transformation"
        elif confidence >= min_confidence:
            quality_score += 15
            issues.append("Medium confidence level")
            reasoning["confidence"] = "Moderate confidence in transformation"
        else:
            quality_score += 5
            issues.append("Low confidence level")
            reasoning["confidence"] = "Low confidence suggests uncertain transformation"
        
        # Requirements (15 points)
        requirements = intent.get("requirements", {})
        if "authentication" in requirements and "permissions" in requirements:
            quality_score += 15
            reasoning["completeness"] = "All required fields present"
        elif "authentication" in requirements or "permissions" in requirements:
            quality_score += 10
            issues.append("Incomplete requirements")
            reasoning["completeness"] = "Some required fields missing"
        else:
            issues.append("Missing requirements")
            reasoning["completeness"] = "Critical requirements fields missing"
        
        # Context (15 points)
        details = intent.get("details", {})
        required_details = self.semantic_config.get("required_intent_details", ["complexity", "user_context"])
        if all(key in details for key in required_details):
            quality_score += 15
            reasoning["context"] = "Rich context information provided"
        else:
            issues.append("Missing context information")
            reasoning["context"] = "Context information incomplete"
        
        return {
            "score": quality_score,
            "grade": self._score_to_grade(quality_score),
            "issues": issues,
            "reasoning": reasoning,
            "acceptable": quality_score >= self.quality_config.get("min_acceptable_score", 70)
        }

    async def assess_mcp_tool_quality(self, tool: Dict) -> Dict:
        """Assess MCP tool quality using LLM or fallback rules."""
        if self.llm_client and self.quality_config.get("use_llm_assessment", False):
            return await self._llm_assess_mcp_tool_quality(tool)
        return await self._rule_based_mcp_tool_quality(tool)

    async def _llm_assess_mcp_tool_quality(self, tool: Dict) -> Dict:
        """Use LLM to assess MCP tool quality."""
        prompt = self.semantic_config.get("llm_prompts", {}).get("mcp_tool_quality_assessment", """
        Assess the quality of this MCP tool specification based on these requirements:
        
        Requirements:
        - Name: Descriptive, snake_case, {min_name_length}-100 chars
        - Description: Clear, {min_desc_length}-{max_desc_length} chars
        - Schemas: Complete input/output schemas
        - Examples: At least {min_examples} examples, each {min_example_length} chars
        - Headers: Include necessary authentication
        - Client-specific fields: Valid for client type {client_type}
        
        Tool:
        {tool_json}
        
        Return a JSON object:
        {{
            "score": float (0-100),
            "grade": str (A/B/C/D/F),
            "issues": list[str],
            "reasoning": {{
                "naming": str,
                "description": str,
                "schemas": str,
                "examples": str,
                "headers": str,
                "client_specific": str
            }},
            "acceptable": bool
        }}
        """).format(
            min_name_length=self.quality_config.get("semantic_name_min_length", 3),
            min_desc_length=self.quality_config.get("description_min_length", 20),
            max_desc_length=self.quality_config.get("description_max_length", 500),
            min_examples=self.quality_config.get("example_min_count", 1),
            min_example_length=self.quality_config.get("example_min_length", 5),
            client_type=tool.get("metadata", {}).get("client_type", "generic"),
            tool_json=json.dumps(tool, indent=2)
        )
        expected_schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "grade": {"type": "string", "enum": ["A", "B", "C", "D", "F"]},
                "issues": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "object"},
                "acceptable": {"type": "boolean"}
            },
            "required": ["score", "grade", "issues", "reasoning", "acceptable"]
        }
        
        try:
            if self.llm_client is None:
                self.logger.warning("LLM client not available, falling back to rule-based assessment")
                return await self._rule_based_mcp_tool_quality(tool)
            response = await self.llm_client.query_semantic(prompt, expected_schema=expected_schema, **self.semantic_config.get("llm_settings", {}))
            return {
                "score": float(response["score"]),
                "grade": response["grade"],
                "issues": response["issues"],
                "reasoning": response["reasoning"],
                "acceptable": response["acceptable"]
            }
        except Exception as e:
            self.logger.error("LLM tool quality assessment failed", error=str(e))
            return await self._rule_based_mcp_tool_quality(tool)

    async def _rule_based_mcp_tool_quality(self, tool: Dict) -> Dict:
        """Rule-based MCP tool quality assessment."""
        quality_score = 0.0
        issues = []
        reasoning = {}
        quality_thresholds = self.semantic_config.get("quality_thresholds", {})
        
        # Name (15 points)
        name = tool.get("name", "")
        pattern = self.config.get("validation", {}).get("semantic_name_pattern", r'^[a-z][a-z0-9_]*[a-z0-9]$')
        min_name_length = quality_thresholds.get("semantic_name_min_length", 3)
        
        if len(name) >= min_name_length and re.match(pattern, name):
            quality_score += 15
            reasoning["naming"] = "Descriptive tool name"
        else:
            issues.append("Poor tool name quality")
            reasoning["naming"] = "Name does not follow conventions"
        
        # Description (20 points)
        description = tool.get("description", "")
        min_desc_length = quality_thresholds.get("description_min_length", 20)
        if len(description) >= min_desc_length * 2.5:
            quality_score += 20
            reasoning["description"] = "Comprehensive tool description"
        elif len(description) >= min_desc_length:
            quality_score += 15
            issues.append("Description could be more detailed")
            reasoning["description"] = "Basic description provided"
        else:
            issues.append("Poor description quality")
            reasoning["description"] = "Description too short or missing"
        
        # Schemas (40 points total)
        input_schema = tool.get("input_schema", {})
        if await self._assess_schema_quality(input_schema):
            quality_score += 20
            reasoning["schemas"] = "Well-structured input schema"
        else:
            quality_score += 10
            issues.append("Basic input schema")
            reasoning["schemas"] = "Input schema could be more detailed"
        
        output_schema = tool.get("output_schema", {})
        if await self._assess_schema_quality(output_schema):
            quality_score += 20
            reasoning["schemas"] += "; Well-structured output schema"
        else:
            quality_score += 10
            issues.append("Basic output schema")
            reasoning["schemas"] += "; Output schema could be more detailed"
        
        # Examples (15 points)
        examples = tool.get("examples", [])
        min_examples = quality_thresholds.get("example_min_count", 1)
        min_example_length = quality_thresholds.get("example_min_length", 5)
        if len(examples) >= min_examples * 2 and all(len(ex) >= min_example_length for ex in examples):
            quality_score += 15
            reasoning["examples"] = "Good examples provided"
        elif len(examples) >= min_examples:
            quality_score += 8
            issues.append("Could use more examples")
            reasoning["examples"] = "Limited examples provided"
        else:
            issues.append("Missing examples")
            reasoning["examples"] = "No examples provided"
        
        # Headers (10 points)
        headers = tool.get("headers", {})
        if headers and "Content-Type" in headers:
            quality_score += 10
            reasoning["headers"] = "Appropriate headers configured"
        else:
            quality_score += 5
            issues.append("Missing or incomplete headers")
            reasoning["headers"] = "Headers configuration incomplete"
        
        # Client-specific fields
        client_type = tool.get("metadata", {}).get("client_type", "generic")
        if client_type == "langchain" and not tool.get("prompt_template"):
            issues.append("Missing prompt_template for langchain")
            reasoning["client_specific"] = "Missing required langchain fields"
        elif client_type in ["openai", "autogen"] and not tool.get("parameters"):
            issues.append(f"Missing parameters for {client_type}")
            reasoning["client_specific"] = f"Missing required {client_type} fields"
        else:
            reasoning["client_specific"] = f"Client-specific fields valid for {client_type}"
            quality_score += 5
        
        return {
            "score": quality_score,
            "grade": self._score_to_grade(quality_score),
            "issues": issues,
            "reasoning": reasoning,
            "acceptable": quality_score >= self.quality_config.get("min_acceptable_score", 70)
        }

    async def _assess_schema_quality(self, schema: Dict) -> bool:
        """Assess if a JSON schema is well-structured."""
        if not isinstance(schema, dict) or "type" not in schema:
            return False
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            return len(properties) > 0 and any("description" in prop for prop in properties.values() if isinstance(prop, dict))
        return True

    def _score_to_grade(self, score: float) -> str:
        """Convert score to grade."""
        grading_thresholds = self.quality_config.get("grading_thresholds", {"A": 90, "B": 80, "C": 70, "D": 60})
        for grade, threshold in sorted(grading_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return grade
        return "F"

class EnhancedOutputGenerator:
    """Output generator with validation and LLM-driven quality assurance."""
    def __init__(self, config: Dict[str, Any], llm_client: Optional[LLMClientInterface] = None):
        """Initialize with config.yaml and optional LLM client."""
        if "output" not in config:
            raise ValueError("Missing 'output' section in config")
        
        self.config = config
        self.llm_client = llm_client
        self.logger = create_logger(config)
        self.output_dir = Path(config["output"].get("output_dir", "outputs")).absolute()
        self.output_dir.mkdir(exist_ok=True)
        
        # Add missing attributes
        self.semantic_config = config.get("semantic_transformation", {})
        self.quality_config = config.get("output", {}).get("quality_assessment", {})
        
        self.quality_assessor = QualityAssessment(config, llm_client)
        self.generation_stats = {
            "intents_processed": 0,
            "intents_valid": 0,
            "mcp_tools_generated": 0,
            "mcp_tools_valid": 0,
            "quality_issues": [],
            "validation_errors": []
        }
        
        self.IntentModel = EnhancedIntentModel.with_config(config)
        self.CapabilitiesModel = EnhancedCapabilitiesModel.with_config(config)
        self.MCPToolModel = EnhancedMCPToolModel.with_config(config)
        
        self.logger.info("Enhanced output generator initialized", output_dir=str(self.output_dir))

    async def save_outputs(self, intents: List[Dict], capabilities: Dict[str, List[str]], mcp_tools: Optional[List[Dict]] = None) -> None:
        """Async save outputs with validation and quality assessment."""
        start_time = datetime.now()
        self.logger.info("Starting enhanced output generation",
                        intents_count=len(intents),
                        capabilities_categories=len(capabilities),
                        mcp_tools_count=len(mcp_tools) if mcp_tools else 0)
        
        self.generation_stats = {
            "intents_processed": 0,
            "intents_valid": 0,
            "mcp_tools_generated": 0,
            "mcp_tools_valid": 0,
            "quality_issues": [],
            "validation_errors": []
        }
        
        validation_results = await self._comprehensive_validation(intents, capabilities, mcp_tools)
        if not validation_results["proceed"]:
            if self.config["output"].get("strict_validation", True):
                raise ValueError(f"Validation failed: {validation_results['errors']}")
            self.logger.warning("Validation issues found, proceeding with warnings", errors=validation_results["errors"])
        
        quality_results = await self._assess_output_quality(intents, capabilities, mcp_tools)
        processed_data = await self._process_and_scrub_data(intents, capabilities, mcp_tools)
        timestamp = await self._generate_timestamp()
        
        saved_files = []
        if processed_data["intents"]:
            intents_file = await self._save_intents(processed_data["intents"], timestamp, quality_results["intents"])
            saved_files.append(intents_file)
        
        if processed_data["capabilities"]:
            capabilities_file = await self._save_capabilities(processed_data["capabilities"], timestamp, quality_results["capabilities"])
            saved_files.append(capabilities_file)
        
        if processed_data["mcp_tools"]:
            mcp_file = await self._save_mcp_tools(processed_data["mcp_tools"], timestamp, quality_results["mcp_tools"])
            saved_files.append(mcp_file)
        
        if self.config["output"].get("quality_assessment", {}).get("generate_report", True):
            quality_report_file = await self._save_quality_report(quality_results, timestamp)
            saved_files.append(quality_report_file)
        
        generation_time = datetime.now() - start_time
        await self._save_generation_summary(saved_files, generation_time.total_seconds(), timestamp)
        
        self.logger.info("Enhanced output generation completed",
                        files_saved=len(saved_files),
                        generation_time=generation_time.total_seconds(),
                        quality_grade=quality_results["overall"]["grade"])

    async def _comprehensive_validation(self, intents: List[Dict], capabilities: Dict[str, List[str]], mcp_tools: Optional[List[Dict]]) -> Dict:
        """Async validation with detailed error reporting."""
        errors = []
        warnings = []
        
        if intents:
            for i, intent in enumerate(intents):
                try:
                    self.IntentModel.model_validate(intent, context={"config": self.config})
                    self.generation_stats["intents_valid"] += 1
                except pydantic.ValidationError as e:
                    error_msg = f"Intent {i} validation failed: {e}"
                    errors.append(error_msg)
                    self.generation_stats["validation_errors"].append(error_msg)
                self.generation_stats["intents_processed"] += 1
        
        if capabilities:
            try:
                self.CapabilitiesModel.model_validate(capabilities, context={"config": self.config})
            except pydantic.ValidationError as e:
                error_msg = f"Capabilities validation failed: {e}"
                errors.append(error_msg)
                self.generation_stats["validation_errors"].append(error_msg)
        
        if mcp_tools:
            for i, tool in enumerate(mcp_tools):
                try:
                    self.MCPToolModel.model_validate(tool, context={"config": self.config})
                    self.generation_stats["mcp_tools_valid"] += 1
                except pydantic.ValidationError as e:
                    error_msg = f"MCP tool {i} validation failed: {e}"
                    errors.append(error_msg)
                    self.generation_stats["validation_errors"].append(error_msg)
                self.generation_stats["mcp_tools_generated"] += 1
        
        proceed = not errors or not self.config["output"].get("strict_validation", True)
        if not intents and not capabilities and not mcp_tools:
            errors.append("All inputs are empty")
            proceed = False
        
        return {
            "proceed": proceed,
            "errors": errors,
            "warnings": warnings,
            "stats": self.generation_stats.copy()
        }

    async def _assess_output_quality(self, intents: List[Dict], capabilities: Dict[str, List[str]], mcp_tools: Optional[List[Dict]]) -> Dict:
        """Assess quality of all outputs asynchronously."""
        sampling_rate = self.config.get("output", {}).get("quality_assessment", {}).get("sampling_rate", 1.0)
        quality_results = {
            "intents": {"assessments": [], "avg_score": 0, "grade": "N/A"},
            "capabilities": {"assessment": {}, "grade": "N/A"},
            "mcp_tools": {"assessments": [], "avg_score": 0, "grade": "N/A"},
            "overall": {"score": 0, "grade": "N/A"}
        }
        
        if intents:
            sample_size = max(1, int(len(intents) * sampling_rate))
            sampled_intents = random.sample(intents, min(len(intents), sample_size))
            intent_tasks = [self.quality_assessor.assess_intent_quality(intent) for intent in sampled_intents]
            intent_assessments = await asyncio.gather(*intent_tasks, return_exceptions=True)
            quality_results["intents"]["assessments"] = []
            for a in intent_assessments:
                if isinstance(a, Exception):
                    fallback_assessment = {"score": 0, "grade": "F", "issues": [str(a)], "reasoning": {}, "acceptable": False}
                    quality_results["intents"]["assessments"].append(fallback_assessment)
                else:
                    quality_results["intents"]["assessments"].append(a)
            
            for assessment in quality_results["intents"]["assessments"]:
                if not assessment["acceptable"]:
                    self.generation_stats["quality_issues"].extend(assessment["issues"])
            if quality_results["intents"]["assessments"]:
                avg_score = sum(a["score"] for a in quality_results["intents"]["assessments"]) / len(quality_results["intents"]["assessments"])
                quality_results["intents"]["avg_score"] = avg_score
                quality_results["intents"]["grade"] = self._score_to_grade(avg_score)
        
        if capabilities:
            total_capabilities = sum(len(caps) for caps in capabilities.values())
            unique_capabilities = len(set().union(*capabilities.values()))
            capabilities_score = 100 if unique_capabilities == total_capabilities else 80
            quality_results["capabilities"]["assessment"] = {
                "total_capabilities": total_capabilities,
                "unique_capabilities": unique_capabilities,
                "has_duplicates": unique_capabilities != total_capabilities,
                "score": capabilities_score,
                "reasoning": {"duplicates": "No duplicates found" if unique_capabilities == total_capabilities else "Duplicates detected"}
            }
            quality_results["capabilities"]["grade"] = self._score_to_grade(capabilities_score)
        
        if mcp_tools:
            sample_size = max(1, int(len(mcp_tools) * sampling_rate))
            sampled_tools = random.sample(mcp_tools, min(len(mcp_tools), sample_size))
            tool_tasks = [self.quality_assessor.assess_mcp_tool_quality(tool) for tool in sampled_tools]
            tool_assessments = await asyncio.gather(*tool_tasks, return_exceptions=True)
            quality_results["mcp_tools"]["assessments"] = []
            for a in tool_assessments:
                if isinstance(a, Exception):
                    fallback_assessment = {"score": 0, "grade": "F", "issues": [str(a)], "reasoning": {}, "acceptable": False}
                    quality_results["mcp_tools"]["assessments"].append(fallback_assessment)
                else:
                    quality_results["mcp_tools"]["assessments"].append(a)
            
            for assessment in quality_results["mcp_tools"]["assessments"]:
                if not assessment["acceptable"]:
                    self.generation_stats["quality_issues"].extend(assessment["issues"])
            if quality_results["mcp_tools"]["assessments"]:
                avg_score = sum(a["score"] for a in quality_results["mcp_tools"]["assessments"]) / len(quality_results["mcp_tools"]["assessments"])
                quality_results["mcp_tools"]["avg_score"] = avg_score
                quality_results["mcp_tools"]["grade"] = self._score_to_grade(avg_score)
        
        scores = []
        if quality_results["intents"]["avg_score"] > 0:
            scores.append(quality_results["intents"]["avg_score"])
        if quality_results["capabilities"]["assessment"]:
            scores.append(quality_results["capabilities"]["assessment"]["score"])
        if quality_results["mcp_tools"]["avg_score"] > 0:
            scores.append(quality_results["mcp_tools"]["avg_score"])
        
        if scores:
            overall_score = sum(scores) / len(scores)
            quality_results["overall"]["score"] = overall_score
            quality_results["overall"]["grade"] = self._score_to_grade(overall_score)
        
        return quality_results

    async def _process_and_scrub_data(self, intents: List[Dict], capabilities: Dict[str, List[str]], mcp_tools: Optional[List[Dict]]) -> Dict:
        """Async data processing and PII scrubbing."""
        # FIX: Ensured all sensitivity levels are used (high, medium, low) for PII scrubbing
        sensitive_patterns = self.semantic_config.get("sensitive_data_patterns", {})
        all_patterns = (sensitive_patterns.get("high", []) + 
                       sensitive_patterns.get("medium", []) + 
                       sensitive_patterns.get("low", []))
        scrubber = scrubadub.Scrubber()
        
        async def scrub_item(item: Any) -> Any:
            if isinstance(item, dict):
                scrubbed = {}
                for key, value in item.items():
                    scrubbed[key] = await scrub_item(value)
                return scrubbed
            elif isinstance(item, list):
                return [await scrub_item(i) for i in item]
            elif isinstance(item, str):
                cleaned = scrubber.clean(item)
                for pattern in all_patterns:
                    cleaned = re.sub(rf'\b{re.escape(pattern)}\b', "[REDACTED]", cleaned, flags=re.IGNORECASE)
                return cleaned
            return item
        
        scrubbed_intents = [await scrub_item(copy.deepcopy(intent)) for intent in intents or []]
        scrubbed_capabilities = {
            key: [await scrub_item(action) for action in actions]
            for key, actions in (capabilities or {}).items()
        }
        scrubbed_mcp_tools = [await scrub_item(copy.deepcopy(tool)) for tool in mcp_tools or []]
        
        return {
            "intents": scrubbed_intents,
            "capabilities": scrubbed_capabilities,
            "mcp_tools": scrubbed_mcp_tools
        }

    async def _generate_timestamp(self) -> str:
        """Generate timestamp for versioning."""
        return f"_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}" if self.config["output"].get("versioning", False) else ""

    async def _save_intents(self, intents: List[Dict], timestamp: str, quality_info: Dict) -> str:
        """Save intents with metadata."""
        intents_data = {
            "_meta": await self._generate_enhanced_metadata("intents", quality_info),
            "intents": intents
        }
        file_path = await self._save_json_or_yaml(intents_data, f"intents{timestamp}")
        self.logger.info("Saved enhanced intents", path=file_path, count=len(intents), quality_grade=quality_info.get("grade", "N/A"))
        return str(file_path)

    async def _save_capabilities(self, capabilities: Dict[str, List[str]], timestamp: str, quality_info: Dict) -> str:
        """Save capabilities with enhanced format."""
        capabilities_path = self.output_dir / f"capabilities{timestamp}.txt"
        if capabilities_path.exists() and not self.config["output"].get("versioning", False):
            capabilities_path = self.output_dir / f"capabilities_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
            self.logger.warning("Capabilities file exists; appending timestamp", path=capabilities_path)
        
        permission_levels = self.semantic_config.get("permission_levels", {})
        with open(capabilities_path, "w") as f:
            f.write(f"# Generated on {datetime.now().isoformat()}\n")
            f.write(f"# Quality Grade: {quality_info.get('grade', 'N/A')}\n")
            f.write(f"# Total Capabilities: {quality_info.get('assessment', {}).get('total_capabilities', 0)}\n\n")
            for key, actions in capabilities.items():
                description = permission_levels.get(key, {}).get("description", f"{key.upper()}-LISTED")
                f.write(f"{description}:\n")
                f.write(f"{', '.join(sorted(actions))}\n\n" if actions else "(none)\n\n")
        
        self.logger.info("Saved enhanced capabilities", path=capabilities_path, categories=len(capabilities))
        return str(capabilities_path)

    async def _save_mcp_tools(self, mcp_tools: List[Dict], timestamp: str, quality_info: Dict) -> str:
        """Save MCP tools with metadata."""
        if not self.config["mcp"].get("generate_tools", True):
            return ""
        
        mcp_data = {
            "_meta": await self._generate_enhanced_metadata("mcp_tools", quality_info),
            "tools": mcp_tools
        }
        file_path = await self._save_json_or_yaml(mcp_data, f"mcp_tools{timestamp}", use_mcp_format=True)
        self.logger.info("Saved enhanced MCP tools", path=file_path, count=len(mcp_tools), quality_grade=quality_info.get("grade", "N/A"))
        return str(file_path)

    async def _generate_enhanced_metadata(self, output_type: str, quality_info: Dict) -> Dict:
        """Generate enhanced metadata."""
        base_meta = {
            "instructions": "This file is editable. Modify as needed.",
            "schema_version": self.config.get("version", "1.0.0"),
            "generated_at": datetime.now().isoformat(),
            "generator": "Enhanced MCP Tool Generation Framework",
            "output_type": output_type
        }
        
        if self.config["output"].get("editable", True):
            base_meta.update({
                "quality_assessment": {
                    "grade": quality_info.get("grade", "N/A"),
                    "score": quality_info.get("avg_score", 0),
                    "acceptable": quality_info.get("avg_score", 0) >= self.quality_config.get("min_acceptable_score", 70)
                },
                "validation_stats": self.generation_stats,
                "framework_version": self.config.get("version", "1.0.0")
            })
        return base_meta

    async def _save_json_or_yaml(self, data: Dict, filename: str, use_mcp_format: bool = False) -> Path:
        """Save data as JSON or YAML."""
        format_type = self.config["mcp"].get("tool_template", "json") if use_mcp_format else self.config["output"].get("save_format", "json")
        file_path = self.output_dir / f"{filename}.{'json' if format_type == 'json' else 'yaml'}"
        
        if file_path.exists() and not self.config["output"].get("versioning", False):
            file_path = self.output_dir / f"{filename}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.{'json' if format_type == 'json' else 'yaml'}"
            self.logger.warning("File exists; appending timestamp", path=file_path)
        
        with open(file_path, "w") as f:
            if format_type == "json":
                json.dump(data, f, indent=4, ensure_ascii=False)
            else:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        
        return file_path

    async def _save_quality_report(self, quality_results: Dict, timestamp: str) -> str:
        """Save quality assessment report."""
        quality_report = {
            "_meta": {
                "report_type": "quality_assessment",
                "generated_at": datetime.now().isoformat(),
                "framework_version": self.config.get("version", "1.0.0")
            },
            "overall_assessment": quality_results["overall"],
            "detailed_results": quality_results,
            "generation_statistics": self.generation_stats,
            "recommendations": await self._generate_quality_recommendations(quality_results)
        }
        
        report_path = self.output_dir / f"quality_report{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(quality_report, f, indent=4)
        
        self.logger.info("Saved quality assessment report", path=report_path, overall_grade=quality_results["overall"]["grade"])
        return str(report_path)

    async def _generate_quality_recommendations(self, quality_results: Dict) -> List[str]:
        """Generate quality recommendations."""
        recommendations = []
        max_issues = self.quality_config.get("max_issues_threshold", 5)
        
        if quality_results["intents"].get("grade", "N/A") in ["D", "F"]:
            recommendations.append("Improve semantic naming and summary quality for intents")
        if quality_results["mcp_tools"].get("grade", "N/A") in ["D", "F"]:
            recommendations.append("Enhance MCP tool descriptions and schema definitions")
        if quality_results["overall"].get("grade", "N/A") in ["C", "D", "F"]:
            recommendations.append("Consider improving LLM prompts for better semantic transformation")
        if len(self.generation_stats.get("quality_issues", [])) > max_issues:
            recommendations.append("High number of quality issues detected - review enrichment pipeline")
        
        return recommendations

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        grading_thresholds = self.quality_config.get("grading_thresholds", {
            "A": 90, "B": 80, "C": 70, "D": 60, "F": 0
        })
        
        for grade, threshold in grading_thresholds.items():
            if score >= threshold:
                return grade
        return "F"

    async def _save_generation_summary(self, saved_files: List[str], generation_time: float, timestamp: str) -> None:
        """Save generation summary with metrics."""
        summary = {
            "generation_timestamp": timestamp,
            "generation_time_seconds": generation_time,
            "files_generated": saved_files,
            "statistics": self.generation_stats,
            "config_profile": self.config.get("profile", "development"),
            "quality_metrics": {
                "total_intents": self.generation_stats["intents_processed"],
                "valid_intents": self.generation_stats["intents_valid"],
                "total_tools": self.generation_stats["mcp_tools_generated"],
                "valid_tools": self.generation_stats["mcp_tools_valid"],
                "quality_issues_count": len(self.generation_stats["quality_issues"]),
                "validation_errors_count": len(self.generation_stats["validation_errors"])
            }
        }
        
        summary_file = self.output_dir / f"generation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info("Generation summary saved", file=str(summary_file))

    async def save_mcp_metadata(self, mcp_tools: List[Dict]) -> None:
        """Save MCP tool metadata independently."""
        await self.save_outputs(intents=[], capabilities={}, mcp_tools=mcp_tools)

# Maintain backward compatibility
OutputGenerator = EnhancedOutputGenerator