# async_llm_client.py - Optimized LLM Client for MCP Tool Generation Framework
# =============================================================================
# This module provides an async LLM client for semantic enrichment and transformations.
# Key Features:
# - Fully config-driven: Providers, endpoints, headers, retries, PII patterns, circuit breakers.
# - Supports **kwargs in query/batch_query for dynamic settings (fixes parsers.py errors).
# - Supports PII scrubbing in prompts/responses with config patterns and regex.
# - Async query, batch, health check for scalability.
# - Robust error handling: Config retries, circuit breakers with half-open state.
# - Metrics tracking, semantic validation, batch optimization.
# - Supports MCP clients: Flexible providers/settings for langchain, openai, generic, etc.
# Critical Requirements:
# - Industry-standard: aiohttp, tenacity, structlog for async/retries/logging.
# - Production-ready: Config retries/circuit breakers, metrics, env API keys.
# - Secure: PII scrubbing with config patterns.
# - Rich: JSON/text/structured responses; semantic methods; batch processing.
# - Robust: Handles timeouts, rate limits, failures; no hardcoding.
# - No hardcoded logic: All from config; dynamic **kwargs for flexibility.
# Assumptions and Dependencies
# llm_client_interface.py: Assumes to define LLMClientInterface and ResponseFormat (enum with json, text, structured). If not, ResponseFormat can be defined here or adjusted.
# config.yaml Structure: Assumes llm_client section with provider, model, endpoints, provider_settings, health_check.prompt; error_handling with max_retries, retry_on_status_codes, circuit_breaker; semantic_transformation with sensitive_data_patterns; timeouts with llm_query, session_max_age.
# Pydantic Version: Assumes v2.x for validator syntax. For v1.x, adjust to @validator.
# AsyncLLMClient Usage: Assumed used in parsers.py and enricher.py for LLM calls; supports **provider_settings via **kwargs.
# =============================================================================

# Standard library
import asyncio
import uuid
import os
import json
import time
import re
from enum import Enum
from typing import List, Dict, Optional, Any, Awaitable
from dataclasses import dataclass

# Third-party
import aiohttp
import scrubadub
import logging
import structlog
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log
from requests.exceptions import RequestException
from aiohttp import ClientError  # FIX: Imported ClientError for retry logic

# Local
from .llm_client_interface import LLMClientInterface, ResponseFormat, LLMProviderSettings

@dataclass
class LLMMetrics:
    """LLM performance metrics for monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    last_request_time: float = 0.0

def create_logger(config: Dict[str, Any]) -> structlog.BoundLogger:
    """Create and configure structlog logger from config."""
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

    # FIX: Used structlog.stdlib.get_logger for stdlib Logger compatibility in before_sleep_log
    std_logger = logging.getLogger(__name__)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_sensitive,
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    return structlog.wrap_logger(std_logger)

class EnhancedAsyncLLMClient(LLMClientInterface):
    """
    Optimized asynchronous LLM client for semantic transformation workloads.
    """

    def __init__(self, config: Dict[str, Any], endpoint_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        # FIX: Restored endpoint_url and headers parameters (fixes missing parameters in __init__)
        self.config = config
        llm_cfg = config.get("llm_client", {})
        if not llm_cfg:
            raise ValueError("Missing 'llm_client' section in config")

        self.logger = create_logger(config)

        # Config-driven provider configuration
        self.provider = llm_cfg.get("provider", "groq")
        self.model = llm_cfg.get("model", "grok-beta")
        self.endpoint_url = endpoint_url or llm_cfg.get("endpoints", {}).get(self.provider, "")
        if not self.endpoint_url:
            raise ValueError(f"No endpoint for provider: {self.provider}")

        # Config-driven provider settings
        self.provider_settings = llm_cfg.get("provider_settings", {}).get(self.provider, {})
        self.max_tokens_default = self.provider_settings.get("max_tokens", 4096)
        self.temperature_default = self.provider_settings.get("temperature", 0.1)
        self.batch_size_limit = self.provider_settings.get("batch_size_limit", 20)
        self.optimal_batch_size = self.provider_settings.get("optimal_batch_size", 8)
        self.max_prompt_length = self.provider_settings.get("max_prompt_length", 8000)
        self.supports_json_mode = self.provider_settings.get("supports_json_mode", True)
        self.anthropic_version = self.provider_settings.get("anthropic_version", "2023-06-01")

        # Config-driven PII scrubbing
        self.sensitive_data_patterns = config.get("semantic_transformation", {}).get("sensitive_data_patterns", {})
        self.all_sensitive_keys = sum(self.sensitive_data_patterns.values(), [])

        # Config-driven error handling
        error_handling = config.get("error_handling", {})
        self.retry_attempts = error_handling.get("max_retries", 3)
        self.retry_wait_min = self.provider_settings.get("retry_wait_min", 2)
        self.retry_wait_max = self.provider_settings.get("retry_wait_max", 8)
        self.retry_multiplier = self.provider_settings.get("retry_multiplier", 1)
        self.retry_on_status_codes = error_handling.get("retry_on_status_codes", [429, 502, 503, 504])
        self.retry_on_exceptions = [ClientError, RequestException]  # FIX: Explicitly list exceptions (uses imported ClientError)
        self.failure_threshold = error_handling.get("circuit_breaker", {}).get("failure_threshold", 5)
        self.recovery_timeout = error_handling.get("circuit_breaker", {}).get("recovery_timeout", 60)

        # Config-driven session management
        timeouts = config.get("timeouts", {})
        self.session_max_age = timeouts.get("session_max_age", 3600)
        self.timeout = aiohttp.ClientTimeout(
            total=timeouts.get("llm_query", 60),
            connect=30,
            sock_read=30
        )
        self.connector_limit = self.provider_settings.get("connector_limit", 100)
        self.connector_limit_per_host = self.provider_settings.get("connector_limit_per_host", 20)

        # Setup authentication
        self.headers = headers or self._setup_authentication(llm_cfg)

        # Performance tracking
        self.metrics = LLMMetrics()

        # Session management
        self.session = None
        self._session_created_at = 0
        self._session_lock = asyncio.Lock()  # FIX: Added Lock to prevent concurrent session creation (fixes race condition)

        # Circuit breaker state
        self.circuit_breaker = {
            "state": "closed",
            "failure_count": 0,
            "last_failure_time": 0,
            "next_attempt_time": 0
        }

        # Health check prompt
        self.health_prompt = llm_cfg.get("health_check", {}).get("prompt", "Respond with 'OK'")

        self.logger.info("Enhanced LLM client initialized",
                        provider=self.provider,
                        model=self.model,
                        endpoint=self.endpoint_url)

    def _setup_authentication(self, llm_cfg: Dict[str, Any]) -> Dict[str, str]:
        """Setup authentication from env/config."""
        env_key_names = [
            "LLM_API_KEY",
            f"{self.provider.upper()}_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY"
        ]
        
        api_key = None
        for env_key in env_key_names:
            api_key = os.environ.get(env_key)
            if api_key:
                self.logger.debug("Using API key from environment", env_var=env_key)
                break
        
        if not api_key:
            api_key = llm_cfg.get("api_key", "")
            if api_key:
                self.logger.warning("Using API key from config; recommend environment variable")

        if not api_key:
            self.logger.warning("No API key found; LLM features will not work")
            return {}

        headers = {}
        if self.provider in ["groq", "openai"]:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.provider in ["anthropic", "claude"]:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = self.anthropic_version
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        
        headers["User-Agent"] = llm_cfg.get("user_agent", "MCP-SpecAnalyzer/1.0")
        return headers

    async def _ensure_session(self):
        """Ensure session exists and is not expired."""
        async with self._session_lock:
            now = time.time()
            if (self.session is None or 
                self.session.closed or 
                (now - self._session_created_at) > self.session_max_age):
                
                if self.session and not self.session.closed:
                    await self.session.close()
                
                connector = aiohttp.TCPConnector(
                    limit=self.connector_limit,
                    limit_per_host=self.connector_limit_per_host,
                    keepalive_timeout=60,
                    enable_cleanup_closed=True
                )
                
                self.session = aiohttp.ClientSession(
                    timeout=self.timeout,
                    connector=connector,
                    headers=self.headers
                )
                
                self._session_created_at = now
                self.logger.debug("Session created/renewed")

    async def __aenter__(self):
        """Initialize aiohttp session."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Clean up aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def _scrub_pii(self, text: str) -> str:
        """Scrub PII using config patterns and scrubadub."""
        cleaned_text = scrubadub.clean(text)
        for pattern in self.all_sensitive_keys:
            cleaned_text = re.sub(rf'\b{re.escape(pattern)}\b', "[REDACTED]", cleaned_text, flags=re.IGNORECASE)
        return cleaned_text

    def _check_circuit_breaker(self):
        """Check circuit breaker state."""
        now = time.time()
        if self.circuit_breaker["state"] == "open":
            if now >= self.circuit_breaker["next_attempt_time"]:
                self.circuit_breaker["state"] = "half_open"
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise Exception(f"Circuit breaker open. Next attempt in {self.circuit_breaker['next_attempt_time'] - now:.1f}s")

    def _update_circuit_breaker(self, success: bool):
        """Update circuit breaker state."""
        if success:
            if self.circuit_breaker["state"] == "half_open":
                self.circuit_breaker["state"] = "closed"
                self.circuit_breaker["failure_count"] = 0
                self.logger.info("Circuit breaker closed")
        else:
            self.circuit_breaker["failure_count"] += 1
            self.circuit_breaker["last_failure_time"] = time.time()
            if self.circuit_breaker["failure_count"] >= self.failure_threshold:
                self.circuit_breaker["state"] = "open"
                self.circuit_breaker["next_attempt_time"] = time.time() + self.recovery_timeout
                self.logger.warning("Circuit breaker opened")

    def _validate_response_format(self, response_format: Optional[ResponseFormat]):
        """Validate response format."""
        if response_format and response_format not in ResponseFormat:
            raise ValueError(f"Invalid response_format: {response_format}. Must be one of ResponseFormat.")

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt for provider-specific optimization."""
        # FIX: Restored prompt validation for robustness (fixes loss of validation)
        if not prompt or not isinstance(prompt, str):
            self.logger.error("Invalid prompt", prompt_type=type(prompt))
            raise ValueError("Prompt must be a non-empty string")
        cleaned_prompt = self._scrub_pii(prompt)
        if self.provider == "anthropic" and not cleaned_prompt.startswith("You are"):
            cleaned_prompt = f"You are an expert API analyzer. {cleaned_prompt}"
        if len(cleaned_prompt) > self.max_prompt_length:
            cleaned_prompt = cleaned_prompt[:self.max_prompt_length] + "... [truncated]"
            self.logger.warning("Prompt truncated", original_length=len(prompt))
        return cleaned_prompt

    def _build_payload(self, prompt: str, response_format: Optional[ResponseFormat], **kwargs) -> Dict:
        """Build provider-specific payload with **kwargs."""
        # FIX: Explicitly typed base_payload as Dict[str, Any] to allow list[dict] for "messages" (fixes type mismatch at Lines 322, 324, 326, 329)
        base_payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature_default),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens_default)
        }

        if self.provider in ["groq", "openai"]:
            base_payload["messages"] = [{"role": "user", "content": prompt}]
            if response_format == ResponseFormat.JSON and self.supports_json_mode:
                base_payload["response_format"] = {"type": "json_object"}
        elif self.provider in ["anthropic", "claude"]:
            base_payload["messages"] = [{"role": "user", "content": prompt}]
            base_payload["anthropic_version"] = kwargs.get("anthropic_version", self.anthropic_version)
        else:
            base_payload["messages"] = [{"role": "user", "content": prompt}]

        return base_payload

    def _process_response(self, json_resp: Dict, response_format: Optional[ResponseFormat]) -> Any:
        """Process provider-specific response."""
        if self.provider in ["groq", "openai"]:
            choices = json_resp.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            content = choices[0].get("message", {}).get("content", "")
        elif self.provider in ["anthropic", "claude"]:
            content_blocks = json_resp.get("content", [])
            if not content_blocks:
                raise ValueError("No content in response")
            content = content_blocks[0].get("text", "")
        else:
            content = str(json_resp)

        cleaned_content = self._scrub_pii(content)
        if response_format == ResponseFormat.JSON:
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                self.logger.warning("Failed to parse JSON response", error=str(e))
                return {"content": cleaned_content, "parse_error": True}
        elif response_format == ResponseFormat.STRUCTURED:
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                return {"content": cleaned_content.strip()}
        return cleaned_content.strip()

    def _should_retry(self, exception: BaseException | None) -> bool:
        """Determine if exception should trigger retry."""
        # FIX: Updated param to Optional[BaseException], handled None, and narrowed to Exception (fixes type for BaseException | None to Exception at Line 391)
        if exception is None:
            return False
        if isinstance(exception, Exception):  # Narrow to Exception
            if isinstance(exception, tuple(self.retry_on_exceptions)):
                return True
            if isinstance(exception, ValueError) and any(str(code) in str(exception).lower() for code in self.retry_on_status_codes):
                return True
        return False

    def _update_metrics(self, success: bool, latency: float, response_size: int):
        """Update performance metrics."""
        self.metrics.total_requests += 1
        self.metrics.last_request_time = time.time()
        self.metrics.total_latency += latency
        self.metrics.avg_latency = self.metrics.total_latency / self.metrics.total_requests if self.metrics.total_requests > 0 else 0

        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

        estimated_tokens = response_size // 4
        self.metrics.total_tokens += estimated_tokens

    def _make_retryable_async_query(self):
        """Create retryable async query with config-driven retries."""
        @retry(
            wait=wait_exponential(multiplier=self.retry_multiplier, min=self.retry_wait_min, max=self.retry_wait_max),
            stop=stop_after_attempt(self.retry_attempts),
            before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
            retry=lambda retry_state: self._should_retry(retry_state.outcome.exception() if retry_state.outcome else None)
        )
        async def _query(prompt: str, response_format: Optional[ResponseFormat] = None, **kwargs):
            self._check_circuit_breaker()
            await self._ensure_session()
            self._validate_response_format(response_format)
            cleaned_prompt = self._preprocess_prompt(prompt)

            trace_id = str(uuid.uuid4())
            start_time = time.time()
            try:
                self.logger.info("Starting LLM query", trace_id=trace_id, provider=self.provider)

                payload = self._build_payload(cleaned_prompt, response_format, **kwargs)
                # FIX: Added assertion for self.session to prevent "post not a known attribute of None" (Line 405)
                assert self.session is not None, "Session must be initialized before query"
                async with self.session.post(self.endpoint_url, json=payload) as resp:
                    latency = time.time() - start_time
                    if resp.status != 200:
                        error_text = await resp.text()
                        self._update_circuit_breaker(False)
                        raise ValueError(f"LLM API error: {resp.status} - {error_text}")

                    if "application/json" not in resp.content_type:
                        self._update_circuit_breaker(False)
                        raise ValueError(f"Non-JSON response: {resp.content_type}")

                    json_resp = await resp.json()
                    processed_response = self._process_response(json_resp, response_format)
                    self._update_circuit_breaker(True)
                    self._update_metrics(True, latency, len(str(json_resp)))

                    return processed_response

            except Exception as e:
                latency = time.time() - start_time
                self._update_circuit_breaker(False)
                self._update_metrics(False, latency, 0)
                self.logger.error("LLM query failed", trace_id=trace_id, error=str(e))
                raise

        return _query

    async def query(self, prompt: str, response_format: Optional[ResponseFormat] = None, **kwargs):
        """Async query with **kwargs for dynamic settings (fixes parsers.py errors)."""
        query_func = self._make_retryable_async_query()
        return await query_func(prompt, response_format, **kwargs)

    async def batch_query(self, prompts: List[str], response_format: Optional[ResponseFormat] = None, **kwargs) -> List[Any]:
        """Async batch query with optimal batch size."""
        self._check_circuit_breaker()
        await self._ensure_session()

        results = []
        for i in range(0, len(prompts), self.optimal_batch_size):
            batch = prompts[i:i + self.optimal_batch_size]
            tasks = [self.query(p, response_format, **kwargs) for p in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({"error": str(result), "failed": True})
                else:
                    results.append(result)

        return results

    async def health_check(self) -> bool:
        """Async health check with config prompt."""
        try:
            result = await self.query(self.health_prompt, ResponseFormat.TEXT)
            return "OK" in str(result)
        except Exception as e:
            self.logger.error("LLM health check failed", error=str(e))
            return False

    async def query_semantic(self, prompt: str, expected_schema: Optional[Dict] = None, **kwargs):
        """Async semantic query with validation."""
        result = await self.query(prompt, ResponseFormat.STRUCTURED, **kwargs)
        if expected_schema is not None and isinstance(result, dict):
            validation = self._validate_semantic_response(result, expected_schema)
            if not validation["valid"]:
                self.logger.warning("Semantic response validation failed", errors=validation["errors"])
                result["validation_errors"] = validation["errors"]
        return result

    async def batch_query_semantic(self, semantic_queries: List[Dict], **kwargs) -> List[Dict]:
        """Async batch semantic query with validation."""
        prompts = [q["prompt"] for q in semantic_queries]
        schemas = [q.get("expected_schema") for q in semantic_queries]
        results = await self.batch_query(prompts, ResponseFormat.STRUCTURED, **kwargs)
        validated_results = []
        for i, result in enumerate(results):
            schema = schemas[i]
            if schema is not None and isinstance(result, dict) and not result.get("failed"):
                validation = self._validate_semantic_response(result, schema)
                if not validation["valid"]:
                    result["validation_errors"] = validation["errors"]
            validated_results.append(result)
        return validated_results

    def _validate_semantic_response(self, response: Dict, expected_schema: Dict) -> Dict:
        """Validate semantic response against schema."""
        errors = []
        required_fields = expected_schema.get("required", [])
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")

        properties = expected_schema.get("properties", {})
        for field, schema in properties.items():
            if field in response:
                expected_type = schema.get("type")
                actual_value = response[field]
                if expected_type == "string" and not isinstance(actual_value, str):
                    errors.append(f"Field {field} should be string, got {type(actual_value).__name__}")
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    errors.append(f"Field {field} should be number, got {type(actual_value).__name__}")
                elif expected_type == "boolean" and not isinstance(actual_value, bool):
                    errors.append(f"Field {field} should be boolean, got {type(actual_value).__name__}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (self.metrics.successful_requests / max(self.metrics.total_requests, 1)) * 100,
            "total_tokens": self.metrics.total_tokens,
            "avg_latency": self.metrics.avg_latency,
            "last_request_time": self.metrics.last_request_time,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "circuit_breaker_failures": self.circuit_breaker["failure_count"]
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = LLMMetrics()
        self.logger.info("LLM metrics reset")

    # FIX: Added validate_kwargs implementation to check provider-specific settings
    def validate_kwargs(self, provider: str, **kwargs: LLMProviderSettings) -> None:
        """Validate provider-specific settings passed via **kwargs."""
        if provider != self.provider:
            raise ValueError(f"Provider mismatch: expected {self.provider}, got {provider}")

        # Retrieve provider-specific constraints from config.yaml
        provider_settings = self.config.get("llm_client", {}).get("provider_settings", {}).get(provider, {})
        if not provider_settings:
            self.logger.warning("No provider settings found for validation", provider=provider)
            return  # Skip validation if no settings defined

        # Define common validation rules
        if "max_tokens" in kwargs:
            max_tokens_limit = provider_settings.get("max_tokens", 4096)
            if not isinstance(kwargs["max_tokens"], int) or kwargs["max_tokens"] > max_tokens_limit:
                raise ValueError(f"max_tokens must be an integer <= {max_tokens_limit}, got {kwargs['max_tokens']}")
        
        if "temperature" in kwargs:
            if not isinstance(kwargs["temperature"], (int, float)) or not (0 <= kwargs["temperature"] <= 1.0):
                raise ValueError(f"temperature must be between 0 and 1.0, got {kwargs['temperature']}")
        
        if "model" in kwargs and not isinstance(kwargs["model"], str):
            raise ValueError(f"model must be a string, got {type(kwargs['model']).__name__}")
        
        if "anthropic_version" in kwargs and provider in ["anthropic", "claude"] and not isinstance(kwargs["anthropic_version"], str):
            raise ValueError(f"anthropic_version must be a string, got {type(kwargs['anthropic_version']).__name__}")
        
        # Additional provider-specific validations can be added here

# Backward compatibility
AsyncLLMClient = EnhancedAsyncLLMClient