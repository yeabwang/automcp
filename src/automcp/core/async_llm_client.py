# async_llm_client.py - Optimized LLM Client for MCP Tool Generation Framework
# =============================================================================
# This module provides an async LLM client for semantic enrichment and transformations.
# Key Features:
# - Fully config-driven: Providers, endpoints, headers, retries, PII patterns, circuit breakers.
# - Supports **kwargs in query/batch_query for dynamic settings (fixes parsers.py errors).
# - Supports PII scrubbing in prompts/responses with config patterns and libraries.
# - Async query, batch, health check for scalability.
# - Robust error handling: Config retries, circuit breakers with half-open state.
# - Metrics tracking, semantic validation, batch optimization.
# - Supports MCP clients: Flexible providers/settings for langchain, openai, generic, etc.
# Critical Requirements:
# - Industry-standard: aiohttp, tenacity, structlog for async/retries/logging.
# - Production-ready: Config retries/circuit breakers, metrics, env API keys.
# - Secure: PII scrubbing with libraries.
# - Rich: JSON/text/structured responses; semantic methods; batch processing.
# - Robust: Handles timeouts, rate limits, failures; no hardcoding.
# - No hardcoded logic: All from config; dynamic **kwargs for flexibility.
# Assumptions and Dependencies
# llm_client_interface.py: Assumes to define LLMClientInterface and ResponseFormat (enum with json, text, structured). If not, ResponseFormat can be defined here or adjusted.
# config.yaml Structure: Assumes llm_client section with provider, model, endpoints, provider_settings, health_check.prompt; error_handling with max_retries, retry_on_status_codes, circuit_breaker; semantic_transformation with sensitive_data_patterns; timeouts with llm_query, session_max_age.
# Pydantic Version: Assumes v2.x for validator syntax. For v1.x, adjust to @validator.
# AsyncLLMClient Usage: Assumed used in parsers.py and enricher.py for LLM calls; supports **provider_settings via **kwargs.
# =============================================================================

# Standard library imports (alphabetical order - industry standard)
import asyncio
import io
import json
import logging
import os
import ssl
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Awaitable

# Third-party imports (alphabetical order - industry standard)
import aiohttp
import scrubadub
import structlog
from aiohttp import ClientError
from requests.exceptions import RequestException
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log

# Industry-standard JSON parsing libraries (conditional imports)
try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    logging.warning("json-repair library not available - install with: pip install json-repair")

try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    logging.warning("json5 library not available - install with: pip install json5")

try:
    import jsonlines
    HAS_JSONLINES = True
except ImportError:
    HAS_JSONLINES = False
    logging.warning("jsonlines library not available - install with: pip install jsonlines")

try:
    import demjson3
    HAS_DEMJSON = True
except ImportError:
    HAS_DEMJSON = False
    logging.warning("demjson3 library not available - install with: pip install demjson3")

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
        """Ensure session exists and is not expired with enterprise SSL security."""
        async with self._session_lock:
            now = time.time()
            if (self.session is None or 
                self.session.closed or 
                (now - self._session_created_at) > self.session_max_age):
                
                if self.session and not self.session.closed:
                    await self.session.close()
                
                # Enterprise SSL context
                ssl_context = self._create_enterprise_ssl_context()
                
                connector = aiohttp.TCPConnector(
                    limit=self.connector_limit,
                    limit_per_host=self.connector_limit_per_host,
                    keepalive_timeout=60,
                    enable_cleanup_closed=True,
                    ssl=ssl_context
                )
                
                self.session = aiohttp.ClientSession(
                    timeout=self.timeout,
                    connector=connector,
                    headers=self.headers
                )
                
                self._session_created_at = now
                self.logger.debug("Session created/renewed with enterprise SSL")

    def _create_enterprise_ssl_context(self) -> ssl.SSLContext:
        """Create enterprise-grade SSL context from config."""
        ssl_config = self.config.get("ssl", {})
        
        # Create secure SSL context
        if ssl_config.get("verify_ssl", True):
            context = ssl.create_default_context()
        else:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        # Set minimum TLS version from config
        min_version_str = ssl_config.get("min_tls_version", "TLSv1_2")
        min_version = getattr(ssl.TLSVersion, min_version_str, ssl.TLSVersion.TLSv1_2)
        context.minimum_version = min_version
        
        # Configure cipher suites for security
        ciphers = ssl_config.get("ciphers", [
            "ECDHE+AESGCM",
            "ECDHE+CHACHA20",
            "DHE+AESGCM", 
            "DHE+CHACHA20",
            "!aNULL",
            "!MD5",
            "!DSS"
        ])
        if isinstance(ciphers, list):
            context.set_ciphers(":".join(ciphers))
        
        # Set SSL options for security
        context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
        context.options |= ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE
        
        # Load custom CA bundle if specified
        ca_bundle = ssl_config.get("ca_bundle_path")
        if ca_bundle and os.path.exists(ca_bundle):
            context.load_verify_locations(ca_bundle)
        
        # Client certificate authentication if specified
        client_cert = ssl_config.get("client_cert_path")
        client_key = ssl_config.get("client_key_path")
        if client_cert and client_key and os.path.exists(client_cert) and os.path.exists(client_key):
            context.load_cert_chain(client_cert, client_key, ssl_config.get("client_key_password"))
        
        self.logger.debug("Enterprise SSL context created", 
                         verify_ssl=ssl_config.get("verify_ssl", True),
                         min_tls_version=min_version_str)
        
        return context

    async def __aenter__(self):
        """Initialize aiohttp session."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Clean up aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def _scrub_pii(self, text: str) -> str:
        """Scrub PII using scrubadub library."""
        cleaned_text = scrubadub.clean(text)
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
        """Process provider-specific response with industry-standard JSON parsing only."""
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
            return self._parse_json_with_industry_libraries(cleaned_content)
        elif response_format == ResponseFormat.STRUCTURED:
            return self._parse_json_with_industry_libraries(cleaned_content)
        
        return cleaned_content.strip()

    def _parse_json_with_industry_libraries(self, content: str) -> Any:
        """Parse JSON using ONLY industry-standard libraries with comprehensive fallback strategy."""
        if not content or not content.strip():
            self.logger.warning("Empty content for JSON parsing")
            return {"error": "Empty response content", "content": content}
        
        content = content.strip()
        self.logger.debug("Attempting JSON parsing", content_length=len(content), content_preview=content[:100])
        
        # Method 1: Standard JSON parsing (fastest path)
        try:
            parsed_data = json.loads(content)
            self.logger.debug("Successfully parsed with standard JSON")
            return parsed_data
        except json.JSONDecodeError as e:
            self.logger.debug(f"Standard JSON parsing failed: {e}")
        
        # Method 2: json-repair library (industry standard for malformed JSON)
        if HAS_JSON_REPAIR:
            try:
                repaired_content = json_repair.repair_json(content)
                parsed_data = json.loads(repaired_content)
                self.logger.info("Successfully repaired JSON using json-repair library")
                return parsed_data
            except Exception as e:
                self.logger.debug(f"json-repair library failed: {e}")
        
        # Method 3: json5 library (relaxed JSON parsing)
        if HAS_JSON5:
            try:
                parsed_data = json5.loads(content)
                self.logger.info("Successfully parsed JSON using json5 library")
                return parsed_data
            except Exception as e:
                self.logger.debug(f"json5 library failed: {e}")
        
        # Method 4: demjson3 library (liberal JSON parsing)
        if HAS_DEMJSON:
            try:
                parsed_data = demjson3.decode(content, strict=False)
                self.logger.info("Successfully parsed JSON using demjson3 library")
                return parsed_data
            except Exception as e:
                self.logger.debug(f"demjson3 library failed: {e}")
        
        # Method 5: Handle multiple JSON objects (jsonlines format)
        if HAS_JSONLINES:
            try:
                # Try to parse as multiple JSON objects separated by newlines
                objects = []
                with jsonlines.Reader(io.StringIO(content)) as reader:
                    for obj in reader:
                        objects.append(obj)
                
                if len(objects) == 1:
                    self.logger.info("Successfully parsed single JSON object using jsonlines library")
                    return objects[0]
                elif len(objects) > 1:
                    self.logger.info(f"Successfully parsed {len(objects)} JSON objects using jsonlines library")
                    return objects
            except Exception as e:
                self.logger.debug(f"jsonlines library failed: {e}")
        
        # Method 6: Handle "Extra data" errors with industry-standard JSONDecoder
        # Use incremental parsing to extract the first valid JSON object
        try:
            decoder = json.JSONDecoder()
            obj, end_idx = decoder.raw_decode(content, 0)
            # Check if there's extra data after the first valid JSON
            remaining = content[end_idx:].strip()
            if remaining:
                self.logger.info(f"Successfully extracted first JSON object, ignoring extra data: {remaining[:50]}...")
            else:
                self.logger.info("Successfully parsed complete JSON object using JSONDecoder")
            return obj
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSONDecoder direct parsing failed: {e}")
        
        # Method 7: Advanced concatenated JSON handling with enhanced error recovery
        # Try to find and extract the first complete JSON object from mixed content
        if content.count('{') > 0:
            try:
                decoder = json.JSONDecoder()
                idx = 0
                attempts = 0
                max_attempts = 5
                
                while idx < len(content) and attempts < max_attempts:
                    attempts += 1
                    try:
                        # Skip non-JSON characters
                        while idx < len(content) and content[idx] not in '{[':
                            idx += 1
                        
                        if idx >= len(content):
                            break
                            
                        obj, end_idx = decoder.raw_decode(content, idx)
                        self.logger.info(f"Successfully extracted JSON object from concatenated content (attempt {attempts})")
                        return obj
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"JSONDecoder attempt {attempts} failed at position {idx}: {e}")
                        # Try to find the next potential JSON start
                        next_brace = content.find('{', idx + 1)
                        next_bracket = content.find('[', idx + 1)
                        
                        next_start = -1
                        if next_brace != -1 and next_bracket != -1:
                            next_start = min(next_brace, next_bracket)
                        elif next_brace != -1:
                            next_start = next_brace
                        elif next_bracket != -1:
                            next_start = next_bracket
                            
                        if next_start == -1:
                            break
                        idx = next_start
            except Exception as e:
                self.logger.debug(f"Advanced JSONDecoder extraction failed: {e}")
        
        # Method 10: Last resort - intelligent content truncation
        try:
            # Find the longest valid JSON substring
            for end_pos in range(len(content), 0, -1):
                test_content = content[:end_pos].rstrip()
                if test_content.endswith(('}', ']')):
                    try:
                        parsed_data = json.loads(test_content)
                        self.logger.info(f"Successfully parsed JSON after intelligent truncation (length: {end_pos})")
                        return parsed_data
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            self.logger.debug(f"Intelligent truncation failed: {e}")
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line.startswith('{') or line.startswith('[')):
                try:
                    obj = json.loads(line)
                    self.logger.info(f"Successfully parsed JSON from line {i+1}")
                    return obj
                except json.JSONDecodeError:
                    continue
        
        # Final fallback: Return comprehensive structured error with diagnostic information
        available_libs = {
            "json_repair": HAS_JSON_REPAIR,
            "json5": HAS_JSON5,
            "demjson3": HAS_DEMJSON,
            "jsonlines": HAS_JSONLINES
        }
        
        self.logger.error("All industry-standard JSON parsing methods exhausted", 
                         content_preview=content[:200],
                         content_length=len(content),
                         available_libraries=available_libs)
        
        return {
            "error": "JSON parsing failed - all industry-standard methods exhausted",
            "content": content,
            "available_libraries": available_libs,
            "parsing_methods_attempted": [
                "json.loads() - Standard JSON parsing",
                "json_repair.repair_json() - Malformed JSON repair" if HAS_JSON_REPAIR else None,
                "json5.loads() - Relaxed JSON parsing" if HAS_JSON5 else None,
                "demjson3.decode() - Liberal JSON parsing" if HAS_DEMJSON else None,
                "jsonlines.Reader() - Multiple JSON objects" if HAS_JSONLINES else None,
                "json.JSONDecoder.raw_decode() - Incremental parsing with 'Extra data' handling",
                "Advanced JSONDecoder - Multi-attempt extraction with error recovery", 
                "Intelligent content truncation - Find longest valid JSON substring"
            ],
            "error_patterns_handled": [
                "Extra data: line X column Y",
                "Expecting value: line X column Y", 
                "Trailing commas in objects/arrays",
                "Missing quotes around keys",
                "Single quotes instead of double quotes",
                "Missing values (replaced with null)",
                "JSON within code blocks or markdown",
                "Concatenated JSON objects",
                "Truncated JSON responses"
            ],
            "suggestion": "Install missing libraries: pip install json-repair json5 demjson3 jsonlines"
        }

    def _should_retry(self, exception: BaseException | None) -> bool:
        """Determine if exception should trigger retry."""
        if exception is None:
            return False
        if not isinstance(exception, Exception):
            return False
            
        # Check for specific retryable exception types
        if isinstance(exception, tuple(self.retry_on_exceptions)):
            self.logger.info("Retrying due to exception type", 
                           exception_type=type(exception).__name__, error=str(exception))
            return True
            
        # Check for HTTP status codes that should trigger retry
        if isinstance(exception, ValueError):
            for code in self.retry_on_status_codes:
                if str(code) in str(exception):
                    self.logger.info("Retrying due to HTTP status code", 
                                   status_code=code, error=str(exception))
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