#llm_client_interface.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Any, Awaitable, TypedDict

class ResponseFormat(str, Enum):
    """Enum for valid LLM response formats."""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"  # Added for semantic transformations

class LLMProviderSettings(TypedDict, total=False):
    """Type hint for provider-specific LLM settings passed via **kwargs."""
    max_tokens: int
    temperature: float
    model: str
    anthropic_version: str
    response_format: dict
    top_p: float
    stop_sequences: List[str]

class LLMClientInterface(ABC):
    """Abstract interface for LLM clients in the MCP framework."""

    @abstractmethod
    async def query(self, prompt: str, response_format: Optional[ResponseFormat] = None, **kwargs: LLMProviderSettings) -> Any:
        """
        Query the LLM with a single prompt asynchronously.

        Args:
            prompt: The input prompt for the LLM (must be non-empty string).
            response_format: Desired response format (json, text, structured). If None, defaults to provider's default.
            **kwargs: Dynamic provider settings, e.g.:
                - max_tokens: Maximum tokens in response (e.g., 4096).
                - temperature: Sampling temperature (e.g., 0.1 for deterministic).
                - model: Specific model name (e.g., 'grok-beta').
                - anthropic_version: Version for Anthropic/Claude (e.g., '2023-06-01').
                See config.yaml's llm_client.provider_settings for valid keys per provider.

        Returns:
            Any: Response as dict (json/structured) or str (text), depending on response_format.

        Raises:
            ValueError: If prompt is empty, response_format is invalid, or API call fails.
            LLMClientError: For provider-specific errors (e.g., rate limits, network issues).
            
        Notes:
            - Implementations must scrub PII in prompts/responses per config.yaml's sensitive_data_patterns.
            - Used in parsers.py (e.g., EnhancedOpenAPIParser._infer_security_with_llm) and enricher.py.
            - Validate kwargs with validate_kwargs() to prevent invalid settings.
        """
        pass

    @abstractmethod
    async def batch_query(self, prompts: List[str], response_format: Optional[ResponseFormat] = None, **kwargs: LLMProviderSettings) -> List[Any]:
        """
        Batch query the LLM with multiple prompts asynchronously.

        Args:
            prompts: List of input prompts (non-empty strings).
            response_format: Desired response format (json, text, structured). If None, defaults to provider's default.
            **kwargs: Dynamic provider settings (see query() for details).

        Returns:
            List[Any]: List of responses (dict for json/structured, str for text).

        Raises:
            ValueError: If prompts list is empty, contains empty strings, or response_format is invalid.
            LLMClientError: For provider-specific errors.

        Notes:
            - Optimizes for batch_size per config.yaml's llm_client.provider_settings.optimal_batch_size.
            - Implementations must handle rate limits and timeouts robustly.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM API is available asynchronously.

        Returns:
            bool: True if healthy (e.g., responds with expected output like 'OK'), False otherwise.

        Raises:
            LLMClientError: If network or provider errors occur.

        Notes:
            - Uses config.yaml's llm_client.health_check.prompt (e.g., 'Respond with "OK"').
            - Called periodically per health_check.interval_seconds.
        """
        pass

    @abstractmethod
    async def query_semantic(self, prompt: str, expected_schema: Optional[Dict] = None, **kwargs: LLMProviderSettings) -> Dict:
        """
        Async semantic query with schema validation.

        Args:
            prompt: Input prompt for semantic transformation (e.g., generating a structured response).
            expected_schema: Optional JSON Schema to validate response (e.g., {"type": "object", "properties": {...}}).
            **kwargs: Dynamic provider settings (see query() for details).

        Returns:
            Dict: Validated response with optional 'validation_errors' key if schema validation fails.

        Raises:
            ValueError: If prompt is empty or schema is malformed.
            LLMClientError: For provider-specific errors.

        Notes:
            - Ideal for structured outputs (e.g., semantic naming in enricher.py).
            - Implementations must parse JSON responses and validate against expected_schema.
            - Encouraged for enricher.py methods like _generate_semantic_name to ensure type safety.
        """
        pass

    @abstractmethod
    async def batch_query_semantic(self, semantic_queries: List[Dict], **kwargs: LLMProviderSettings) -> List[Dict]:
        """
        Async batch semantic query with validation.

        Args:
            semantic_queries: List of dicts, each with 'prompt' (str) and optional 'expected_schema' (Dict).
            **kwargs: Dynamic provider settings (see query() for details).

        Returns:
            List[Dict]: List of validated responses, each with optional 'validation_errors'.

        Raises:
            ValueError: If semantic_queries is empty or contains invalid entries.
            LLMClientError: For provider-specific errors.

        Notes:
            - Useful for batch processing in enricher.py (e.g., multiple intent transformations).
            - Implementations must handle batch_size limits per provider settings.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict:
        """
        Get performance metrics for production monitoring.

        Returns:
            Dict: Metrics including:
                - total_requests: Total LLM API calls.
                - successful_requests: Successful calls.
                - failed_requests: Failed calls.
                - success_rate: Percentage of successful calls.
                - total_tokens: Estimated tokens used.
                - avg_latency: Average request latency (seconds).
                - last_request_time: Timestamp of last request.
                - circuit_breaker_state: Current state (closed/open/half-open).

        Notes:
            - Used for observability (e.g., Prometheus integration per config.yaml's monitoring).
        """
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        """
        Reset performance metrics to initial state.

        Notes:
            - Useful for testing or clearing stale metrics in long-running processes.
            - Implementations must log reset events for traceability.
        """
        pass

    @abstractmethod
    def validate_kwargs(self, provider: str, **kwargs: LLMProviderSettings) -> None:
        """
        Validate provider-specific settings passed via **kwargs.

        Args:
            provider: LLM provider name (e.g., 'groq', 'openai') from config.yaml's llm_client.provider.
            **kwargs: Provider settings to validate (e.g., max_tokens, temperature).

        Raises:
            ValueError: If any kwargs are invalid for the specified provider (e.g., out-of-range values).
            
        Notes:
            - Checks against config.yaml's llm_client.provider_settings[provider] constraints.
            - Example: Ensures max_tokens <= provider_settings[provider].max_tokens.
            - Call before query() or batch_query() to prevent runtime errors.
        """
        pass