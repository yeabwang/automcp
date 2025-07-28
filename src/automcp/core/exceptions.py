# exceptions.py - Enhanced custom exceptions for spec analyzer framework
# =============================================================================
# This update enhances the exception hierarchy to:
# - Cover parsing, enrichment, validation, LLM client, and output generation errors.
# - Include context metadata (error code, source, details) for traceability and debugging.
# - Support async operations for compatibility with parsers.py, enricher.py, and output_generator.py.
# - Integrate with config.yaml for error handling settings (e.g., retries, logging verbosity).
# - Add exceptions for MCP client-type issues to support langchain, fastmcp, openai, etc., and generic fallback.
# - Ensure production-ready robustness with structured logging and recovery suggestions.
# - Retain SpecAnalyzerError and ParserError for backward compatibility.
# =============================================================================

from typing import Dict, Optional, Any
import structlog

class SpecAnalyzerError(Exception):
    """Base class for all custom exceptions in the spec analyzer component."""
    def __init__(self, message: str, error_code: str = "UNKNOWN", source: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.source = source  # Component raising the error (e.g., 'parser', 'enricher')
        self.context = context or {}  # Additional metadata (e.g., file path, intent name)
        self.recovery_suggestion = self._get_recovery_suggestion()
        super().__init__(self.message)
        
        # Structured logging
        logger = structlog.get_logger(__name__)
        logger.error(
            "SpecAnalyzerError raised",
            error_code=self.error_code,
            source=self.source,
            context=self.context,
            recovery_suggestion=self.recovery_suggestion
        )

    def _get_recovery_suggestion(self) -> str:
        """Provide default recovery suggestion."""
        return "Check logs for details and verify input data or configuration."

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured dictionary for logging or API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "source": self.source,
            "context": self.context,
            "recovery_suggestion": self.recovery_suggestion
        }

class ParserError(SpecAnalyzerError):
    """Raised when an input spec file cannot be parsed correctly."""
    def __init__(self, message: str, spec_path: Optional[str] = None, parser_type: Optional[str] = None, error_code: str = "PARSING_FAILED"):
        context = {"spec_path": spec_path, "parser_type": parser_type}
        super().__init__(message, error_code=error_code, source="parser", context=context)

    def _get_recovery_suggestion(self) -> str:
        return f"Verify the specification file at {self.context.get('spec_path', 'unknown path')} is valid and matches the expected format (e.g., OpenAPI, Postman)."

class EnrichmentError(SpecAnalyzerError):
    """Raised when semantic enrichment fails."""
    def __init__(self, message: str, intent_name: Optional[str] = None, stage: Optional[str] = None, error_code: str = "ENRICHMENT_FAILED"):
        context = {"intent_name": intent_name, "stage": stage}
        super().__init__(message, error_code=error_code, source="enricher", context=context)

    def _get_recovery_suggestion(self) -> str:
        return "Check LLM prompts and input intent data for correctness. Consider using fallback strategies in config.yaml."

class ValidationError(SpecAnalyzerError):
    """Raised when validation of intents, capabilities, or MCP tools fails."""
    def __init__(self, message: str, object_type: Optional[str] = None, object_name: Optional[str] = None, error_code: str = "VALIDATION_FAILED"):
        context = {"object_type": object_type, "object_name": object_name}
        super().__init__(message, error_code=error_code, source="output_generator", context=context)

    def _get_recovery_suggestion(self) -> str:
        return f"Review validation errors for {self.context.get('object_type', 'object')} '{self.context.get('object_name', 'unknown')}'. Adjust input data or config.yaml thresholds."

class LLMClientError(SpecAnalyzerError):
    """Raised when LLM client operations fail."""
    def __init__(self, message: str, provider: Optional[str] = None, operation: Optional[str] = None, error_code: str = "LLM_CLIENT_FAILED"):
        context = {"provider": provider, "operation": operation}
        super().__init__(message, error_code=error_code, source="llm_client", context=context)

    def _get_recovery_suggestion(self) -> str:
        return f"Verify LLM provider '{self.context.get('provider', 'unknown')}' configuration and API key in config.yaml. Check network connectivity or circuit breaker status."

class ClientTypeError(SpecAnalyzerError):
    """Raised when an unsupported or invalid MCP client type is encountered."""
    def __init__(self, message: str, client_type: Optional[str] = None, error_code: str = "INVALID_CLIENT_TYPE"):
        context = {"client_type": client_type}
        super().__init__(message, error_code=error_code, source="enricher_or_output", context=context)

    def _get_recovery_suggestion(self) -> str:
        return f"Ensure 'client_type' in config.yaml is one of: langchain, fastmcp, openai, llamaindex, autogen, generic. Found: {self.context.get('client_type', 'unknown')}."

class OutputGenerationError(SpecAnalyzerError):
    """Raised when output generation or file saving fails."""
    def __init__(self, message: str, output_type: Optional[str] = None, file_path: Optional[str] = None, error_code: str = "OUTPUT_GENERATION_FAILED"):
        context = {"output_type": output_type, "file_path": file_path}
        super().__init__(message, error_code=error_code, source="output_generator", context=context)

    def _get_recovery_suggestion(self) -> str:
        return f"Check file path '{self.context.get('file_path', 'unknown')}' permissions and config.yaml output settings for {self.context.get('output_type', 'unknown')}."
class ClientError(Exception):
    """Exception for HTTP client-related errors (e.g., network issues, 4xx/5xx responses)."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code