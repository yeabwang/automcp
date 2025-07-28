"""
Sample MCP Tool Generation Framework Components

This package contains sample implementations of the MCP tool generation pipeline:
- parsers: API specification parsing with multiple format support
- enricher: Semantic enrichment using LLM integration
- output_generator: MCP tool specification generation
- async_llm_client: Asynchronous LLM client implementation
- llm_client_interface: Interface definition for LLM clients
- exceptions: Custom exception classes
"""

__version__ = "1.0.0"
__author__ = "MCP Framework Team"

# Make key classes available at package level
try:
    from .parsers import EnhancedOpenAPIParser
    from .enricher import SemanticEnricher
    from .output_generator import EnhancedOutputGenerator
    from .async_llm_client import EnhancedAsyncLLMClient
    from .llm_client_interface import LLMClientInterface
    from .exceptions import ParserError, ClientError, ValidationError
    
    __all__ = [
        'EnhancedOpenAPIParser',
        'SemanticEnricher', 
        'EnhancedOutputGenerator',
        'EnhancedAsyncLLMClient',
        'LLMClientInterface',
        'ParserError',
        'ClientError',
        'ValidationError'
    ]
except ImportError as e:
    # Graceful handling if some modules have issues
    print(f"Warning: Could not import some components: {e}")
    __all__ = []
