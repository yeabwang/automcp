"""Pydantic models for AutoMCP data structures."""

# Import all model classes
try:
    from .intent import IntentParameter, Intent, EnrichedIntent
    from .capability import CapabilityType, Capability, CapabilityGroup
    from .mcp_tool import MCPToolType, MCPParameter, MCPTool, MCPToolSet
except ImportError:
    # Fallback for import issues
    pass

__all__ = [
    # Intent models
    "IntentParameter",
    "Intent", 
    "EnrichedIntent",
    # Capability models
    "CapabilityType",
    "Capability",
    "CapabilityGroup", 
    # MCP Tool models
    "MCPToolType",
    "MCPParameter",
    "MCPTool",
    "MCPToolSet"
]
