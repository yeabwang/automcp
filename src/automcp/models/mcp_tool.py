"""
Pydantic models for MCP tool structures in AutoMCP.

These models define the structure and validation for MCP (Model Context Protocol)
tools generated from API specifications.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class MCPToolType(str, Enum):
    """Types of MCP tools."""
    
    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT = "prompt"


class MCPParameter(BaseModel):
    """Model for MCP tool parameters."""
    
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[str]] = Field(None, description="Allowed values")
    pattern: Optional[str] = Field(None, description="Validation pattern")
    
    @validator('type')
    def validate_type(cls, v):
        """Validate parameter type."""
        allowed_types = ['string', 'number', 'integer', 'boolean', 'array', 'object']
        if v not in allowed_types:
            raise ValueError(f'Type must be one of {allowed_types}')
        return v


class MCPTool(BaseModel):
    """Model for MCP tool representation."""
    
    name: str = Field(..., description="Tool name")
    type: MCPToolType = Field(..., description="Tool type")
    description: str = Field(..., description="Tool description")
    
    # Function-specific fields
    function: Optional[Dict[str, Any]] = Field(None, description="Function definition")
    parameters: List[MCPParameter] = Field(default_factory=list, description="Tool parameters")
    
    # Metadata
    version: str = Field(default="1.0.0", description="Tool version")
    author: Optional[str] = Field(None, description="Tool author")
    license: Optional[str] = Field(None, description="Tool license")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    
    # Integration metadata
    endpoint: Optional[str] = Field(None, description="Associated API endpoint")
    method: Optional[str] = Field(None, description="HTTP method")
    auth_required: bool = Field(default=False, description="Whether authentication is required")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    
    # Usage information
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    error_codes: Dict[str, str] = Field(default_factory=dict, description="Error code descriptions")


class MCPToolSet(BaseModel):
    """Model for a collection of MCP tools."""
    
    name: str = Field(..., description="Tool set name")
    version: str = Field(..., description="Tool set version")
    description: str = Field(..., description="Tool set description")
    tools: List[MCPTool] = Field(..., description="Tools in this set")
    
    # Metadata
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    source_api: Optional[str] = Field(None, description="Source API specification")
    generator_version: Optional[str] = Field(None, description="AutoMCP version used")
    
    # Configuration
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    authentication: Optional[Dict[str, Any]] = Field(None, description="Authentication configuration")
    global_headers: Dict[str, str] = Field(default_factory=dict, description="Global headers")


__all__ = [
    "MCPToolType",
    "MCPParameter",
    "MCPTool",
    "MCPToolSet"
]
