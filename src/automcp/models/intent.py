"""
Pydantic models for intent structures in AutoMCP.

These models define the structure and validation for intents extracted
from API specifications.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class IntentParameter(BaseModel):
    """Model for intent parameters."""
    
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    required: bool = Field(default=False, description="Whether parameter is required")
    description: Optional[str] = Field(None, description="Parameter description")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[str]] = Field(None, description="Allowed values")


class Intent(BaseModel):
    """Model for API intent representation."""
    
    id: str = Field(..., description="Unique intent identifier")
    path: str = Field(..., description="API endpoint path")
    method: str = Field(..., description="HTTP method")
    summary: Optional[str] = Field(None, description="Intent summary")
    description: Optional[str] = Field(None, description="Detailed description")
    parameters: List[IntentParameter] = Field(default_factory=list, description="Intent parameters")
    responses: Dict[str, Any] = Field(default_factory=dict, description="Expected responses")
    security: List[str] = Field(default_factory=list, description="Security requirements")
    tags: List[str] = Field(default_factory=list, description="Intent tags")
    
    @validator('method')
    def validate_method(cls, v):
        """Validate HTTP method."""
        allowed_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if v.upper() not in allowed_methods:
            raise ValueError(f'Method must be one of {allowed_methods}')
        return v.upper()


class EnrichedIntent(Intent):
    """Enhanced intent with semantic enrichment."""
    
    semantic_description: Optional[str] = Field(None, description="AI-generated semantic description")
    domain_context: Optional[str] = Field(None, description="Domain context")
    user_story: Optional[str] = Field(None, description="Generated user story")
    complexity_score: Optional[float] = Field(None, description="Complexity assessment")
    related_intents: List[str] = Field(default_factory=list, description="Related intent IDs")


__all__ = [
    "IntentParameter",
    "Intent", 
    "EnrichedIntent"
]
