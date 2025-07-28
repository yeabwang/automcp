"""
Validation utilities for AutoMCP.

Provides common validation functions for API specifications,
configurations, and data structures.
"""

import re
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_api_spec_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate and load an API specification file.
    
    Args:
        file_path: Path to API specification file
        
    Returns:
        Loaded specification data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"API specification file not found: {path}")
    
    # Determine file type and load
    if path.suffix.lower() in ['.yaml', '.yml']:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    elif path.suffix.lower() == '.json':
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    if not isinstance(data, dict):
        raise ValueError("API specification must be a dictionary/object")
    
    return data


def validate_openapi_spec(spec: Dict[str, Any]) -> List[str]:
    """
    Validate OpenAPI specification structure.
    
    Args:
        spec: OpenAPI specification dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['openapi', 'info', 'paths']
    for field in required_fields:
        if field not in spec:
            errors.append(f"Missing required field: {field}")
    
    # Validate OpenAPI version
    if 'openapi' in spec:
        version = spec['openapi']
        if not isinstance(version, str) or not version.startswith('3.'):
            errors.append(f"Unsupported OpenAPI version: {version}")
    
    # Validate info section
    if 'info' in spec:
        info = spec['info']
        if not isinstance(info, dict):
            errors.append("'info' must be a dictionary")
        else:
            if 'title' not in info:
                errors.append("Missing required field: info.title")
            if 'version' not in info:
                errors.append("Missing required field: info.version")
    
    # Validate paths section
    if 'paths' in spec:
        paths = spec['paths']
        if not isinstance(paths, dict):
            errors.append("'paths' must be a dictionary")
        elif not paths:
            errors.append("'paths' cannot be empty")
    
    return errors


def validate_http_method(method: str) -> bool:
    """
    Validate HTTP method.
    
    Args:
        method: HTTP method string
        
    Returns:
        True if valid HTTP method, False otherwise
    """
    valid_methods = {
        'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 
        'HEAD', 'OPTIONS', 'TRACE', 'CONNECT'
    }
    return method.upper() in valid_methods


def validate_identifier(identifier: str) -> bool:
    """
    Validate if string is a valid identifier (alphanumeric + underscore).
    
    Args:
        identifier: String to validate
        
    Returns:
        True if valid identifier, False otherwise
    """
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, identifier))


def sanitize_identifier(text: str) -> str:
    """
    Convert text to a valid identifier.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized identifier string
    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', text)
    
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized or 'unknown'


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> List[str]:
    """
    Validate data against JSON schema (basic validation).
    
    Args:
        data: Data to validate
        schema: JSON schema dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Basic type validation
    if 'type' in schema:
        expected_type = schema['type']
        actual_type = type(data).__name__
        
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type and not isinstance(data, expected_python_type):
            errors.append(f"Expected type {expected_type}, got {actual_type}")
    
    # Required properties validation for objects
    if isinstance(data, dict) and 'required' in schema:
        for required_prop in schema['required']:
            if required_prop not in data:
                errors.append(f"Missing required property: {required_prop}")
    
    return errors


__all__ = [
    "validate_url",
    "validate_api_spec_file",
    "validate_openapi_spec",
    "validate_http_method",
    "validate_identifier",
    "sanitize_identifier",
    "validate_json_schema"
]
