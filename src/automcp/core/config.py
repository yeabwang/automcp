"""
Configuration Management System for AutoMCP

This module provides intelligent configuration loading with:
- Environment-specific configuration inheritance
- Environment variable interpolation
- Schema validation
- Hot-reloading capabilities
- Enterprise-grade features

Example:
    ```python
    from automcp.core.config import Config
    
    # Load development config
    config = Config(environment="development")
    
    # Load with custom config file
    config = Config(config_path="custom.yaml")
    
    # Access nested configuration
    batch_size = config.enrichment.batch_size
    llm_provider = config.llm.provider
    ```
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from copy import deepcopy
import re

# Try to import pydantic for validation, fallback to basic dict if not available
try:
    from pydantic import BaseModel, ValidationError, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = dict

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when configuration file is not found."""
    pass

@dataclass
class ConfigSection:
    """A configuration section that supports dot notation access."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        
        # Convert nested dicts to ConfigSection objects
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name: str) -> Any:
        """Support dot notation access."""
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with optional default."""
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        return deepcopy(self._data)

class Config:
    """
    Intelligent configuration management system.
    
    Features:
    - Environment-specific configuration inheritance
    - Environment variable interpolation
    - Dot notation access (config.llm.provider)
    - Hot-reloading capabilities
    - Schema validation
    """
    
    def __init__(
        self, 
        environment: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        config_dir: Optional[Union[str, Path]] = None,
        enable_env_vars: bool = True,
        enable_validation: bool = True
    ):
        """
        Initialize configuration system.
        
        Args:
            environment: Environment name (development, production, enterprise)
            config_path: Direct path to config file (overrides environment loading)
            config_dir: Directory containing config files (default: ./config)
            enable_env_vars: Enable environment variable interpolation
            enable_validation: Enable configuration validation
        """
        self._environment = environment or os.getenv("AUTOMCP_ENV", "development")
        self._config_dir = Path(config_dir or self._get_default_config_dir())
        self._enable_env_vars = enable_env_vars
        self._enable_validation = enable_validation
        self._config_data: Dict[str, Any] = {}
        
        # Load configuration
        if config_path:
            self._load_from_path(Path(config_path))
        else:
            self._load_environment_config()
        
        # Apply environment variable overrides
        if self._enable_env_vars:
            self._apply_env_vars()
        
        # Validate configuration
        if self._enable_validation:
            self._validate_config()
        
        # Convert to ConfigSection objects for dot notation access
        self._create_config_sections()
    
    def _get_default_config_dir(self) -> Path:
        """Get the default configuration directory."""
        # Try to find config directory relative to this file
        current_dir = Path(__file__).parent.parent.parent  # Go up to project root
        config_dir = current_dir / "config"
        
        if config_dir.exists():
            return config_dir
        
        # Fallback to current working directory
        cwd_config = Path.cwd() / "config"
        if cwd_config.exists():
            return cwd_config
        
        # Create config directory if it doesn't exist
        config_dir.mkdir(exist_ok=True)
        return config_dir
    
    def _load_from_path(self, config_path: Path):
        """Load configuration from a specific file path."""
        if not config_path.exists():
            raise ConfigNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration file: {e}")
    
    def _load_environment_config(self):
        """Load environment-specific configuration with inheritance."""
        # Load default configuration first
        default_config_path = self._config_dir / "default.yaml"
        if default_config_path.exists():
            self._load_from_path(default_config_path)
        else:
            logger.warning(f"Default configuration not found: {default_config_path}")
        
        # Load environment-specific configuration
        env_config_path = self._config_dir / f"{self._environment}.yaml"
        if env_config_path.exists():
            try:
                with open(env_config_path, 'r', encoding='utf-8') as f:
                    env_config = yaml.safe_load(f) or {}
                
                # Handle inheritance
                if 'extends' in env_config:
                    parent_config = env_config.pop('extends')
                    self._merge_parent_config(parent_config)
                
                # Merge environment-specific settings
                self._deep_merge(self._config_data, env_config)
                
            except yaml.YAMLError as e:
                raise ConfigError(f"Failed to parse environment configuration: {e}")
            except Exception as e:
                raise ConfigError(f"Failed to load environment configuration: {e}")
        else:
            logger.warning(f"Environment configuration not found: {env_config_path}")
    
    def _merge_parent_config(self, parent_config_name: str):
        """Merge configuration from a parent config file."""
        parent_path = self._config_dir / parent_config_name
        if parent_path.exists():
            try:
                with open(parent_path, 'r', encoding='utf-8') as f:
                    parent_data = yaml.safe_load(f) or {}
                self._deep_merge(self._config_data, parent_data)
            except Exception as e:
                logger.warning(f"Failed to load parent configuration {parent_config_name}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        """Deep merge two dictionaries."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_vars(self):
        """Apply environment variable overrides."""
        # Look for environment variables with AUTOMCP_ prefix
        env_overrides = {}
        
        for env_var, value in os.environ.items():
            if env_var.startswith('AUTOMCP_'):
                # Convert AUTOMCP_LLM_PROVIDER to llm.provider
                config_path = env_var[8:].lower().replace('_', '.')
                self._set_nested_value(env_overrides, config_path, value)
        
        # Merge environment overrides
        if env_overrides:
            self._deep_merge(self._config_data, env_overrides)
        
        # Apply variable interpolation
        self._interpolate_variables()
    
    def _set_nested_value(self, config: Dict, path: str, value: str):
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        final_value = self._convert_type(value)
        current[keys[-1]] = final_value
    
    def _convert_type(self, value: str) -> Any:
        """Convert string values to appropriate Python types."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            # Try integer conversion
            if '.' not in value:
                return int(value)
            # Try float conversion
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _interpolate_variables(self):
        """Interpolate environment variables in configuration values."""
        def interpolate_value(value):
            if isinstance(value, str):
                # Replace ${VAR_NAME} patterns with environment variables
                pattern = r'\\$\\{([^}]+)\\}'
                def replace_var(match):
                    var_name = match.group(1)
                    return os.getenv(var_name, match.group(0))
                return re.sub(pattern, replace_var, value)
            elif isinstance(value, dict):
                return {k: interpolate_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [interpolate_value(item) for item in value]
            return value
        
        self._config_data = interpolate_value(self._config_data)
    
    def _validate_config(self):
        """Validate configuration against schema if available."""
        # Basic validation - check required fields
        required_sections = ['llm_client', 'enrichment', 'output', 'mcp']
        
        for section in required_sections:
            if section not in self._config_data:
                logger.warning(f"Missing required configuration section: {section}")
        
        # Validate LLM configuration
        if 'llm_client' in self._config_data:
            llm_config = self._config_data['llm_client']
            if 'provider' not in llm_config:
                raise ConfigValidationError("LLM provider must be specified in llm_client section")
    
    def _create_config_sections(self):
        """Create ConfigSection objects for dot notation access."""
        for key, value in self._config_data.items():
            # Skip keys that conflict with existing properties/methods
            if key in ('environment', '_environment', '_config_data', '_config_dir'):
                continue
                
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        return self._config_data.get(key, default)
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation."""
        keys = path.split('.')
        current = self._config_data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return deepcopy(self._config_data)
    
    def reload(self):
        """Reload configuration from files."""
        self._config_data = {}
        self._load_environment_config()
        
        if self._enable_env_vars:
            self._apply_env_vars()
        
        if self._enable_validation:
            self._validate_config()
        
        self._create_config_sections()
    
    @property
    def environment(self) -> str:
        """Get the current environment."""
        return self._environment
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"Config(environment='{self._environment}', sections={list(self._config_data.keys())})"

# Convenience function to create a global config instance
_global_config: Optional[Config] = None

def get_config(
    environment: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    reload: bool = False
) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        environment: Environment to load (if not already loaded)
        config_path: Custom config path (if not already loaded) 
        reload: Force reload of configuration
        
    Returns:
        Config: Global configuration instance
    """
    global _global_config
    
    if _global_config is None or reload:
        _global_config = Config(
            environment=environment,
            config_path=config_path
        )
    
    return _global_config

# Export main classes and functions
__all__ = [
    "Config",
    "ConfigSection", 
    "ConfigError",
    "ConfigValidationError",
    "ConfigNotFoundError",
    "get_config"
]
