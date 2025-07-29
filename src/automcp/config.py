#!/usr/bin/env python3
"""
AutoMCP Configuration Loader
Supports environment-based configuration with proper inheritance
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import copy

@dataclass
class Config:
    """Configuration object with environment-aware settings"""
    environment: str
    data: Dict[str, Any]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'llm.model')"""
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        target = self.data
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value

class ConfigLoader:
    """
    Loads configuration with environment inheritance:
    1. Load default.yaml as base
    2. Override with environment-specific config (development/production/enterprise)
    3. Apply environment variables
    4. Support runtime environment selection
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            # Default to config directory relative to project root
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
    
    def load_config(self, environment: Optional[str] = None) -> Config:
        """
        Load configuration for specified environment
        Priority: CLI argument > ENV variable > default.yaml setting
        """
        # Determine environment
        env = (
            environment or  # CLI argument
            os.getenv('AUTOMCP_ENVIRONMENT') or  # Environment variable
            self._get_default_environment()  # From default.yaml
        )
        
        print(f"🔧 Loading AutoMCP configuration for environment: {env}")
        
        # Load base configuration
        config_data = self._load_default_config()
        
        # Apply environment-specific overrides
        if env != "default":
            env_config = self._load_environment_config(env)
            config_data = self._merge_configs(config_data, env_config)
        
        # Apply environment variables
        config_data = self._apply_environment_variables(config_data)
        
        # Set final environment
        config_data['environment'] = env
        
        return Config(environment=env, data=config_data)
    
    def _get_default_environment(self) -> str:
        """Get default environment from default.yaml"""
        try:
            default_config = self._load_yaml_file(self.config_dir / "default.yaml")
            return default_config.get('environment', 'development')
        except:
            return 'development'
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load base configuration from default.yaml"""
        default_file = self.config_dir / "default.yaml"
        if not default_file.exists():
            raise FileNotFoundError(f"Default config not found: {default_file}")
        
        return self._load_yaml_file(default_file)
    
    def _load_environment_config(self, environment: str) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env_file = self.config_dir / f"{environment}.yaml"
        if not env_file.exists():
            print(f"⚠️  Environment config not found: {env_file}, using defaults only")
            return {}
        
        return self._load_yaml_file(env_file)
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to load config file {file_path}: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Common environment variables
        env_mappings = {
            'AUTOMCP_LLM_API_KEY': 'llm.api_key',
            'AUTOMCP_LLM_MODEL': 'llm.model',
            'AUTOMCP_LLM_PROVIDER': 'llm.provider',
            'AUTOMCP_LOG_LEVEL': 'logging.level',
            'AUTOMCP_OUTPUT_DIR': 'output.dir',
            'GROQ_API_KEY': 'llm.api_key',  # Support common Groq env var
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                keys = config_key.split('.')
                target = config
                
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                
                target[keys[-1]] = env_value
                print(f"   ✅ Override from {env_var}: {config_key} = {env_value[:20]}...")
        
        return config
    
    def list_available_environments(self) -> list[str]:
        """List available environment configurations"""
        environments = ['default']
        
        for file_path in self.config_dir.glob("*.yaml"):
            if file_path.stem != 'default':
                environments.append(file_path.stem)
        
        return sorted(environments)

# Global config loader instance
_config_loader = None
_current_config = None

def get_config_loader() -> ConfigLoader:
    """Get singleton config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def load_config(environment: Optional[str] = None, force_reload: bool = False) -> Config:
    """Load configuration (cached unless force_reload=True)"""
    global _current_config
    
    if _current_config is None or force_reload or (_current_config.environment != environment and environment is not None):
        loader = get_config_loader()
        _current_config = loader.load_config(environment)
    
    return _current_config

def get_config() -> Config:
    """Get current loaded configuration (loads default if none loaded)"""
    global _current_config
    if _current_config is None:
        _current_config = load_config()
    return _current_config

# Convenience functions for common config access
def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration"""
    config = get_config()
    return config.get('llm', {})

def get_api_key() -> str:
    """Get LLM API key"""
    config = get_config()
    api_key = config.get('llm_client.api_key')
    if not api_key:
        raise ValueError("LLM API key not configured. Set AUTOMCP_LLM_API_KEY environment variable or configure in config files.")
    return api_key

def get_model() -> str:
    """Get LLM model name"""
    config = get_config()
    return config.get('llm_client.model', 'llama-3.1-8b-instant')

def get_endpoint() -> str:
    """Get LLM API endpoint"""
    config = get_config()
    provider = config.get('llm_client.provider', 'groq')
    
    endpoints = {
        'groq': 'https://api.groq.com/openai/v1/chat/completions',
        'openai': 'https://api.openai.com/v1/chat/completions',
        'anthropic': 'https://api.anthropic.com/v1/messages'
    }
    
    return config.get('llm_client.endpoint') or endpoints.get(provider, endpoints['groq'])

def get_output_dir() -> Path:
    """Get output directory"""
    config = get_config()
    output_dir = config.get('output.dir', 'outputs')
    return Path(output_dir)

def list_environments() -> list[str]:
    """List available environments"""
    loader = get_config_loader()
    return loader.list_available_environments()

if __name__ == "__main__":
    # Test configuration loading
    print("🔧 Testing AutoMCP Configuration System")
    print("=" * 50)
    
    loader = ConfigLoader()
    
    print(f"Available environments: {loader.list_available_environments()}")
    
    for env in ['development', 'production', 'enterprise']:
        try:
            config = loader.load_config(env)
            print(f"\n✅ {env.upper()} Configuration:")
            print(f"   LLM Model: {config.get('llm_client.model', 'Not configured')}")
            print(f"   API Key: {'✅ Configured' if config.get('llm_client.api_key') else '❌ Missing'}")
            print(f"   Log Level: {config.get('logging.level', 'Not configured')}")
            print(f"   Output Dir: {config.get('output.dir', 'Not configured')}")
        except Exception as e:
            print(f"❌ Failed to load {env}: {e}")
