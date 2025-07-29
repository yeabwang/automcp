# config_loader.py
import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, 'default.yaml')
ENVIRONMENTS = ['development', 'production', 'enterprise']
ENV_VAR = 'SPEC_ANALYZER_ENV'

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def deep_update(base, overrides):
    """Recursively update base dict with overrides."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def get_config(env=None):
    """
    Load the configuration by merging default.yaml with the selected environment config.
    Environment can be set via the SPEC_ANALYZER_ENV environment variable or passed as an argument.
    """
    base = load_yaml(DEFAULT_CONFIG)
    env = env or os.environ.get(ENV_VAR, 'development').lower()
    if env not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env}. Must be one of {ENVIRONMENTS}")
    env_config_path = os.path.join(CONFIG_DIR, f"{env}.yaml")
    if os.path.exists(env_config_path):
        overrides = load_yaml(env_config_path)
        base = deep_update(base, overrides)
    return base

if __name__ == "__main__":
    import pprint
    config = get_config()
    pprint.pprint(config)
