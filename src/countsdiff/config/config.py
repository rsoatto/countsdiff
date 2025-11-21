"""
Configuration handling for SNP Diffusion
"""

import ast
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import neptune


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file
            config_dict: Configuration dictionary (alternative to file)
        """
        if config_path is not None:
            self.config = self.load_from_file(config_path)
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
    
    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Environment variable substitution
        config = Config._substitute_env_vars(config)
        
        print(f"Loaded configuration from {config_path}")
        return config
    
    @staticmethod
    def _normalize_config_literals(obj):
        """
        Recursively convert stringified Python literals (lists, dicts, tuples, numbers, booleans)
        into real Python objects. Leaves non-literal strings unchanged.
        """
        if isinstance(obj, dict):
           return {k: Config._normalize_config_literals(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Config._normalize_config_literals(v) for v in obj]
        if isinstance(obj, str):
            s = obj.strip()
            try:
                # Safely parse only if it's a Python literal; otherwise keep as-is
                parsed = ast.literal_eval(s)
            except Exception:
                return obj
            else:
                return Config._normalize_config_literals(parsed)
        return obj

    @staticmethod
    def load_from_neptune(run_id: str, project_name) -> Dict[str, Any]:
        """Load configuration from a Neptune run"""
        
        print(f"Connecting to Neptune run {project_name}/{run_id}")
        run = neptune.init_run(with_id=run_id, project=project_name, mode='read-only')
        print("Successfully connected to Neptune")
        
        # Extract configuration
        config = {}
        # Try to get the config directly
        config = run["config"].fetch()

        print(f"Loaded configuration from Neptune run {project_name}/{run_id}")
        config = Config._normalize_config_literals(config)
        run.stop()
        return config


    @staticmethod
    def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: Config._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [Config._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            default_value = None
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)
            return os.getenv(env_var, default_value)
        else:
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]) -> None:
        """Update configuration with another config dict"""
        self._deep_update(self.config, other_config)
    
    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                Config._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()


def load_config(config_path: str) -> Config:
    """Convenience function to load configuration"""
    return Config(config_path=config_path)
