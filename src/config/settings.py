"""
Configuration management module for FashionMNIST-Analysis.

This module provides utilities for loading and managing configuration from YAML files.
"""

import os
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for managing project settings.
    
    Loads configuration from YAML file and provides attribute-based access to settings.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self.config_path = config_path
        logger.info(f"Configuration loaded from {config_path}")
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute access."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        raise AttributeError(f"Configuration key '{name}' not found")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.
        
        Args:
            key (str): Configuration key (supports dot notation: 'model.architecture')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value (supports dot notation).
        
        Args:
            key (str): Configuration key (e.g., 'model.architecture')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, keys={list(self._config.keys())})"


class ConfigSection:
    """
    Represents a section of configuration as an object.
    
    Allows accessing nested configuration values using dot notation.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration section.
        
        Args:
            config_dict (dict): Configuration dictionary for this section
        """
        self._config_dict = config_dict
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute access."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if name in self._config_dict:
            value = self._config_dict[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        raise AttributeError(f"Configuration key '{name}' not found in section")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with optional default."""
        return self._config_dict.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return self._config_dict.copy()
    
    def __repr__(self) -> str:
        return f"ConfigSection({list(self._config_dict.keys())})"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Config: Configuration object
    """
    return Config(config_path)


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config (Config): Configuration object to save
        output_path (str): Path to save configuration to
    """
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    logger.info(f"Configuration saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    config = load_config("config.yaml")
    print("Loaded configuration:")
    print(f"  Model: {config.model.architecture}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
