import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Type
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Base exception for configuration related errors"""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails"""
    pass

class ConfigManager:
    """
    A robust configuration manager that handles loading and validating
    configuration from JSON files with proper error handling.
    
    Features:
    - Automatic creation of default config if none exists
    - Type validation of configuration values
    - Schema validation
    - File operation error handling
    - Thread-safe operations
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir: Directory where config files are stored
        """
        try:
            self.config_dir = Path(config_dir).resolve()
            self.config_dir.mkdir(exist_ok=True, parents=True)
            self._lock = threading.RLock()  # For thread safety
        except (OSError, TypeError) as e:
            logger.error(f"Failed to initialize config directory {config_dir}: {e}")
            raise ConfigError(f"Configuration initialization failed: {e}")
    
    def load_config(self, config_file: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Name of the configuration file
            validate: Whether to validate the loaded configuration
            
        Returns:
            Dict containing the configuration
            
        Raises:
            ConfigError: If the configuration file is invalid or cannot be read
        """
        if not config_file or not isinstance(config_file, str):
            raise ConfigError("Config file name must be a non-empty string")
            
        if not config_file.endswith('.json'):
            config_file = f"{config_file}.json"
            
        config_path = self.config_dir / config_file
        
        try:
            # Create default config if it doesn't exist
            if not config_path.exists():
                logger.info(f"Config file {config_path} not found, creating default")
                return self.create_default_config(config_path)
            
            # Load and validate the config
            with self._lock:  # Ensure thread safety
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            if validate:
                self._validate_config(config)
                
            return config
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in config file {config_path}: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        except OSError as e:
            error_msg = f"Failed to read config file {config_path}: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading config {config_path}: {e}"
            logger.error(error_msg, exc_info=True)
            raise ConfigError(error_msg)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration structure and values.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        required_sections = {
            'analysis_settings': dict,
            'plot_settings': dict,
            'risk_settings': dict
        }
        
        # Check required sections
        for section, expected_type in required_sections.items():
            if section not in config:
                raise ConfigValidationError(f"Missing required section: {section}")
            if not isinstance(config[section], expected_type):
                raise ConfigValidationError(f"Section {section} must be a {expected_type.__name__}")
        
        # Validate analysis settings
        analysis = config['analysis_settings']
        if not all(k in analysis for k in ['default_period', 'technical_indicators']):
            raise ConfigValidationError("Missing required analysis settings")
            
        if not isinstance(analysis['technical_indicators'], list):
            raise ConfigValidationError("technical_indicators must be a list")
    
    def create_default_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Create a default configuration file if it doesn't exist.
        
        Args:
            config_path: Path where the config file should be created
            
        Returns:
            The default configuration dictionary
            
        Raises:
            ConfigError: If default config creation fails
        """
        default_config = {
            "analysis_settings": {
                "default_period": "1y",
                "technical_indicators": ["RSI", "MACD", "BBANDS", "SMA", "VWAP"],
                "risk_free_rate": 0.05,
                "cache_duration": 300,
                "max_retries": 3
            },
            "plot_settings": {
                "style": "fivethirtyeight",
                "figsize": [15, 8],
                "dpi": 100,
                "save_directory": "plots"
            },
            "risk_settings": {
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "high_volatility_threshold": 0.02,
                "volume_spike_threshold": 2.0
            }
        }
        
        try:
            # Ensure parent directories exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the default config
            with self._lock:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Created default config at {config_path}")
            return default_config
            
        except (IOError, OSError) as e:
            error_msg = f"Failed to create default config at {config_path}: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error creating default config: {e}"
            logger.error(error_msg, exc_info=True)
            raise ConfigError(error_msg)
    
    def save_config(self, config: Dict[str, Any], config_file: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            config: Configuration dictionary to save
            config_file: Name of the configuration file
            
        Raises:
            ConfigError: If saving fails
            ConfigValidationError: If config validation fails
        """
        try:
            # Validate before saving
            self._validate_config(config)
            
            if not config_file.endswith('.json'):
                config_file = f"{config_file}.json"
                
            config_path = self.config_dir / config_file
            
            # Create backup if file exists
            if config_path.exists():
                backup_path = config_path.with_suffix(f'.{int(time.time())}.bak')
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Write the config
            with self._lock:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except (IOError, OSError) as e:
            error_msg = f"Failed to save config to {config_file}: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error saving config: {e}"
            logger.error(error_msg, exc_info=True)
            raise ConfigError(error_msg)
    
    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Safely get a value from a nested dictionary using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to the value (e.g., 'analysis_settings.default_period')
            default: Default value if key is not found
            
        Returns:
            The value if found, otherwise the default value
        """
        try:
            keys = key_path.split('.')
            value = config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError, AttributeError):
            return default
            json.dump(default_config, f, indent=2)