"""
Configuration loader and validator for the AutoML text classification pipeline.

Simple configuration management that loads YAML files and provides
easy access to configuration parameters.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


class Config:
    """
    Simple configuration class that loads and validates YAML config files.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _validate_config(self) -> None:
        """Basic validation of required configuration sections."""
        required_sections = ['data', 'models', 'hpo', 'training', 'output', 'reproducibility', 'preprocessing']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.dataset_name')
            default: Default value if key not found
            
        Returns:
            Configuration value
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
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.dataset_name')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final key
        config[keys[-1]] = value
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config['data']
    
    @property
    def models(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config['models']
    
    @property
    def hpo(self) -> Dict[str, Any]:
        """Get HPO configuration."""
        return self._config['hpo']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config['training']
    
    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._config['output']
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get('preprocessing', {})
    
    @property
    def reproducibility(self) -> Dict[str, Any]:
        """Get reproducibility configuration."""
        return self._config.get('reproducibility', {})
    
    def get_model_hyperparams(self, model_type: str) -> Dict[str, Any]:
        """
        Get hyperparameter space for a specific model type.
        
        Args:
            model_type: Model type (ffn, cnn, transformer, bert)
            
        Returns:
            Hyperparameter configuration for the model
        """
        return self.get(f'models.hyperparameters.{model_type}', {})
    
    def get_model_types(self) -> List[str]:
        """Get list of model types to train."""
        return self.get('models.model_types', [])
    
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            save_path: Path to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self._config, f, indent=2, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path}, sections={list(self._config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return self.__str__()


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)