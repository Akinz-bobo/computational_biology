"""
Configuration management utilities
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the WNV analysis pipeline"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_paths(self):
        """Setup and validate all paths"""
        base_dir = self.config_path.parent.parent
        
        # Convert relative paths to absolute paths
        for key, value in self._config.get('data', {}).items():
            if isinstance(value, str) and not os.path.isabs(value):
                self._config['data'][key] = str(base_dir / value)
        
        for key, value in self._config.get('visualization', {}).items():
            if key.endswith('_dir') and isinstance(value, str) and not os.path.isabs(value):
                self._config['visualization'][key] = str(base_dir / value)
        
        for key, value in self._config.get('results', {}).items():
            if isinstance(value, str) and not os.path.isabs(value):
                self._config['results'][key] = str(base_dir / value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value with dot notation support"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def data_paths(self) -> Dict[str, str]:
        """Get all data paths"""
        return self._config.get('data', {})
    
    @property
    def sequence_config(self) -> Dict[str, Any]:
        """Get sequence processing configuration"""
        return self._config.get('sequence', {})
    
    @property
    def features_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration"""
        return self._config.get('features', {})
    
    @property
    def classification_config(self) -> Dict[str, Any]:
        """Get classification configuration"""
        return self._config.get('classification', {})
    
    @property
    def visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return self._config.get('visualization', {})
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs_to_create = [
            self.get('visualization.figures_dir'),
            self.get('results.models_dir'),
            self.get('results.tables_dir'),
            self.get('results.reports_dir'),
            self.get('data.processed_dir'),
            self.get('data.external_dir'),
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def save(self, output_path: str = None):
        """Save current configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)