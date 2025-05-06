import os
import yaml
from typing import Dict

class ConfigManager:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_ollama_config(self) -> Dict:
        return self.config.get('ollama', {})

    def get_vectordb_config(self) -> Dict:
        return self.config.get('vectordb', {})
    
    def get_file_tracker_config(self) -> Dict:
        return self.config.get('file_tracker', {})