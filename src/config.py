# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# .env laden
load_dotenv()

class Config:
    """Zentrale Konfigurationsklasse für das GraphRAG-System."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self._load_config()
        self._load_env()
    
    def _load_config(self):
        """Lädt die YAML-Konfiguration."""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def _load_env(self):
        """Lädt Umgebungsvariablen."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    @property
    def llm(self) -> dict:
        return self._config.get("llm", {})
    
    @property
    def graph(self) -> dict:
        return self._config.get("graph", {})
    
    @property
    def consistency(self) -> dict:
        return self._config.get("consistency", {})
    
    @property
    def retrieval(self) -> dict:
        return self._config.get("retrieval", {})
