# src/graph/repository_factory.py
"""
Factory für Graph-Repository-Erstellung.

Ermöglicht einfaches Umschalten zwischen:
- InMemoryGraphRepository (Test/Entwicklung)
- Neo4jRepository (Produktion/Docker)

Verwendung:
    from src.graph.repository_factory import create_repository

    # Automatisch basierend auf Umgebungsvariablen
    repo = create_repository()

    # Explizit Neo4j
    repo = create_repository(use_neo4j=True)

    # Explizit In-Memory
    repo = create_repository(use_neo4j=False)
"""

import os
import logging
from typing import Union

from src.graph.memory_repository import InMemoryGraphRepository
from src.graph.repository import Neo4jRepository
from src.graph.neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)

# Type alias für beide Repository-Typen
GraphRepository = Union[InMemoryGraphRepository, Neo4jRepository]


def create_repository(use_neo4j: bool = None) -> GraphRepository:
    """
    Erstellt ein Graph-Repository basierend auf Konfiguration.

    Args:
        use_neo4j: Wenn True, Neo4jRepository verwenden.
                   Wenn False, InMemoryGraphRepository verwenden.
                   Wenn None, aus Umgebungsvariable USE_NEO4J lesen.

    Returns:
        InMemoryGraphRepository oder Neo4jRepository

    Environment Variables:
        USE_NEO4J: "true" oder "false" (default: "false")
        NEO4J_URI: Neo4j Bolt URI (default: "bolt://localhost:7687")
        NEO4J_USER: Neo4j Benutzer (default: "neo4j")
        NEO4J_PASSWORD: Neo4j Passwort (required wenn USE_NEO4J=true)
    """
    # Bestimme ob Neo4j verwendet werden soll
    if use_neo4j is None:
        use_neo4j = os.getenv("USE_NEO4J", "false").lower() == "true"

    if use_neo4j:
        return _create_neo4j_repository()
    else:
        return _create_memory_repository()


def _create_neo4j_repository() -> Neo4jRepository:
    """Erstellt ein Neo4jRepository mit Verbindungsparametern aus der Umgebung."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise ValueError(
            "NEO4J_PASSWORD muss gesetzt sein wenn USE_NEO4J=true. "
            "Setze die Umgebungsvariable oder verwende .env Datei."
        )

    logger.info(f"Erstelle Neo4jRepository: {uri}")

    try:
        connector = Neo4jConnector(uri, user, password)
        repository = Neo4jRepository(connector)
        logger.info("Neo4jRepository erfolgreich erstellt")
        return repository
    except Exception as e:
        logger.error(f"Neo4j-Verbindung fehlgeschlagen: {e}")
        raise


def _create_memory_repository() -> InMemoryGraphRepository:
    """Erstellt ein InMemoryGraphRepository."""
    logger.info("Erstelle InMemoryGraphRepository")
    return InMemoryGraphRepository()


def get_repository_from_config(config: dict) -> GraphRepository:
    """
    Erstellt ein Repository basierend auf einem Konfigurationsdict.

    Args:
        config: Dict mit möglichen Keys:
            - use_neo4j: bool
            - neo4j_uri: str
            - neo4j_user: str
            - neo4j_password: str

    Returns:
        Graph-Repository
    """
    use_neo4j = config.get("use_neo4j", False)

    if use_neo4j:
        uri = config.get("neo4j_uri", "bolt://localhost:7687")
        user = config.get("neo4j_user", "neo4j")
        password = config.get("neo4j_password")

        if not password:
            raise ValueError("neo4j_password muss in config gesetzt sein")

        connector = Neo4jConnector(uri, user, password)
        return Neo4jRepository(connector)

    return InMemoryGraphRepository()


# Convenience-Funktion für Tests
def create_test_repository() -> InMemoryGraphRepository:
    """Erstellt ein leeres InMemoryGraphRepository für Tests."""
    return InMemoryGraphRepository()


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    print("=== Repository Factory Test ===\n")

    # Test InMemory (default)
    repo = create_repository(use_neo4j=False)
    print(f"Created: {repo}")
    print(f"Type: {type(repo).__name__}")
