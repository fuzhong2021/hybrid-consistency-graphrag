# src/graph/neo4j_exporter.py
"""
Neo4j Exporter für InMemory-Graph.

Ermöglicht den Export eines InMemoryGraphRepository nach Neo4j
für persistente Speicherung und visuelle Inspektion im Neo4j Browser.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.graph.neo4j_connector import Neo4jConnector
from src.graph.repository import Neo4jRepository
from src.graph.memory_repository import InMemoryGraphRepository
from src.models.entities import Entity, Relation

logger = logging.getLogger(__name__)


class Neo4jExporter:
    """
    Exportiert InMemoryGraphRepository nach Neo4j.

    Features:
    - Vollständiger Export aller Entitäten und Relationen
    - Optionales Leeren der Datenbank vor Export
    - Statistiken über exportierte Daten
    - Fehlerbehandlung für robuste Exports
    """

    def __init__(self, connector: Neo4jConnector):
        """
        Initialisiert den Exporter.

        Args:
            connector: Neo4j Connector für Datenbankverbindung
        """
        self.connector = connector
        self.neo4j_repo = Neo4jRepository(connector)
        logger.info("Neo4jExporter initialisiert")

    def export_from_memory(
        self,
        memory_repo: InMemoryGraphRepository,
        clear_first: bool = True,
        include_invalid: bool = False
    ) -> Dict[str, Any]:
        """
        Exportiert InMemoryGraph nach Neo4j.

        Args:
            memory_repo: Quell-Repository im Arbeitsspeicher
            clear_first: Wenn True, lösche Neo4j vorher
            include_invalid: Wenn True, exportiere auch invalidierte Entitäten

        Returns:
            Statistiken über den Export:
            {
                "entities_exported": int,
                "relations_exported": int,
                "entities_skipped": int,
                "relations_skipped": int,
                "errors": List[str],
                "duration_ms": float
            }
        """
        start_time = datetime.now()
        stats = {
            "entities_exported": 0,
            "relations_exported": 0,
            "entities_skipped": 0,
            "relations_skipped": 0,
            "errors": [],
            "duration_ms": 0.0
        }

        logger.info("Starte Export von InMemory nach Neo4j...")

        # Optional: Datenbank leeren
        if clear_first:
            logger.info("Lösche bestehende Daten in Neo4j...")
            try:
                self.connector.clear_database()
                logger.info("Neo4j Datenbank geleert")
            except Exception as e:
                error_msg = f"Fehler beim Leeren der Datenbank: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        # ID-Mapping für Relationen (alte ID -> neue ID falls nötig)
        entity_id_map = {}

        # =====================================================================
        # Schritt 1: Entitäten exportieren
        # =====================================================================
        logger.info("Exportiere Entitäten...")
        entities = memory_repo.find_all_entities(include_invalid=include_invalid)

        for entity in entities:
            try:
                # Erstelle Entity in Neo4j
                self.neo4j_repo.create_entity(entity)
                entity_id_map[entity.id] = entity.id  # Behalte gleiche ID
                stats["entities_exported"] += 1
                logger.debug(f"Entity exportiert: {entity.name} ({entity.id})")
            except Exception as e:
                error_msg = f"Fehler beim Export von Entity '{entity.name}': {e}"
                logger.warning(error_msg)
                stats["errors"].append(error_msg)
                stats["entities_skipped"] += 1

        logger.info(f"Entitäten exportiert: {stats['entities_exported']}")

        # =====================================================================
        # Schritt 2: Relationen exportieren
        # =====================================================================
        logger.info("Exportiere Relationen...")

        # Hole alle Relationen aus dem Memory Repository
        for relation_id, relation in memory_repo._relations.items():
            # Skip invalidierte Relationen wenn nicht gewünscht
            if not include_invalid and relation.invalidated_at is not None:
                stats["relations_skipped"] += 1
                continue

            # Prüfe ob Source und Target exportiert wurden
            if relation.source_id not in entity_id_map:
                logger.debug(f"Relation übersprungen: Source {relation.source_id} nicht gefunden")
                stats["relations_skipped"] += 1
                continue

            if relation.target_id not in entity_id_map:
                logger.debug(f"Relation übersprungen: Target {relation.target_id} nicht gefunden")
                stats["relations_skipped"] += 1
                continue

            try:
                self.neo4j_repo.create_relation(relation)
                stats["relations_exported"] += 1
                logger.debug(f"Relation exportiert: {relation.relation_type}")
            except Exception as e:
                error_msg = f"Fehler beim Export von Relation '{relation.relation_type}': {e}"
                logger.warning(error_msg)
                stats["errors"].append(error_msg)
                stats["relations_skipped"] += 1

        logger.info(f"Relationen exportiert: {stats['relations_exported']}")

        # Berechne Dauer
        duration = (datetime.now() - start_time).total_seconds() * 1000
        stats["duration_ms"] = duration

        # Zusammenfassung loggen
        logger.info(f"\n{'='*50}")
        logger.info("Neo4j Export abgeschlossen:")
        logger.info(f"  Entitäten: {stats['entities_exported']} exportiert, {stats['entities_skipped']} übersprungen")
        logger.info(f"  Relationen: {stats['relations_exported']} exportiert, {stats['relations_skipped']} übersprungen")
        logger.info(f"  Fehler: {len(stats['errors'])}")
        logger.info(f"  Dauer: {duration:.2f}ms")
        logger.info(f"{'='*50}\n")

        return stats

    def verify_export(self, memory_repo: InMemoryGraphRepository) -> Dict[str, Any]:
        """
        Verifiziert den Export durch Vergleich der Statistiken.

        Args:
            memory_repo: Das ursprüngliche InMemory Repository

        Returns:
            Verifikationsergebnis mit Diskrepanzen
        """
        memory_stats = memory_repo.get_stats()
        neo4j_stats = self.neo4j_repo.get_stats()

        result = {
            "memory_entities": memory_stats.get("valid_entities", 0),
            "neo4j_entities": neo4j_stats.get("valid_entities", 0),
            "memory_relations": memory_stats.get("valid_relations", 0),
            "neo4j_relations": neo4j_stats.get("relations", 0),
            "entities_match": False,
            "relations_match": False,
            "verified": False
        }

        result["entities_match"] = (
            result["memory_entities"] == result["neo4j_entities"]
        )
        result["relations_match"] = (
            result["memory_relations"] == result["neo4j_relations"]
        )
        result["verified"] = result["entities_match"] and result["relations_match"]

        if result["verified"]:
            logger.info("Export verifiziert: Alle Daten erfolgreich übertragen")
        else:
            logger.warning(f"Export-Diskrepanzen gefunden: {result}")

        return result

    def get_neo4j_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Neo4j Statistiken zurück."""
        return self.neo4j_repo.get_stats()


def create_exporter_from_config() -> Optional[Neo4jExporter]:
    """
    Factory-Funktion zum Erstellen eines Exporters aus der Konfiguration.

    Returns:
        Neo4jExporter oder None wenn Verbindung fehlschlägt
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        connector = Neo4jConnector(uri, user, password)
        return Neo4jExporter(connector)
    except Exception as e:
        logger.error(f"Konnte Neo4j Exporter nicht erstellen: {e}")
        return None


if __name__ == "__main__":
    # Test-Ausführung
    logging.basicConfig(level=logging.INFO)

    print("\n=== Neo4j Exporter Test ===\n")

    # Erstelle Test-Repository
    from src.models.entities import Entity, EntityType

    memory_repo = InMemoryGraphRepository()

    # Füge Test-Entitäten hinzu
    e1 = Entity(name="Albert Einstein", entity_type=EntityType.PERSON)
    e2 = Entity(name="Physik", entity_type=EntityType.CONCEPT)

    memory_repo.create_entity(e1)
    memory_repo.create_entity(e2)

    print(f"InMemory Stats: {memory_repo.get_stats()}")

    # Erstelle Exporter
    exporter = create_exporter_from_config()

    if exporter:
        # Exportiere
        stats = exporter.export_from_memory(memory_repo, clear_first=True)
        print(f"\nExport Stats: {stats}")

        # Verifiziere
        verification = exporter.verify_export(memory_repo)
        print(f"\nVerification: {verification}")
    else:
        print("Neo4j nicht erreichbar - Test übersprungen")

    print("\n=== Test abgeschlossen ===")
