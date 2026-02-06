# src/graph/repository.py
"""
Neo4j Repository mit bi-temporalem Modell.
Erweitert den bestehenden Connector um Konsistenz-Features.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Importiere deinen existierenden Connector
from src.graph.neo4j_connector import Neo4jConnector

# Importiere die neuen Modelle
from src.models.entities import (
    Entity, Relation, Triple, EntityType, 
    ValidationStatus
)

logger = logging.getLogger(__name__)


class Neo4jRepository:
    """
    High-Level Repository für Knowledge Graph Operationen.
    Nutzt den bestehenden Neo4jConnector unter der Haube.
    """
    
    def __init__(self, connector: Neo4jConnector):
        """
        Args:
            connector: Dein existierender Neo4jConnector
        """
        self.connector = connector
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Erstellt notwendige Indizes für Performance."""
        indexes = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_fingerprint IF NOT EXISTS FOR (e:Entity) ON (e.fingerprint)",
            "CREATE INDEX entity_valid IF NOT EXISTS FOR (e:Entity) ON (e.invalidated_at)",
        ]
        for idx in indexes:
            try:
                self.connector.execute_query(idx)
            except Exception as e:
                logger.debug(f"Index existiert bereits oder Fehler: {e}")
    
    # =========================================================================
    # Entity-Operationen
    # =========================================================================
    
    def create_entity(self, entity: Entity) -> str:
        """Erstellt eine neue Entität."""
        label = entity.entity_type.value
        props = entity.to_neo4j_properties()
        
        query = f"""
        CREATE (e:Entity:{label} $props)
        RETURN e.id as id
        """
        result = self.connector.execute_query(query, {"props": props})
        logger.info(f"Entität erstellt: {entity.name} ({entity.id})")
        return entity.id
    
    def get_entity(self, entity_id: str, include_invalid: bool = False) -> Optional[Entity]:
        """Holt eine Entität nach ID."""
        where = "WHERE e.id = $id"
        if not include_invalid:
            where += " AND e.invalidated_at IS NULL"
        
        query = f"MATCH (e:Entity) {where} RETURN e"
        result = self.connector.execute_query(query, {"id": entity_id})
        
        if result:
            return self._to_entity(result[0]["e"])
        return None
    
    def find_by_name(self, name: str, entity_type: EntityType = None) -> List[Entity]:
        """Sucht Entitäten nach Name."""
        where = "WHERE toLower(e.name) CONTAINS toLower($name) AND e.invalidated_at IS NULL"
        if entity_type:
            where += f" AND e.entity_type = '{entity_type.value}'"
        
        query = f"MATCH (e:Entity) {where} RETURN e ORDER BY e.confidence DESC"
        result = self.connector.execute_query(query, {"name": name})
        return [self._to_entity(r["e"]) for r in result]
    
    def find_duplicates(self, entity: Entity) -> List[Entity]:
        """Findet potentielle Duplikate basierend auf Fingerprint."""
        query = """
        MATCH (e:Entity {fingerprint: $fp})
        WHERE e.invalidated_at IS NULL AND e.id <> $id
        RETURN e
        """
        result = self.connector.execute_query(query, {
            "fp": entity.fingerprint, 
            "id": entity.id
        })
        return [self._to_entity(r["e"]) for r in result]
    
    def invalidate_entity(self, entity_id: str, reason: str = None) -> bool:
        """
        Invalidiert eine Entität (Graphiti: Edge Invalidation statt Deletion).
        """
        query = """
        MATCH (e:Entity {id: $id})
        WHERE e.invalidated_at IS NULL
        SET e.invalidated_at = $now,
            e.validation_status = 'invalidated',
            e.invalidation_reason = $reason
        RETURN e
        """
        result = self.connector.execute_query(query, {
            "id": entity_id,
            "now": datetime.utcnow().isoformat(),
            "reason": reason
        })
        if result:
            logger.info(f"Entität invalidiert: {entity_id}")
        return len(result) > 0
    
    # =========================================================================
    # Relation-Operationen
    # =========================================================================
    
    def create_relation(self, relation: Relation) -> Optional[str]:
        """Erstellt eine Relation zwischen zwei Entitäten."""
        rel_type = relation.relation_type.upper().replace(" ", "_")
        props = relation.to_neo4j_properties()
        
        query = f"""
        MATCH (s:Entity {{id: $source_id}})
        MATCH (t:Entity {{id: $target_id}})
        WHERE s.invalidated_at IS NULL AND t.invalidated_at IS NULL
        CREATE (s)-[r:{rel_type} $props]->(t)
        RETURN r.id as id
        """
        result = self.connector.execute_query(query, {
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "props": props
        })
        
        if result:
            logger.info(f"Relation erstellt: {relation.source_id} --[{rel_type}]--> {relation.target_id}")
            return relation.id
        return None
    
    def find_relations(self, source_id: str = None, target_id: str = None) -> List[Dict]:
        """Findet Relationen nach Quell- und/oder Ziel-Entität."""
        where_parts = ["r.invalidated_at IS NULL"]
        params = {}
        
        if source_id:
            where_parts.append("s.id = $source_id")
            params["source_id"] = source_id
        if target_id:
            where_parts.append("t.id = $target_id")
            params["target_id"] = target_id
        
        query = f"""
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE {" AND ".join(where_parts)}
        RETURN s, r, t, type(r) as rel_type
        """
        return self.connector.execute_query(query, params)
    
    # =========================================================================
    # Triple-Operationen
    # =========================================================================
    
    def save_triple(self, triple: Triple) -> Tuple[str, str, str]:
        """Speichert ein vollständiges Triple."""
        # Subject: Get or Create
        subject_id = self._get_or_create_entity(triple.subject)
        
        # Object: Get or Create
        object_id = self._get_or_create_entity(triple.object)
        
        # Relation erstellen
        relation = Relation(
            source_id=subject_id,
            target_id=object_id,
            relation_type=triple.predicate,
            confidence=triple.extraction_confidence,
            validation_status=triple.validation_status
        )
        relation_id = self.create_relation(relation)
        
        return (subject_id, relation_id, object_id)
    
    def _get_or_create_entity(self, entity: Entity) -> str:
        """Holt existierende Entität oder erstellt neue."""
        duplicates = self.find_duplicates(entity)
        if duplicates:
            logger.debug(f"Existierende Entität gefunden: {duplicates[0].name}")
            return duplicates[0].id
        return self.create_entity(entity)
    
    # =========================================================================
    # Graph-Traversierung
    # =========================================================================
    
    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> Dict[str, Any]:
        """Holt benachbarte Knoten."""
        query = f"""
        MATCH path = (start:Entity {{id: $id}})-[*1..{max_hops}]-(end:Entity)
        WHERE start.invalidated_at IS NULL
        AND all(n in nodes(path) WHERE n.invalidated_at IS NULL)
        RETURN path
        LIMIT 50
        """
        
        result = self.connector.execute_query(query, {"id": entity_id})
        
        nodes = {}
        edges = []
        for record in result:
            path = record["path"]
            for node in path.nodes:
                nodes[node["id"]] = dict(node)
            for rel in path.relationships:
                edges.append({
                    "source": rel.start_node["id"],
                    "target": rel.end_node["id"],
                    "type": rel.type
                })
        
        return {"nodes": list(nodes.values()), "edges": edges}
    
    # =========================================================================
    # Historische Queries (Graphiti-Modell)
    # =========================================================================
    
    def get_state_at_time(self, entity_id: str, timestamp: datetime) -> Optional[Entity]:
        """Holt Entität im Zustand zu einem bestimmten Zeitpunkt."""
        ts = timestamp.isoformat()
        query = """
        MATCH (e:Entity {id: $id})
        WHERE e.created_at <= $ts
        AND (e.invalidated_at IS NULL OR e.invalidated_at > $ts)
        RETURN e
        """
        result = self.connector.execute_query(query, {"id": entity_id, "ts": ts})
        if result:
            return self._to_entity(result[0]["e"])
        return None
    
    # =========================================================================
    # Hilfsmethoden
    # =========================================================================
    
    def _to_entity(self, node_data: Dict) -> Entity:
        """Konvertiert Neo4j-Node zu Entity-Objekt."""
        return Entity(
            id=node_data.get("id"),
            name=node_data.get("name"),
            entity_type=EntityType.from_string(node_data.get("entity_type", "Unknown")),
            description=node_data.get("description"),
            aliases=node_data.get("aliases", []),
            confidence=node_data.get("confidence", 1.0),
            validation_status=ValidationStatus(node_data.get("validation_status", "pending")),
            source_document=node_data.get("source_document")
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Gibt Graph-Statistiken zurück."""
        query = """
        MATCH (e:Entity)
        WITH count(e) as total,
             count(CASE WHEN e.invalidated_at IS NULL THEN 1 END) as valid
        OPTIONAL MATCH ()-[r]->()
        RETURN total as total_entities, valid as valid_entities, count(r) as relations
        """
        result = self.connector.execute_query(query)
        return result[0] if result else {}
