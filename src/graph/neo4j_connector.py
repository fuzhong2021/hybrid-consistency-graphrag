# src/graph/neo4j_connector.py
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnector:
    """Connector für Neo4j Graph-Datenbank Operationen."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j Verbindung hergestellt: {uri}")
    
    def close(self):
        """Schließt die Datenbankverbindung."""
        self.driver.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Führt eine Cypher-Query aus."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Erstellt eine neue Entität im Graphen."""
        query = f"""
        CREATE (n:{entity_type} $properties)
        RETURN elementId(n) as id
        """
        result = self.execute_query(query, {"properties": properties})
        return result[0]["id"] if result else None
    
    def create_relation(
        self, 
        from_id: str, 
        to_id: str, 
        relation_type: str, 
        properties: Optional[Dict] = None
    ) -> bool:
        """Erstellt eine Relation zwischen zwei Entitäten."""
        query = f"""
        MATCH (a), (b)
        WHERE elementId(a) = $from_id AND elementId(b) = $to_id
        CREATE (a)-[r:{relation_type} $properties]->(b)
        RETURN r
        """
        result = self.execute_query(query, {
            "from_id": from_id,
            "to_id": to_id,
            "properties": properties or {}
        })
        return len(result) > 0
    
    def find_entities(
        self, 
        entity_type: Optional[str] = None, 
        properties: Optional[Dict] = None
    ) -> List[Dict]:
        """Sucht Entitäten nach Typ und/oder Eigenschaften."""
        where_clauses = []
        params = {}
        
        if properties:
            for key, value in properties.items():
                where_clauses.append(f"n.{key} = ${key}")
                params[key] = value
        
        where_str = " AND ".join(where_clauses) if where_clauses else "1=1"
        type_str = f":{entity_type}" if entity_type else ""
        
        query = f"""
        MATCH (n{type_str})
        WHERE {where_str}
        RETURN n, elementId(n) as id
        """
        return self.execute_query(query, params)
    
    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> List[Dict]:
        """Holt benachbarte Knoten bis zu einer bestimmten Tiefe."""
        query = f"""
        MATCH path = (n)-[*1..{max_hops}]-(m)
        WHERE elementId(n) = $entity_id
        RETURN path
        """
        return self.execute_query(query, {"entity_id": entity_id})
    
    def clear_database(self):
        """Löscht alle Daten (nur für Entwicklung!)."""
        self.execute_query("MATCH (n) DETACH DELETE n")
        logger.warning("Datenbank wurde geleert!")
