# src/graph/memory_repository.py
"""
In-Memory Graph Repository für Tests und Evaluation.

Implementiert das gleiche Interface wie Neo4jRepository,
speichert aber alles im Arbeitsspeicher.

Verwendung:
- Unit Tests
- Intrinsische Evaluation
- Entwicklung ohne Neo4j
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from src.models.entities import (
    Entity, Relation, Triple, EntityType, ValidationStatus
)

logger = logging.getLogger(__name__)


class InMemoryGraphRepository:
    """
    In-Memory Graph Repository für schnelle Tests.

    Speichert Entitäten und Relationen in Python-Dictionaries.
    Unterstützt die gleichen Operationen wie Neo4jRepository.
    """

    def __init__(self):
        """Initialisiert leeren Graph."""
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}

        # Indizes für schnelle Suche
        self._name_index: Dict[str, List[str]] = defaultdict(list)  # name_lower -> [entity_ids]
        self._type_index: Dict[str, List[str]] = defaultdict(list)  # type -> [entity_ids]
        self._fingerprint_index: Dict[str, List[str]] = defaultdict(list)  # fp -> [entity_ids]

        # Relation-Indizes
        self._source_index: Dict[str, List[str]] = defaultdict(list)  # source_id -> [rel_ids]
        self._target_index: Dict[str, List[str]] = defaultdict(list)  # target_id -> [rel_ids]

        logger.info("InMemoryGraphRepository initialisiert")

    # =========================================================================
    # Entity-Operationen
    # =========================================================================

    def create_entity(self, entity: Entity) -> str:
        """Erstellt eine neue Entität."""
        # Speichere Entität
        self._entities[entity.id] = entity

        # Aktualisiere Indizes
        name_key = entity.name.lower()
        self._name_index[name_key].append(entity.id)
        self._type_index[entity.entity_type.value].append(entity.id)

        if entity.fingerprint:
            self._fingerprint_index[entity.fingerprint].append(entity.id)

        logger.debug(f"Entität erstellt: {entity.name} ({entity.id})")
        return entity.id

    def get_entity(self, entity_id: str, include_invalid: bool = False) -> Optional[Entity]:
        """Holt eine Entität nach ID."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return None

        if not include_invalid and entity.invalidated_at is not None:
            return None

        return entity

    def find_by_name(
        self,
        name: str,
        entity_type: EntityType = None,
        include_invalid: bool = False
    ) -> List[Entity]:
        """
        Sucht Entitäten nach Name (case-insensitive, Präfix-Match).

        Args:
            name: Suchstring (wird als Präfix oder Substring behandelt)
            entity_type: Optional - filtert nach Typ
            include_invalid: Wenn True, auch invalidierte Entitäten

        Returns:
            Liste passender Entitäten
        """
        results = []
        name_lower = name.lower()

        for entity_id, entity in self._entities.items():
            # Skip invalidierte
            if not include_invalid and entity.invalidated_at is not None:
                continue

            # Name-Match (Präfix oder Substring)
            if name_lower in entity.name.lower():
                # Typ-Filter
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)

        # Sortiere nach Konfidenz (höchste zuerst)
        results.sort(key=lambda e: e.confidence, reverse=True)
        return results

    def find_duplicates(self, entity: Entity) -> List[Entity]:
        """Findet potentielle Duplikate basierend auf Fingerprint."""
        if not entity.fingerprint:
            return []

        duplicate_ids = self._fingerprint_index.get(entity.fingerprint, [])
        duplicates = []

        for dup_id in duplicate_ids:
            if dup_id == entity.id:
                continue

            dup_entity = self._entities.get(dup_id)
            if dup_entity and dup_entity.invalidated_at is None:
                duplicates.append(dup_entity)

        return duplicates

    def find_all_entities(
        self,
        entity_type: EntityType = None,
        include_invalid: bool = False
    ) -> List[Entity]:
        """Gibt alle Entitäten zurück."""
        results = []

        for entity in self._entities.values():
            if not include_invalid and entity.invalidated_at is not None:
                continue
            if entity_type and entity.entity_type != entity_type:
                continue
            results.append(entity)

        return results

    def invalidate_entity(self, entity_id: str, reason: str = None) -> bool:
        """Invalidiert eine Entität."""
        entity = self._entities.get(entity_id)
        if entity is None or entity.invalidated_at is not None:
            return False

        entity.invalidated_at = datetime.utcnow()
        entity.validation_status = ValidationStatus.REJECTED
        if reason:
            # Speichere Grund in description oder metadata
            if entity.description:
                entity.description += f" [Invalidiert: {reason}]"
            else:
                entity.description = f"Invalidiert: {reason}"

        logger.info(f"Entität invalidiert: {entity_id} - {reason}")
        return True

    # =========================================================================
    # Relation-Operationen
    # =========================================================================

    def create_relation(self, relation: Relation) -> Optional[str]:
        """Erstellt eine Relation zwischen zwei Entitäten."""
        # Prüfe ob Source und Target existieren
        source = self._entities.get(relation.source_id)
        target = self._entities.get(relation.target_id)

        if not source or not target:
            logger.warning(f"Relation konnte nicht erstellt werden: "
                          f"Source oder Target nicht gefunden")
            return None

        if source.invalidated_at or target.invalidated_at:
            logger.warning(f"Relation konnte nicht erstellt werden: "
                          f"Source oder Target invalidiert")
            return None

        # Speichere Relation
        self._relations[relation.id] = relation

        # Aktualisiere Indizes
        self._source_index[relation.source_id].append(relation.id)
        self._target_index[relation.target_id].append(relation.id)

        logger.debug(f"Relation erstellt: {source.name} --[{relation.relation_type}]--> {target.name}")
        return relation.id

    def find_relations(
        self,
        source_id: str = None,
        target_id: str = None,
        relation_type: str = None,
        include_invalid: bool = False
    ) -> List[Dict]:
        """
        Findet Relationen nach Kriterien.

        Returns:
            Liste von Dicts mit Relation-Details
        """
        results = []

        # Bestimme Kandidaten basierend auf Indizes
        if source_id:
            candidate_ids = self._source_index.get(source_id, [])
        elif target_id:
            candidate_ids = self._target_index.get(target_id, [])
        else:
            candidate_ids = list(self._relations.keys())

        for rel_id in candidate_ids:
            relation = self._relations.get(rel_id)
            if not relation:
                continue

            # Filter: Invalidated
            if not include_invalid and relation.invalidated_at is not None:
                continue

            # Filter: Source
            if source_id and relation.source_id != source_id:
                continue

            # Filter: Target
            if target_id and relation.target_id != target_id:
                continue

            # Filter: Relation Type
            if relation_type and relation.relation_type.upper() != relation_type.upper():
                continue

            # Hole Source und Target Entitäten
            source = self._entities.get(relation.source_id)
            target = self._entities.get(relation.target_id)

            results.append({
                "source": {"id": source.id, "name": source.name} if source else None,
                "target": {"id": target.id, "name": target.name} if target else None,
                "rel_type": relation.relation_type,
                "relation": relation
            })

        return results

    def find_relations_by_type(self, rel_type: str) -> List[Dict]:
        """
        Findet Relationen nach Typ.

        Kompatibilitätsmethode für MockGraphRepository-Ersetzung.

        Args:
            rel_type: Relationstyp (z.B. "GEBOREN_IN")

        Returns:
            Liste von Dicts mit Relation-Details
        """
        return self.find_relations(relation_type=rel_type)

    # =========================================================================
    # Triple-Operationen
    # =========================================================================

    def save_triple(self, triple: Triple) -> Tuple[str, Optional[str], str]:
        """
        Speichert ein vollständiges Triple.

        Returns:
            Tuple von (subject_id, relation_id, object_id)
        """
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
        # Prüfe auf Duplikate via Fingerprint
        duplicates = self.find_duplicates(entity)
        if duplicates:
            logger.debug(f"Existierende Entität gefunden: {duplicates[0].name}")
            return duplicates[0].id

        # Erstelle neue Entität
        return self.create_entity(entity)

    def has_relation(
        self,
        subject_name: str,
        predicate: str,
        object_name: str = None
    ) -> bool:
        """
        Prüft ob eine bestimmte Relation existiert.

        Nützlich für Widerspruchserkennung.
        """
        # Finde Subject
        subjects = self.find_by_name(subject_name)
        if not subjects:
            return False

        for subject in subjects:
            relations = self.find_relations(source_id=subject.id)
            for rel in relations:
                if rel["rel_type"].upper() == predicate.upper():
                    if object_name is None:
                        return True
                    if rel["target"] and object_name.lower() in rel["target"]["name"].lower():
                        return True

        return False

    def get_conflicting_relations(
        self,
        subject_name: str,
        predicate: str,
        new_object_name: str
    ) -> List[Dict]:
        """
        Findet existierende Relationen die mit einer neuen kollidieren könnten.

        Für Kardinalitätsprüfung bei GEBOREN_IN, GESTORBEN_IN, etc.
        """
        conflicts = []

        # Finde Subject-Entitäten mit ähnlichem Namen
        subjects = self.find_by_name(subject_name)

        for subject in subjects:
            relations = self.find_relations(
                source_id=subject.id,
                relation_type=predicate
            )

            for rel in relations:
                target = rel.get("target", {})
                target_name = target.get("name", "") if target else ""

                # Wenn anderes Ziel, ist es ein potentieller Konflikt
                if target_name.lower() != new_object_name.lower():
                    conflicts.append({
                        "subject": subject.name,
                        "predicate": predicate,
                        "existing_object": target_name,
                        "new_object": new_object_name
                    })

        return conflicts

    # =========================================================================
    # Statistiken
    # =========================================================================

    def get_stats(self) -> Dict[str, int]:
        """Gibt Graph-Statistiken zurück."""
        total_entities = len(self._entities)
        valid_entities = sum(
            1 for e in self._entities.values()
            if e.invalidated_at is None
        )
        total_relations = len(self._relations)
        valid_relations = sum(
            1 for r in self._relations.values()
            if r.invalidated_at is None
        )

        return {
            "total_entities": total_entities,
            "valid_entities": valid_entities,
            "total_relations": total_relations,
            "valid_relations": valid_relations
        }

    def clear(self):
        """Löscht alle Daten (für Tests)."""
        self._entities.clear()
        self._relations.clear()
        self._name_index.clear()
        self._type_index.clear()
        self._fingerprint_index.clear()
        self._source_index.clear()
        self._target_index.clear()
        logger.info("Graph geleert")

    def __len__(self) -> int:
        """Anzahl der Entitäten."""
        return len(self._entities)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"InMemoryGraphRepository("
                f"entities={stats['valid_entities']}, "
                f"relations={stats['valid_relations']})")
