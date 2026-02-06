# src/models/entities.py
"""
Datenmodelle für das hybride Konsistenzmodul.
Bi-temporales Modell basierend auf Graphiti (Rasmussen et al., 2025).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import hashlib


class EntityType(Enum):
    """Entitätstypen für den Knowledge Graph."""
    PERSON = "Person"
    ORGANIZATION = "Organisation"
    LOCATION = "Ort"
    EVENT = "Ereignis"
    DOCUMENT = "Dokument"
    CONCEPT = "Konzept"
    UNKNOWN = "Unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "EntityType":
        mapping = {
            "person": cls.PERSON,
            "organisation": cls.ORGANIZATION,
            "organization": cls.ORGANIZATION,
            "ort": cls.LOCATION,
            "location": cls.LOCATION,
            "ereignis": cls.EVENT,
            "event": cls.EVENT,
        }
        return mapping.get(value.lower(), cls.UNKNOWN)


class ValidationStatus(Enum):
    """Status der Validierung."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONFLICTING = "conflicting"
    NEEDS_REVIEW = "needs_review"
    INVALIDATED = "invalidated"


class ConflictType(Enum):
    """Konflikttypen nach Nentidis et al. (2025)."""
    DUPLICATE_ENTITY = "duplicate_entity"
    CONTRADICTORY_RELATION = "contradictory_relation"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    SCHEMA_VIOLATION = "schema_violation"


@dataclass
class Entity:
    """
    Entität mit bi-temporalem Support.
    
    - created_at/invalidated_at: Systemzeit
    - valid_from/valid_until: Realwelt-Gültigkeit
    """
    name: str
    entity_type: EntityType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    # Bi-temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    invalidated_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Validierung
    confidence: float = 1.0
    validation_status: ValidationStatus = ValidationStatus.PENDING
    source_document: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        return self.invalidated_at is None
    
    @property
    def fingerprint(self) -> str:
        """Hash für Duplikaterkennung."""
        content = f"{self.name.lower()}:{self.entity_type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def invalidate(self, reason: str = None):
        """Invalidiert statt löscht (Graphiti-Modell)."""
        self.invalidated_at = datetime.utcnow()
        self.validation_status = ValidationStatus.INVALIDATED
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Konvertiert zu Neo4j-Properties."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "validation_status": self.validation_status.value,
            "created_at": self.created_at.isoformat(),
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
            "fingerprint": self.fingerprint,
        }


@dataclass
class Relation:
    """Relation zwischen zwei Entitäten."""
    source_id: str
    target_id: str
    relation_type: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    invalidated_at: Optional[datetime] = None
    validation_status: ValidationStatus = ValidationStatus.PENDING
    source_document_id: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.invalidated_at is None

    def to_neo4j_properties(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "validation_status": self.validation_status.value,
            "created_at": self.created_at.isoformat(),
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
            "source_document_id": self.source_document_id,
        }


@dataclass
class Triple:
    """Ein Tripel (Subject-Predicate-Object) mit Validierungskontext."""
    subject: Entity
    predicate: str
    object: Entity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    source_text: Optional[str] = None
    source_chunk_id: Optional[str] = None
    source_document_id: Optional[str] = None
    extraction_confidence: float = 1.0
    
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List["ConflictSet"] = field(default_factory=list)
    
    def add_validation_event(self, stage: str, passed: bool, confidence: float, details: Dict = None):
        self.validation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "passed": passed,
            "confidence": confidence,
            "details": details or {}
        })
    
    def __repr__(self) -> str:
        return f"Triple({self.subject.name} --[{self.predicate}]--> {self.object.name})"


@dataclass
class ConflictSet:
    """Ein Konflikt zwischen Tripeln/Entitäten."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = ConflictType.DUPLICATE_ENTITY
    conflicting_items: List[Any] = field(default_factory=list)
    description: str = ""
    severity: float = 0.5
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    @property
    def is_resolved(self) -> bool:
        return self.resolution is not None


@dataclass
class ValidationResult:
    """Ergebnis einer Validierungsstufe."""
    triple_id: str
    stage: str
    passed: bool
    confidence: float
    conflicts: List[ConflictSet] = field(default_factory=list)
    processing_time_ms: float = 0.0


class MergeStrategy(Enum):
    """Strategien für Entity-Merging."""
    NAME_MATCH = "name_match"         # Exakte Namensübereinstimmung
    EMBEDDING_MATCH = "embedding_match"  # Embedding-basierte Ähnlichkeit
    HYBRID = "hybrid"                  # Kombination aus beiden
    MANUAL = "manual"                  # Manuelle Zusammenführung


@dataclass
class EntityResolutionResult:
    """
    Ergebnis der Entity Resolution.

    Repräsentiert das Ergebnis einer Duplikaterkennung und
    optional einer Merge-Operation.
    """
    is_duplicate: bool
    canonical_entity: Optional["Entity"] = None  # Die "Master"-Entität nach Merge
    merged_from: List["Entity"] = field(default_factory=list)  # Alle gemergten Entitäten
    similarity_score: float = 0.0
    merge_strategy: MergeStrategy = MergeStrategy.EMBEDDING_MATCH

    # Zusätzliche Details
    name_similarity: float = 0.0      # Namensähnlichkeit (Levenshtein etc.)
    embedding_similarity: float = 0.0  # Cosine Similarity der Embeddings
    type_match: bool = True            # Haben beide denselben Typ?

    # Reasoning (für Debugging und Evaluation)
    reasoning: str = ""

    @property
    def weighted_similarity(self) -> float:
        """
        Gewichtete Gesamtähnlichkeit (iText2KG-Ansatz).

        α = 0.6 für Name-Gewichtung (nach Lairgi et al., 2024)
        """
        alpha = 0.6
        return alpha * self.name_similarity + (1 - alpha) * self.embedding_similarity

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Serialisierung."""
        return {
            "is_duplicate": self.is_duplicate,
            "canonical_entity_id": self.canonical_entity.id if self.canonical_entity else None,
            "canonical_entity_name": self.canonical_entity.name if self.canonical_entity else None,
            "merged_from_ids": [e.id for e in self.merged_from],
            "similarity_score": self.similarity_score,
            "merge_strategy": self.merge_strategy.value,
            "name_similarity": self.name_similarity,
            "embedding_similarity": self.embedding_similarity,
            "type_match": self.type_match,
            "weighted_similarity": self.weighted_similarity,
            "reasoning": self.reasoning,
        }


def merge_entities(
    entities: List["Entity"],
    strategy: MergeStrategy = MergeStrategy.HYBRID
) -> "Entity":
    """
    Führt mehrere Entitäten zu einer kanonischen Entität zusammen.

    Merge-Regeln:
    - Name: Längster Name oder häufigster
    - Typ: Häufigster Typ
    - Aliases: Union aller Namen/Aliases
    - Properties: Union mit Konfliktauflösung (neueste gewinnt)
    - Konfidenz: Maximum
    - Temporal: Frühestes valid_from, spätestes valid_until

    Args:
        entities: Liste der zu mergenden Entitäten
        strategy: Merge-Strategie für Logging

    Returns:
        Neue kanonische Entity
    """
    if not entities:
        raise ValueError("Mindestens eine Entity erforderlich")

    if len(entities) == 1:
        return entities[0]

    # Sortiere nach Konfidenz (höchste zuerst)
    sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
    base = sorted_entities[0]

    # Sammle alle Namen und Aliases
    all_names = set()
    all_aliases = set()
    for e in entities:
        all_names.add(e.name)
        all_aliases.update(e.aliases)

    # Wähle kanonischen Namen (längster oder von Entity mit höchster Konfidenz)
    canonical_name = max(all_names, key=len)

    # Alle anderen Namen werden Aliases
    all_aliases.update(all_names)
    all_aliases.discard(canonical_name)

    # Merge Properties (neueste Entity gewinnt bei Konflikten)
    merged_properties = {}
    for e in sorted(entities, key=lambda x: x.created_at):
        merged_properties.update(e.properties)

    # Temporal-Range erweitern
    valid_from_dates = [e.valid_from for e in entities if e.valid_from]
    valid_until_dates = [e.valid_until for e in entities if e.valid_until]

    # Erstelle neue kanonische Entity
    merged = Entity(
        name=canonical_name,
        entity_type=base.entity_type,
        id=base.id,  # Behalte ID der Basis-Entity
        description=base.description or next(
            (e.description for e in entities if e.description), None
        ),
        aliases=list(all_aliases),
        properties=merged_properties,
        embedding=base.embedding,  # Embedding von höchster Konfidenz
        created_at=min(e.created_at for e in entities),
        valid_from=min(valid_from_dates) if valid_from_dates else None,
        valid_until=max(valid_until_dates) if valid_until_dates else None,
        confidence=max(e.confidence for e in entities),
        validation_status=base.validation_status,
        source_document=base.source_document,
    )

    return merged
