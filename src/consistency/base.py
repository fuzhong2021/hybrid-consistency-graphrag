# src/consistency/base.py
"""
Basis-Interfaces für das Konsistenzmodul.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from src.models.entities import Triple, ConflictSet, ValidationResult


class ValidationOutcome(Enum):
    """Mögliche Ergebnisse einer Validierungsstufe."""
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    CONFLICT = "conflict"


@dataclass
class StageResult:
    """Ergebnis einer Validierungsstufe."""
    outcome: ValidationOutcome
    confidence: float
    conflicts: List[ConflictSet] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    
    @property
    def should_escalate(self) -> bool:
        """Soll zur nächsten Stufe eskaliert werden?"""
        return self.outcome in [ValidationOutcome.UNCERTAIN, ValidationOutcome.CONFLICT]


class ValidationStage(ABC):
    """Abstrakte Basisklasse für eine Validierungsstufe."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name der Stufe."""
        pass
    
    @abstractmethod
    def validate(self, triple: Triple, graph_repo: Any) -> StageResult:
        """
        Führt die Validierung durch.
        
        Args:
            triple: Das zu validierende Triple
            graph_repo: Repository für Graph-Zugriff
            
        Returns:
            StageResult mit Outcome und Details
        """
        pass


@dataclass
class ConsistencyConfig:
    """Konfiguration für das Konsistenzmodul."""
    
    # Schwellenwerte für Konfidenz-Routing
    high_confidence_threshold: float = 0.9   # Direkt akzeptieren
    medium_confidence_threshold: float = 0.7  # Zu Stufe 3 eskalieren
    
    # Stufe 2: Embedding
    similarity_threshold: float = 0.85  # iText2KG verwendet 0.85
    
    # Stufe 3: LLM
    llm_model: str = "gpt-4-turbo-preview"
    max_llm_retries: int = 2
    
    # Erlaubte Schema-Elemente
    valid_entity_types: List[str] = field(default_factory=lambda: [
        "Person", "Organisation", "Ort", "Ereignis", "Dokument", "Konzept"
    ])
    valid_relation_types: List[str] = field(default_factory=lambda: [
        # Aus ExtractionConfig (17 Typen)
        "GEBOREN_IN", "GESTORBEN_IN", "WOHNT_IN", "ARBEITET_BEI",
        "STUDIERT_AN", "LEITET", "TEIL_VON", "BEFINDET_SICH_IN",
        "VERHEIRATET_MIT", "KIND_VON", "KENNT",
        "ENTWICKELTE", "ERFAND", "SCHRIEB", "ERHIELT",
        "BETEILIGT_AN", "HAT_BEZIEHUNG_ZU",
        # Häufige LLM-Varianten (7 zusätzliche)
        "GEBOREN_AM", "GESTORBEN_AM", "GRUENDETE", "REGIERT",
        "MITGLIED_VON", "GEWANN", "BEEINFLUSSTE",
    ])

    # Explizites Mapping: LLM-Variante → kanonischer Typ
    relation_type_mapping: Dict[str, str] = field(default_factory=lambda: {
        "REGIERTE": "LEITET",
        "HERRSCHTE_UEBER": "LEITET",
        "FUEHRTE": "LEITET",
        "LEITETE": "LEITET",
        "KOMPONIST_VON": "SCHRIEB",
        "AUTOR_VON": "SCHRIEB",
        "VERFASSTE": "SCHRIEB",
        "GESCHRIEBEN_VON": "SCHRIEB",
        "ENTDECKTE": "ERFAND",
        "ERFUNDEN_VON": "ERFAND",
        "ENTWICKELT_VON": "ENTWICKELTE",
        "GEHOERT_ZU": "TEIL_VON",
        "LIEGT_IN": "BEFINDET_SICH_IN",
        "HAUPTSTADT_VON": "BEFINDET_SICH_IN",
        "GEBOREN_AM": "GEBOREN_IN",
        "GESTORBEN_AM": "GESTORBEN_IN",
        "GEWANN": "ERHIELT",
        "ERHIELT_VON": "ERHIELT",
        "GRUENDETE": "ENTWICKELTE",
        "BEEINFLUSSTE": "HAT_BEZIEHUNG_ZU",
    })

    # Schwellenwert für Embedding-basierte Relationstyp-Ähnlichkeit
    relation_type_similarity_threshold: float = 0.75

    # Kardinalitätsregeln: {relation: {"max": n}} (#3: erweitert)
    cardinality_rules: Dict[str, Dict] = field(default_factory=lambda: {
        "GEBOREN_IN": {"max": 1},
        "GEBOREN_AM": {"max": 1},
        "GESTORBEN_IN": {"max": 1},
        "GESTORBEN_AM": {"max": 1},
        "VERHEIRATET_MIT": {"max": 3},
        "KIND_VON": {"max": 2},
    })

    # #1: Domain-Constraint-Matrix — {relation: {subject_types: [...], object_types: [...]}}
    domain_constraints: Dict[str, Dict[str, list]] = field(default_factory=lambda: {
        "GEBOREN_IN": {"subject_types": ["Person"], "object_types": ["Ort"]},
        "GEBOREN_AM": {"subject_types": ["Person"], "object_types": ["Ort", "Ereignis"]},
        "GESTORBEN_IN": {"subject_types": ["Person"], "object_types": ["Ort"]},
        "GESTORBEN_AM": {"subject_types": ["Person"], "object_types": ["Ort", "Ereignis"]},
        "WOHNT_IN": {"subject_types": ["Person"], "object_types": ["Ort"]},
        "ARBEITET_BEI": {"subject_types": ["Person"], "object_types": ["Organisation"]},
        "STUDIERT_AN": {"subject_types": ["Person"], "object_types": ["Organisation"]},
        "LEITET": {"subject_types": ["Person"], "object_types": ["Organisation", "Ereignis"]},
        "TEIL_VON": {"subject_types": ["Organisation", "Ort", "Konzept"], "object_types": ["Organisation", "Ort", "Konzept"]},
        "BEFINDET_SICH_IN": {"subject_types": ["Ort", "Organisation"], "object_types": ["Ort"]},
        "VERHEIRATET_MIT": {"subject_types": ["Person"], "object_types": ["Person"]},
        "KIND_VON": {"subject_types": ["Person"], "object_types": ["Person"]},
        "KENNT": {"subject_types": ["Person"], "object_types": ["Person"]},
        "HAT_BEZIEHUNG_ZU": {"subject_types": ["Person", "Organisation", "Ort", "Ereignis", "Konzept"], "object_types": ["Person", "Organisation", "Ort", "Ereignis", "Konzept"]},
    })

    # #2: Self-Loop — Relationstypen bei denen Reflexivität erlaubt ist
    allow_reflexive: list = field(default_factory=lambda: ["HAT_BEZIEHUNG_ZU"])

    # #4: Zykluserkennung — Relationstypen die azyklisch sein müssen
    acyclic_relations: list = field(default_factory=lambda: ["TEIL_VON", "BEFINDET_SICH_IN", "KIND_VON"])
    max_cycle_depth: int = 10

    # #5: Symmetrie/Asymmetrie
    symmetric_relations: list = field(default_factory=lambda: ["VERHEIRATET_MIT", "KENNT", "HAT_BEZIEHUNG_ZU"])
    asymmetric_relations: list = field(default_factory=lambda: ["LEITET", "KIND_VON", "TEIL_VON", "ARBEITET_BEI", "STUDIERT_AN"])

    # #7: Provenance-Boost
    enable_provenance_boost: bool = True
    provenance_boost_2_sources: float = 1.1
    provenance_boost_3_plus: float = 1.2

    # #12: Missing Source Penalty
    enable_missing_source_penalty: bool = True
    missing_source_penalty: float = 0.7  # 30% Konfidenz-Abzug (Multiplikator)

    # #13: Source Verification - Prüft ob Quelle den Claim tatsächlich belegt
    enable_source_verification: bool = True
    source_verification_threshold: float = 0.3  # Minimum Similarity für "unterstützt"

    # #8: Anomalie-Erkennung
    enable_anomaly_detection: bool = True
    anomaly_zscore_threshold: float = 3.0
    anomaly_confidence_penalty: float = 0.8

    # #10: TransE
    enable_transe: bool = False
    transe_min_triples: int = 50
    transe_embedding_dim: int = 50
    transe_epochs: int = 100
    transe_learning_rate: float = 0.01
    transe_retrain_interval: int = 50
    transe_anomaly_threshold: float = 2.0

    # #11: Semantischer Trigger (Selektive LLM-Aufrufe für logische Widersprüche)
    enable_semantic_trigger: bool = True
    semantic_trigger_low_confidence: float = 0.6
