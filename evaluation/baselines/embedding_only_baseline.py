#!/usr/bin/env python3
# evaluation/baselines/embedding_only_baseline.py
"""
Embedding-Only Baseline für Konsistenzprüfung.

Verwendet nur Stage 2 (embedding-basierte Validierung) ohne Regeln oder LLM.
Zeigt den Beitrag von regelbasierten und LLM-Komponenten.

Wissenschaftliche Grundlage:
- Bordes et al. (2013): TransE - Translating Embeddings
- Nickel et al. (2011): RESCAL - A Three-Way Model for Collective Learning
- Wang et al. (2017): Knowledge Graph Embedding: A Survey
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.models.entities import Triple, ValidationStatus
from src.consistency.base import ConsistencyConfig, ValidationOutcome
from src.consistency.embedding_validator import EmbeddingValidator

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingOnlyConfig:
    """Konfiguration für Embedding-Only Baseline."""
    similarity_threshold: float = 0.85  # iText2KG Standard
    acceptance_threshold: float = 0.85  # Minimum Konfidenz für Akzeptanz
    enable_entity_resolution: bool = True
    enable_anomaly_detection: bool = True
    reject_on_conflict: bool = True     # CONFLICT/UNCERTAIN -> reject (vorher: akzeptiert wenn Conf hoch)


@dataclass
class BaselineResult:
    """Ergebnis einer Baseline-Validierung."""
    accepted: bool
    confidence: float
    processing_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class EmbeddingOnlyBaseline:
    """
    Embedding-Only Baseline: Nur embedding-basierte Validierung.

    Implementiert:
    - Semantische Ähnlichkeitsprüfung
    - Entity Resolution / Duplikaterkennung
    - Embedding-basierte Anomalie-Erkennung
    - Optional: TransE-basierte Plausibilitätsprüfung

    Zeigt wie viel die Regel- und LLM-Komponenten beitragen.
    """

    def __init__(
        self,
        config: EmbeddingOnlyConfig = None,
        consistency_config: ConsistencyConfig = None,
        embedding_model: Any = None
    ):
        self.config = config or EmbeddingOnlyConfig()
        self.consistency_config = consistency_config or ConsistencyConfig(
            similarity_threshold=self.config.similarity_threshold
        )

        if embedding_model is None:
            logger.warning("Kein Embedding-Modell - EmbeddingOnlyBaseline wird eingeschränkt funktionieren")

        # Initialisiere nur Stage 2
        self.validator = EmbeddingValidator(self.consistency_config, embedding_model)
        self.embedding_model = embedding_model

        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "total_time_ms": 0.0,
            "duplicates_found": 0,
            "anomalies_detected": 0,
        }

        logger.info(f"EmbeddingOnlyBaseline initialisiert (model={embedding_model is not None})")

    def validate(self, triple: Triple, graph_repo: Any = None) -> BaselineResult:
        """
        Validiert ein Triple nur mit Embeddings.

        Args:
            triple: Das zu validierende Triple
            graph_repo: Graph-Repository für Entity Resolution

        Returns:
            BaselineResult mit embedding-basierter Entscheidung
        """
        start_time = time.time()

        # Stage 2 Validierung
        result = self.validator.validate(triple, graph_repo)

        # Entscheidung basierend auf Konfidenz
        if result.outcome == ValidationOutcome.FAIL:
            accepted = False
            confidence = result.confidence
        elif result.outcome == ValidationOutcome.PASS:
            accepted = result.confidence >= self.config.acceptance_threshold
            confidence = result.confidence
        elif result.outcome == ValidationOutcome.CONFLICT and self.config.reject_on_conflict:
            # Duplikat oder Widerspruch: nicht akzeptieren, auch wenn Conf hoch
            accepted = False
            confidence = result.confidence
        else:  # UNCERTAIN (oder CONFLICT bei deaktiviertem reject_on_conflict)
            accepted = result.confidence >= self.config.acceptance_threshold
            confidence = result.confidence

        processing_time_ms = (time.time() - start_time) * 1000

        # Entity Resolution durchführen wenn aktiviert
        entities_resolved = 0
        if self.config.enable_entity_resolution and graph_repo:
            try:
                sub_result = self.validator.resolve_entity(triple.subject, graph_repo)
                if sub_result.is_duplicate:
                    entities_resolved += 1
                    self.stats["duplicates_found"] += 1
                obj_result = self.validator.resolve_entity(triple.object, graph_repo)
                if obj_result.is_duplicate:
                    entities_resolved += 1
                    self.stats["duplicates_found"] += 1
            except Exception as e:
                logger.debug(f"Entity Resolution fehlgeschlagen: {e}")

        # Statistiken
        self.stats["total"] += 1
        if accepted:
            self.stats["accepted"] += 1
            triple.validation_status = ValidationStatus.ACCEPTED
        else:
            self.stats["rejected"] += 1
            triple.validation_status = ValidationStatus.REJECTED
        self.stats["total_time_ms"] += processing_time_ms

        if result.details.get("anomaly_detected"):
            self.stats["anomalies_detected"] += 1

        return BaselineResult(
            accepted=accepted,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            details={
                "baseline": "embedding_only",
                "stage2_outcome": result.outcome.value,
                "entities_resolved": entities_resolved,
                "anomaly_detected": result.details.get("anomaly_detected", False),
                "similarity_scores": result.details.get("similarity_scores", {}),
                "embedding_details": result.details,
            }
        )

    def validate_batch(self, triples: List[Triple], graph_repo: Any = None) -> List[BaselineResult]:
        """Validiert mehrere Triples."""
        return [self.validate(t, graph_repo) for t in triples]

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        total = self.stats["total"]
        return {
            "baseline_type": "embedding_only",
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "acceptance_threshold": self.config.acceptance_threshold,
                "enable_entity_resolution": self.config.enable_entity_resolution,
                "reject_on_conflict": self.config.reject_on_conflict,
            },
            "total_processed": total,
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "acceptance_rate": self.stats["accepted"] / total if total > 0 else 0,
            "avg_time_ms": self.stats["total_time_ms"] / total if total > 0 else 0,
            "duplicates_found": self.stats["duplicates_found"],
            "anomalies_detected": self.stats["anomalies_detected"],
        }

    def reset(self):
        """Setzt Statistiken zurück."""
        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "total_time_ms": 0.0,
            "duplicates_found": 0,
            "anomalies_detected": 0,
        }


def create_embedding_only_baseline(
    embedding_model: Any = None,
    consistency_config: ConsistencyConfig = None,
    similarity_threshold: float = 0.85
) -> EmbeddingOnlyBaseline:
    """Factory-Funktion für EmbeddingOnlyBaseline."""
    return EmbeddingOnlyBaseline(
        config=EmbeddingOnlyConfig(similarity_threshold=similarity_threshold),
        consistency_config=consistency_config,
        embedding_model=embedding_model
    )
