#!/usr/bin/env python3
# evaluation/baselines/rules_only_baseline.py
"""
Rules-Only Baseline für Konsistenzprüfung.

Verwendet nur Stage 1 (regelbasierte Validierung) ohne Eskalation.
Zeigt den Beitrag der Embedding- und LLM-Stufen.

Wissenschaftliche Grundlage:
- Regelbasierte Systeme wie SHACL (Knublauch & Kontokostas, 2017)
- Knowledge Graph Completion mit Regeln (Meilicke et al., 2019)
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.models.entities import Triple, ValidationStatus
from src.consistency.base import ConsistencyConfig, ValidationOutcome
from src.consistency.rules.rule_validator import RuleBasedValidator

logger = logging.getLogger(__name__)


@dataclass
class RulesOnlyConfig:
    """Konfiguration für Rules-Only Baseline."""
    # Übernimmt Konfiguration von ConsistencyConfig
    high_confidence_threshold: float = 0.9
    # Akzeptiere auch bei mittlerer Konfidenz (keine Eskalation)
    accept_medium_confidence: bool = True
    medium_confidence_threshold: float = 0.5


@dataclass
class BaselineResult:
    """Ergebnis einer Baseline-Validierung."""
    accepted: bool
    confidence: float
    processing_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class RulesOnlyBaseline:
    """
    Rules-Only Baseline: Nur regelbasierte Validierung.

    Implementiert:
    - Schema-Validierung
    - Kardinalitätsprüfung
    - Domain-Constraints
    - Symmetrie/Asymmetrie-Prüfung

    Zeigt wie viel die Embedding- und LLM-Stufen beitragen.
    """

    def __init__(
        self,
        config: RulesOnlyConfig = None,
        consistency_config: ConsistencyConfig = None,
        embedding_model: Any = None
    ):
        self.config = config or RulesOnlyConfig()
        self.consistency_config = consistency_config or ConsistencyConfig()

        # Initialisiere nur Stage 1
        self.validator = RuleBasedValidator(self.consistency_config, embedding_model)

        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "needs_review": 0,
            "total_time_ms": 0.0,
            "conflicts": 0,
        }

        logger.info("RulesOnlyBaseline initialisiert")

    def validate(self, triple: Triple, graph_repo: Any = None) -> BaselineResult:
        """
        Validiert ein Triple nur mit Regeln.

        Args:
            triple: Das zu validierende Triple
            graph_repo: Graph-Repository für Kardinalitätsprüfung

        Returns:
            BaselineResult mit regelbasierter Entscheidung
        """
        start_time = time.time()

        # Stage 1 Validierung
        result = self.validator.validate(triple, graph_repo)

        # Entscheidung
        if result.outcome == ValidationOutcome.FAIL:
            accepted = False
            confidence = result.confidence
        elif result.outcome == ValidationOutcome.PASS:
            if result.confidence >= self.config.high_confidence_threshold:
                accepted = True
                confidence = result.confidence
            elif self.config.accept_medium_confidence and result.confidence >= self.config.medium_confidence_threshold:
                accepted = True
                confidence = result.confidence
            else:
                # Würde normalerweise eskalieren, aber wir akzeptieren trotzdem
                accepted = True
                confidence = result.confidence
        else:  # UNCERTAIN, CONFLICT
            # Ohne Eskalation müssen wir entscheiden
            # Konservativ: Ablehnen bei Unsicherheit
            accepted = False
            confidence = result.confidence

        processing_time_ms = (time.time() - start_time) * 1000

        # Statistiken
        self.stats["total"] += 1
        if accepted:
            self.stats["accepted"] += 1
            triple.validation_status = ValidationStatus.ACCEPTED
        else:
            self.stats["rejected"] += 1
            triple.validation_status = ValidationStatus.REJECTED
        self.stats["total_time_ms"] += processing_time_ms
        if result.conflicts:
            self.stats["conflicts"] += len(result.conflicts)

        return BaselineResult(
            accepted=accepted,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            details={
                "baseline": "rules_only",
                "stage1_outcome": result.outcome.value,
                "conflicts": [str(c) for c in result.conflicts],
                "rule_details": result.details,
            }
        )

    def validate_batch(self, triples: List[Triple], graph_repo: Any = None) -> List[BaselineResult]:
        """Validiert mehrere Triples."""
        return [self.validate(t, graph_repo) for t in triples]

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        total = self.stats["total"]
        return {
            "baseline_type": "rules_only",
            "config": {
                "high_confidence_threshold": self.config.high_confidence_threshold,
                "accept_medium_confidence": self.config.accept_medium_confidence,
            },
            "total_processed": total,
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "acceptance_rate": self.stats["accepted"] / total if total > 0 else 0,
            "avg_time_ms": self.stats["total_time_ms"] / total if total > 0 else 0,
            "total_conflicts": self.stats["conflicts"],
        }

    def reset(self):
        """Setzt Statistiken zurück."""
        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "needs_review": 0,
            "total_time_ms": 0.0,
            "conflicts": 0,
        }


def create_rules_only_baseline(
    consistency_config: ConsistencyConfig = None,
    embedding_model: Any = None
) -> RulesOnlyBaseline:
    """Factory-Funktion für RulesOnlyBaseline."""
    return RulesOnlyBaseline(
        consistency_config=consistency_config,
        embedding_model=embedding_model
    )
