#!/usr/bin/env python3
# evaluation/baselines/random_baseline.py
"""
Random Baseline für Konsistenzprüfung.

Akzeptiert Triples mit einer konfigurierbaren Wahrscheinlichkeit.
Dient als untere Grenze für die Bewertung des Konsistenzmoduls.

Wissenschaftliche Grundlage:
- Random Baseline ist Standard in ML-Evaluation
- Zeigt Mehrwert des eigentlichen Systems gegenüber Zufall
"""

import random
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.models.entities import Triple, ValidationStatus

logger = logging.getLogger(__name__)


@dataclass
class RandomBaselineConfig:
    """Konfiguration für Random Baseline."""
    acceptance_rate: float = 0.5  # Wahrscheinlichkeit für Akzeptanz
    base_confidence: float = 0.5  # Konfidenz für alle Entscheidungen
    confidence_std: float = 0.1   # Standardabweichung für Konfidenz-Varianz
    seed: Optional[int] = None    # Für Reproduzierbarkeit


@dataclass
class BaselineResult:
    """Ergebnis einer Baseline-Validierung."""
    accepted: bool
    confidence: float
    processing_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class RandomBaseline:
    """
    Random Baseline: Akzeptiert Triples zufällig.

    Verwendung als Baseline für wissenschaftliche Evaluation:
    - Wenn unser System nicht besser als Random ist, ist es nutzlos
    - Random hat typischerweise ~50% Accuracy bei balanciertem Dataset

    Beispiel:
        baseline = RandomBaseline(RandomBaselineConfig(acceptance_rate=0.5))
        result = baseline.validate(triple)
        print(f"Accepted: {result.accepted}, Confidence: {result.confidence}")
    """

    def __init__(self, config: RandomBaselineConfig = None):
        self.config = config or RandomBaselineConfig()
        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "total_time_ms": 0.0,
        }

        if self.config.seed is not None:
            random.seed(self.config.seed)

        logger.info(f"RandomBaseline initialisiert: acceptance_rate={self.config.acceptance_rate}")

    def validate(self, triple: Triple, graph_repo: Any = None) -> BaselineResult:
        """
        Validiert ein Triple zufällig.

        Args:
            triple: Das zu validierende Triple
            graph_repo: Ignoriert (für API-Kompatibilität)

        Returns:
            BaselineResult mit zufälliger Entscheidung
        """
        start_time = time.time()

        # Zufällige Entscheidung
        accepted = random.random() < self.config.acceptance_rate

        # Konfidenz mit etwas Varianz
        confidence = max(0.1, min(0.99,
            self.config.base_confidence + random.gauss(0, self.config.confidence_std)
        ))

        processing_time_ms = (time.time() - start_time) * 1000

        # Statistiken aktualisieren
        self.stats["total"] += 1
        if accepted:
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
        self.stats["total_time_ms"] += processing_time_ms

        # Triple Status setzen
        triple.validation_status = ValidationStatus.ACCEPTED if accepted else ValidationStatus.REJECTED

        return BaselineResult(
            accepted=accepted,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            details={
                "baseline": "random",
                "acceptance_rate": self.config.acceptance_rate,
            }
        )

    def validate_batch(self, triples: List[Triple], graph_repo: Any = None) -> List[BaselineResult]:
        """Validiert mehrere Triples."""
        return [self.validate(t, graph_repo) for t in triples]

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        total = self.stats["total"]
        return {
            "baseline_type": "random",
            "config": {
                "acceptance_rate": self.config.acceptance_rate,
                "base_confidence": self.config.base_confidence,
            },
            "total_processed": total,
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "actual_acceptance_rate": self.stats["accepted"] / total if total > 0 else 0,
            "avg_time_ms": self.stats["total_time_ms"] / total if total > 0 else 0,
        }

    def reset(self):
        """Setzt Statistiken zurück."""
        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "total_time_ms": 0.0,
        }


def create_random_baseline(acceptance_rate: float = 0.5, seed: int = None) -> RandomBaseline:
    """Factory-Funktion für RandomBaseline."""
    return RandomBaseline(RandomBaselineConfig(
        acceptance_rate=acceptance_rate,
        seed=seed
    ))
