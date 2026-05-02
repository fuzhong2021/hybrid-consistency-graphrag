#!/usr/bin/env python3
# evaluation/baselines/nli_baseline.py
"""
NLI-Only Baseline für Fact Verification.

Verwendet ausschließlich NLI-Modell für Validierung (keine Regeln, kein Embedding).
Ideal für FEVER-ähnliche Datasets.
"""

import sys
sys.path.insert(0, '.')

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.models.entities import Triple, ValidationResult
from src.consistency.nli_validator import NLIValidator, NLIConfig
from src.consistency.base import ValidationOutcome

logger = logging.getLogger(__name__)


@dataclass
class NLIBaselineResult:
    """Ergebnis der NLI-Baseline Validierung."""
    accepted: bool
    confidence: float
    nli_label: str
    nli_score: float
    processing_time_ms: float


class NLIBaseline:
    """
    NLI-Only Baseline.

    Validiert Triples ausschließlich mit NLI-Modell.
    Contradiction → Reject, Entailment/Neutral → Accept.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: str = "cpu",
        contradiction_threshold: float = 0.7
    ):
        self.config = NLIConfig(
            model_name=model_name,
            device=device,
            contradiction_threshold=contradiction_threshold,
        )
        self.validator = NLIValidator(self.config)

    def validate(self, triple: Triple, graph_repo: Any = None) -> NLIBaselineResult:
        """Validiert ein einzelnes Triple mit NLI."""
        start_time = time.time()

        # Prüfe ob source_text vorhanden
        if not triple.source_text or len(triple.source_text) < 10:
            return NLIBaselineResult(
                accepted=True,  # Accept wenn keine Quelle
                confidence=0.5,
                nli_label="no_source",
                nli_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # NLI-Validierung
        result = self.validator.validate(triple, graph_repo)

        accepted = result.outcome != ValidationOutcome.FAIL
        nli_label = result.details.get("nli_label", "unknown")
        nli_score = result.details.get("nli_score", 0.0)

        return NLIBaselineResult(
            accepted=accepted,
            confidence=result.confidence,
            nli_label=nli_label,
            nli_score=nli_score,
            processing_time_ms=result.processing_time_ms
        )

    def validate_batch(
        self,
        triples: List[Triple],
        graph_repo: Any = None
    ) -> List[NLIBaselineResult]:
        """Validiert mehrere Triples."""
        return [self.validate(t, graph_repo) for t in triples]

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        return self.validator.get_statistics()


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLI Baseline Evaluation")
    parser.add_argument("--premise", type=str, help="Evidence text")
    parser.add_argument("--hypothesis", type=str, help="Claim to verify")

    args = parser.parse_args()

    if args.premise and args.hypothesis:
        from src.models.entities import Entity, EntityType

        baseline = NLIBaseline()

        subject = Entity(name="Test", entity_type=EntityType.CONCEPT, source_document="test")
        obj = Entity(name="Claim", entity_type=EntityType.CONCEPT, source_document="test")

        triple = Triple(
            subject=subject,
            predicate="CLAIMS",
            object=obj,
            source_text=args.premise,
            source_document_id="test",
            extraction_confidence=0.8
        )

        result = baseline.validate(triple)

        print(f"\nNLI Baseline Result:")
        print(f"  Accepted: {result.accepted}")
        print(f"  NLI Label: {result.nli_label}")
        print(f"  NLI Score: {result.nli_score:.3f}")
        print(f"  Time: {result.processing_time_ms:.1f}ms")
    else:
        print("Usage: python nli_baseline.py --premise 'Evidence...' --hypothesis 'Claim...'")
