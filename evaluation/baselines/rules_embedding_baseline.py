#!/usr/bin/env python3
# evaluation/baselines/rules_embedding_baseline.py
"""
Rules + Embedding Baseline (Stufe 1 + Stufe 2).

Kombiniert regelbasierte Validierung mit Embedding-basierter Duplikaterkennung.
Dies entspricht dem typischen Ablauf ohne LLM-Eskalation.

Komponenten:
- Schema-Validierung
- Kardinalitätsprüfung
- Missing Source Penalty
- Source Verification (Embedding)
- Duplikaterkennung
"""

import sys
sys.path.insert(0, '.')

import time
import logging
from typing import List, Any, Optional
from dataclasses import dataclass

from src.models.entities import Triple, ValidationStatus
from src.consistency.base import ConsistencyConfig, ValidationOutcome
from src.consistency.rules.rule_validator import RuleBasedValidator
from src.consistency.embedding_validator import EmbeddingValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Ergebnis der Validierung."""
    accepted: bool
    confidence: float
    processing_time_ms: float
    stage_reached: int  # 1 = Rules, 2 = Embedding
    reason: str = ""

    # Detaillierte Konfidenz-Aufschlüsselung
    rule_confidence: float = 0.0
    embedding_confidence: float = 0.0
    source_penalty_applied: bool = False
    source_verification_result: Optional[str] = None


class RulesEmbeddingBaseline:
    """
    Baseline die Stufe 1 (Rules) und Stufe 2 (Embedding) kombiniert.

    Konfidenz-Routing:
    - Stufe 1 Konfidenz ≥ 0.9 → ACCEPT (kein Embedding nötig)
    - Stufe 1 FAIL (Kardinalität) → REJECT
    - Stufe 1 Konfidenz < 0.9 → Stufe 2 (Embedding)
    - Kombinierte Konfidenz ≥ 0.7 → ACCEPT
    """

    def __init__(
        self,
        consistency_config: ConsistencyConfig = None,
        embedding_model: Any = None,
        enable_source_verification: bool = True,
        enable_missing_source_penalty: bool = True,
    ):
        self.config = consistency_config or ConsistencyConfig()
        self.embedding_model = embedding_model
        self.enable_source_verification = enable_source_verification
        self.enable_missing_source_penalty = enable_missing_source_penalty

        # Komponenten initialisieren
        self.rule_validator = RuleBasedValidator(self.config)
        self.embedding_validator = EmbeddingValidator(
            config=self.config,
            embedding_model=embedding_model
        )

        # Statistiken
        self.stats = {
            "total": 0,
            "accepted_stage1": 0,
            "rejected_stage1": 0,
            "escalated_to_stage2": 0,
            "accepted_stage2": 0,
            "rejected_stage2": 0,
            "source_penalty_applied": 0,
            "source_verification_low": 0,
        }

        logger.info(f"RulesEmbeddingBaseline initialisiert")
        logger.info(f"  Source Verification: {enable_source_verification}")
        logger.info(f"  Missing Source Penalty: {enable_missing_source_penalty}")

    def validate(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> ValidationResult:
        """
        Validiert ein einzelnes Triple mit Rules + Embedding.
        """
        start_time = time.time()
        self.stats["total"] += 1

        # === STUFE 1: Regelbasierte Validierung ===
        rule_result = self.rule_validator.validate(triple, graph_repo)
        rule_confidence = rule_result.confidence

        # Missing Source Penalty anwenden
        source_penalty_applied = False
        if self.enable_missing_source_penalty:
            if not triple.source_document_id:
                rule_confidence *= self.config.missing_source_penalty
                source_penalty_applied = True
                self.stats["source_penalty_applied"] += 1

        # Kardinalitätsverletzung → REJECT
        if rule_result.outcome == ValidationOutcome.FAIL and "cardinality" in str(rule_result.details):
            self.stats["rejected_stage1"] += 1
            return ValidationResult(
                accepted=False,
                confidence=rule_confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage_reached=1,
                reason="cardinality_violation",
                rule_confidence=rule_confidence,
                source_penalty_applied=source_penalty_applied,
            )

        # Stufe 1 FAIL (andere Gründe) → auch REJECT
        if rule_result.outcome == ValidationOutcome.FAIL:
            self.stats["rejected_stage1"] += 1
            return ValidationResult(
                accepted=False,
                confidence=rule_confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage_reached=1,
                reason="rule_validation_fail",
                rule_confidence=rule_confidence,
                source_penalty_applied=source_penalty_applied,
            )

        # Hohe Konfidenz → ACCEPT ohne Embedding
        if rule_confidence >= 0.9:
            self.stats["accepted_stage1"] += 1
            return ValidationResult(
                accepted=True,
                confidence=rule_confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage_reached=1,
                reason="high_confidence_accept",
                rule_confidence=rule_confidence,
                source_penalty_applied=source_penalty_applied,
            )

        # === STUFE 2: Embedding-basierte Validierung ===
        self.stats["escalated_to_stage2"] += 1

        # Source Verification
        source_verification_result = None
        source_confidence_factor = 1.0
        if self.enable_source_verification and triple.source_text:
            sv_result = self._verify_source(triple)
            source_verification_result = sv_result["support_level"]
            source_confidence_factor = sv_result["confidence_factor"]
            if sv_result["support_level"] in ["LOW", "NONE"]:
                self.stats["source_verification_low"] += 1

        # Embedding-Validierung (Duplikaterkennung)
        embedding_result = self.embedding_validator.validate(triple, graph_repo)
        embedding_confidence = embedding_result.confidence

        # Kombinierte Konfidenz
        # Methode: Gewichteter Durchschnitt mit Source-Faktor
        combined_confidence = (
            0.5 * rule_confidence +
            0.5 * embedding_confidence
        ) * source_confidence_factor

        # Entscheidung basierend auf kombinierter Konfidenz
        if combined_confidence >= 0.7:
            self.stats["accepted_stage2"] += 1
            return ValidationResult(
                accepted=True,
                confidence=combined_confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage_reached=2,
                reason="combined_confidence_accept",
                rule_confidence=rule_confidence,
                embedding_confidence=embedding_confidence,
                source_penalty_applied=source_penalty_applied,
                source_verification_result=source_verification_result,
            )
        else:
            self.stats["rejected_stage2"] += 1
            return ValidationResult(
                accepted=False,
                confidence=combined_confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage_reached=2,
                reason="low_combined_confidence",
                rule_confidence=rule_confidence,
                embedding_confidence=embedding_confidence,
                source_penalty_applied=source_penalty_applied,
                source_verification_result=source_verification_result,
            )

    def _verify_source(self, triple: Triple) -> dict:
        """
        Verifiziert ob der Source-Text den Triple-Claim unterstützt.

        Returns:
            Dict mit support_level und confidence_factor
        """
        if not self.embedding_model or not triple.source_text:
            return {"support_level": "UNKNOWN", "confidence_factor": 1.0}

        try:
            # Claim aus Triple erstellen
            claim = f"{triple.subject.name} {triple.predicate} {triple.object.name}"

            # Embeddings berechnen
            claim_emb = self.embedding_model.encode(claim)
            source_emb = self.embedding_model.encode(triple.source_text[:500])

            # Cosine Similarity
            import numpy as np
            similarity = float(np.dot(claim_emb, source_emb) / (
                np.linalg.norm(claim_emb) * np.linalg.norm(source_emb)
            ))

            # Support Level bestimmen (aus Dokumentation)
            if similarity > 0.5:
                return {"support_level": "HIGH", "confidence_factor": 1.1}
            elif similarity > 0.3:
                return {"support_level": "MEDIUM", "confidence_factor": 1.0}
            elif similarity > 0.15:
                return {"support_level": "LOW", "confidence_factor": 0.7}
            else:
                return {"support_level": "NONE", "confidence_factor": 0.5}

        except Exception as e:
            logger.debug(f"Source verification error: {e}")
            return {"support_level": "ERROR", "confidence_factor": 1.0}

    def validate_batch(
        self,
        triples: List[Triple],
        graph_repo: Any = None
    ) -> List[ValidationResult]:
        """Validiert mehrere Triples."""
        return [self.validate(t, graph_repo) for t in triples]

    def get_statistics(self) -> dict:
        """Gibt Statistiken zurück."""
        total = self.stats["total"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "stage1_accept_rate": self.stats["accepted_stage1"] / total,
            "stage1_reject_rate": self.stats["rejected_stage1"] / total,
            "escalation_rate": self.stats["escalated_to_stage2"] / total,
            "source_penalty_rate": self.stats["source_penalty_applied"] / total,
        }


# Ablation-Varianten
class RulesEmbeddingNoSourcePenalty(RulesEmbeddingBaseline):
    """Variante OHNE Missing Source Penalty."""
    def __init__(self, **kwargs):
        kwargs["enable_missing_source_penalty"] = False
        super().__init__(**kwargs)


class RulesEmbeddingNoSourceVerification(RulesEmbeddingBaseline):
    """Variante OHNE Source Verification."""
    def __init__(self, **kwargs):
        kwargs["enable_source_verification"] = False
        super().__init__(**kwargs)


class RulesEmbeddingMinimal(RulesEmbeddingBaseline):
    """Variante OHNE Source Penalty UND OHNE Source Verification."""
    def __init__(self, **kwargs):
        kwargs["enable_missing_source_penalty"] = False
        kwargs["enable_source_verification"] = False
        super().__init__(**kwargs)
