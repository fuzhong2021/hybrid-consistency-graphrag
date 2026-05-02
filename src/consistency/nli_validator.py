# src/consistency/nli_validator.py
"""
NLI-basierter Validator für Fact Verification.

Verwendet ein kleines NLI-Modell (~44M Parameter) um zu prüfen,
ob eine Quelle einen Claim unterstützt (entailment), widerspricht
(contradiction) oder neutral ist.

Wissenschaftliche Grundlage:
- Natural Language Inference (NLI)
- Bowman et al. (2015): SNLI Dataset
- Williams et al. (2018): MultiNLI
- Thorne et al. (2018): FEVER

Vorteile gegenüber Embedding-Similarity:
- Erkennt semantische Widersprüche (nicht nur thematische Ähnlichkeit)
- Schnell (~30-50ms pro Inferenz auf CPU)
- Kein LLM/API-Kosten
"""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.models.entities import Triple
from src.consistency.base import ValidationStage, StageResult, ValidationOutcome

logger = logging.getLogger(__name__)


class NLILabel(Enum):
    """NLI-Labels."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass
class NLIConfig:
    """Konfiguration für NLI-Validator."""

    # Modell
    model_name: str = "cross-encoder/nli-deberta-v3-xsmall"  # ~44M Parameter
    device: str = "cpu"  # oder "cuda"

    # Thresholds für Confidence-Mapping
    entailment_threshold: float = 0.7   # Ab wann gilt entailment?
    contradiction_threshold: float = 0.7  # Ab wann gilt contradiction?

    # Confidence-Faktoren (wie bei Source Verification)
    entailment_boost: float = 1.1       # 10% Bonus bei entailment
    neutral_factor: float = 0.9         # 10% Abzug bei neutral
    contradiction_penalty: float = 0.3   # 70% Abzug bei contradiction

    # Cache
    enable_cache: bool = True

    # Minimum Textlänge
    min_text_length: int = 10


@dataclass
class NLIResult:
    """Ergebnis der NLI-Analyse."""

    label: NLILabel
    score: float  # Confidence des NLI-Modells

    # Für Konsistenzmodul
    confidence_factor: float  # Multiplikator für Triple-Confidence
    is_supported: bool  # entailment
    is_contradicted: bool  # contradiction

    # Details
    premise: str  # Source text
    hypothesis: str  # Claim aus Triple

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label.value,
            "score": self.score,
            "confidence_factor": self.confidence_factor,
            "is_supported": self.is_supported,
            "is_contradicted": self.is_contradicted,
            "premise_snippet": self.premise[:100] + "..." if len(self.premise) > 100 else self.premise,
            "hypothesis": self.hypothesis,
        }


class NLIValidator(ValidationStage):
    """
    NLI-basierter Validator für Stage 2.

    Prüft ob source_text den Triple-Claim unterstützt oder widerspricht.
    Verwendet ein kleines, lokales NLI-Modell statt LLM.

    Usage:
        validator = NLIValidator()
        result = validator.validate(triple, graph_repo)

        if result.outcome == ValidationOutcome.FAIL:
            print("Contradiction detected!")
    """

    def __init__(self, config: NLIConfig = None):
        self.config = config or NLIConfig()
        self._pipeline = None
        self._cache: Dict[str, NLIResult] = {}

        # Statistiken
        self.total_checks = 0
        self.entailments = 0
        self.contradictions = 0
        self.neutrals = 0

        logger.info(
            f"NLIValidator initialisiert: model={self.config.model_name}, "
            f"device={self.config.device}"
        )

    @property
    def name(self) -> str:
        return "NLI-Validator"

    @property
    def pipeline(self):
        """Lazy Loading des NLI-Modells."""
        if self._pipeline is None:
            try:
                import os
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

                from transformers import pipeline
                import warnings
                warnings.filterwarnings('ignore', category=FutureWarning)

                device = -1 if self.config.device == "cpu" else 0

                self._pipeline = pipeline(
                    'text-classification',
                    model=self.config.model_name,
                    device=device
                )
                logger.info(f"NLI-Modell geladen: {self.config.model_name}")

            except ImportError as e:
                logger.error(f"transformers nicht installiert: {e}")
                return None
            except Exception as e:
                logger.error(f"Fehler beim Laden des NLI-Modells: {e}")
                return None

        return self._pipeline

    def triple_to_hypothesis(self, triple: Triple) -> str:
        """
        Konvertiert ein Triple zu einer natürlichsprachlichen Hypothese.

        Beispiele:
        - (Einstein, BORN_IN, Ulm) → "Einstein was born in Ulm"
        - (Jackie, DIRECTED_BY, Peter Jackson) → "Jackie was directed by Peter Jackson"
        """
        subject = triple.subject.name
        predicate = triple.predicate.upper()
        obj = triple.object.name

        # Templates für gängige Prädikate
        templates = {
            "BORN_IN": f"{subject} was born in {obj}",
            "DIED_IN": f"{subject} died in {obj}",
            "DIRECTED_BY": f"{subject} was directed by {obj}",
            "WRITTEN_BY": f"{subject} was written by {obj}",
            "LOCATED_IN": f"{subject} is located in {obj}",
            "CAPITAL_OF": f"{subject} is the capital of {obj}",
            "MEMBER_OF": f"{subject} is a member of {obj}",
            "PART_OF": f"{subject} is part of {obj}",
            "HAS_ANSWER": f"The answer is {obj}",
            "ANSWERS": f"The answer is {obj}",
            "CLAIMS": f"{subject} {obj}",  # FEVER-Format
            "RELATED_TO": f"{subject} is related to {obj}",
            "SUPPORTS": f"{subject} supports {obj}",
        }

        if predicate in templates:
            return templates[predicate]

        # Generisches Format
        predicate_readable = predicate.replace("_", " ").lower()
        return f"{subject} {predicate_readable} {obj}"

    def check_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Führt NLI-Check durch.

        Args:
            premise: Der Quelltext (Evidence)
            hypothesis: Die zu prüfende Aussage (Claim)

        Returns:
            NLIResult mit Label, Score und Confidence-Faktor
        """
        # Cache check
        cache_key = f"{premise[:100]}|||{hypothesis}"
        if self.config.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if self.pipeline is None:
            # Fallback wenn Modell nicht verfügbar
            return NLIResult(
                label=NLILabel.NEUTRAL,
                score=0.5,
                confidence_factor=1.0,
                is_supported=False,
                is_contradicted=False,
                premise=premise,
                hypothesis=hypothesis,
            )

        # NLI-Inferenz
        input_text = f"{premise} [SEP] {hypothesis}"
        result = self.pipeline(input_text)[0]

        label_str = result['label'].lower()
        score = result['score']

        # Label mapping
        if label_str == 'entailment':
            label = NLILabel.ENTAILMENT
            self.entailments += 1
        elif label_str == 'contradiction':
            label = NLILabel.CONTRADICTION
            self.contradictions += 1
        else:
            label = NLILabel.NEUTRAL
            self.neutrals += 1

        self.total_checks += 1

        # Confidence-Faktor berechnen
        if label == NLILabel.ENTAILMENT and score >= self.config.entailment_threshold:
            confidence_factor = self.config.entailment_boost
            is_supported = True
            is_contradicted = False
        elif label == NLILabel.CONTRADICTION and score >= self.config.contradiction_threshold:
            confidence_factor = self.config.contradiction_penalty
            is_supported = False
            is_contradicted = True
        else:
            confidence_factor = self.config.neutral_factor
            is_supported = False
            is_contradicted = False

        nli_result = NLIResult(
            label=label,
            score=score,
            confidence_factor=confidence_factor,
            is_supported=is_supported,
            is_contradicted=is_contradicted,
            premise=premise,
            hypothesis=hypothesis,
        )

        # Cache
        if self.config.enable_cache:
            self._cache[cache_key] = nli_result

        return nli_result

    def validate(self, triple: Triple, graph_repo: Any) -> StageResult:
        """
        Validiert ein Triple mit NLI.

        Args:
            triple: Das zu validierende Triple
            graph_repo: Graph Repository (nicht verwendet)

        Returns:
            StageResult mit Outcome basierend auf NLI
        """
        import time
        start_time = time.time()

        # Prüfe ob source_text vorhanden
        if not triple.source_text or len(triple.source_text) < self.config.min_text_length:
            return StageResult(
                outcome=ValidationOutcome.UNCERTAIN,
                confidence=0.5,
                details={
                    "reason": "no_source_text",
                    "message": "Kein oder zu kurzer Quelltext für NLI-Analyse"
                },
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Triple zu Hypothese konvertieren
        hypothesis = self.triple_to_hypothesis(triple)
        premise = triple.source_text

        # NLI-Check
        nli_result = self.check_nli(premise, hypothesis)

        # Outcome bestimmen
        if nli_result.is_contradicted:
            outcome = ValidationOutcome.FAIL
            confidence = 1.0 - nli_result.score  # Niedrige Confidence bei Widerspruch
        elif nli_result.is_supported:
            outcome = ValidationOutcome.PASS
            confidence = min(1.0, triple.extraction_confidence * nli_result.confidence_factor)
        else:
            outcome = ValidationOutcome.UNCERTAIN
            confidence = triple.extraction_confidence * nli_result.confidence_factor

        processing_time = (time.time() - start_time) * 1000

        return StageResult(
            outcome=outcome,
            confidence=confidence,
            details={
                "nli_label": nli_result.label.value,
                "nli_score": nli_result.score,
                "confidence_factor": nli_result.confidence_factor,
                "hypothesis": hypothesis,
                "premise_snippet": premise[:100] + "..." if len(premise) > 100 else premise,
            },
            processing_time_ms=processing_time
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        if self.total_checks == 0:
            return {"total": 0}

        return {
            "total_checks": self.total_checks,
            "entailments": self.entailments,
            "contradictions": self.contradictions,
            "neutrals": self.neutrals,
            "entailment_rate": self.entailments / self.total_checks,
            "contradiction_rate": self.contradictions / self.total_checks,
            "neutral_rate": self.neutrals / self.total_checks,
            "cache_size": len(self._cache),
        }


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLI Validator Test")
    parser.add_argument("--premise", type=str, required=True, help="Source text")
    parser.add_argument("--hypothesis", type=str, required=True, help="Claim to verify")

    args = parser.parse_args()

    validator = NLIValidator()
    result = validator.check_nli(args.premise, args.hypothesis)

    print(f"\nNLI Result:")
    print(f"  Label: {result.label.value}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Confidence Factor: {result.confidence_factor:.2f}")
    print(f"  Is Supported: {result.is_supported}")
    print(f"  Is Contradicted: {result.is_contradicted}")
