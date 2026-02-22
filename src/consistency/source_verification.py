# src/consistency/source_verification.py
"""
Source Verification: Prüft ob eine Quelle einen Claim tatsächlich unterstützt.

Problem:
- Aktuell: Jedes Triple mit source_document_id bekommt volle Konfidenz
- Angreifer könnte beliebige source_id angeben, ohne dass der Text den Claim belegt

Lösung:
- Triple wird zu natürlichsprachlichem Claim konvertiert
- Embedding-Similarity zwischen Claim und Quelltext berechnet
- Niedrige Similarity → Quelle unterstützt Claim nicht → Penalty

Wissenschaftliche Grundlage:
- Natural Language Inference (NLI)
- Textual Entailment
- Claim Verification in Fact-Checking
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import numpy as np

from src.models.entities import Triple

logger = logging.getLogger(__name__)


@dataclass
class SourceVerificationConfig:
    """Konfiguration für Source Verification."""

    # Aktivierung
    enable_source_verification: bool = True

    # Similarity Thresholds
    high_support_threshold: float = 0.5    # Quelle unterstützt stark
    medium_support_threshold: float = 0.3  # Quelle unterstützt teilweise
    low_support_threshold: float = 0.15    # Quelle unterstützt kaum

    # Penalties (Multiplikatoren)
    no_support_penalty: float = 0.5        # Quelle unterstützt nicht: 50% Abzug
    low_support_penalty: float = 0.7       # Schwache Unterstützung: 30% Abzug
    medium_support_bonus: float = 1.0      # Mittlere Unterstützung: neutral
    high_support_bonus: float = 1.1        # Starke Unterstützung: 10% Bonus

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    use_gpu: bool = True

    # Minimum Textlänge für Verification
    min_source_text_length: int = 20

    # Cache
    enable_cache: bool = True


@dataclass
class SourceVerificationResult:
    """Ergebnis der Source Verification."""

    # Hauptergebnis
    is_verified: bool
    similarity_score: float
    support_level: str  # "high", "medium", "low", "none"

    # Confidence Adjustment
    confidence_factor: float  # Multiplikator für Konfidenz

    # Details
    claim_text: str
    source_text_snippet: str  # Erste 200 Zeichen

    # Flags
    source_text_available: bool
    source_text_too_short: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_verified": self.is_verified,
            "similarity_score": self.similarity_score,
            "support_level": self.support_level,
            "confidence_factor": self.confidence_factor,
            "claim_text": self.claim_text,
            "source_text_snippet": self.source_text_snippet,
            "source_text_available": self.source_text_available,
            "source_text_too_short": self.source_text_too_short,
        }


class SourceVerifier:
    """
    Verifiziert ob eine Quelle einen Triple-Claim tatsächlich unterstützt.

    Verwendet Embedding-Similarity zwischen:
    - Dem Triple als natürlichsprachlicher Claim
    - Dem Quelltext (source_text)
    """

    def __init__(self, config: SourceVerificationConfig = None):
        self.config = config or SourceVerificationConfig()
        self._embedding_model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Statistiken
        self.total_verified = 0
        self.verified_supported = 0
        self.verified_unsupported = 0
        self.skipped_no_source = 0

        logger.info(
            f"SourceVerifier initialisiert: "
            f"enabled={self.config.enable_source_verification}, "
            f"high_threshold={self.config.high_support_threshold}"
        )

    @property
    def embedding_model(self):
        """Lazy Loading des Embedding-Modells."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
                self._embedding_model = SentenceTransformer(
                    self.config.embedding_model,
                    device=device
                )
                logger.info(f"Embedding-Modell geladen auf {device}")
            except ImportError:
                logger.warning("sentence-transformers nicht installiert")
                return None
        return self._embedding_model

    def triple_to_claim(self, triple: Triple) -> str:
        """
        Konvertiert ein Triple zu einem natürlichsprachlichen Claim.

        Beispiele:
        - (Berlin, HAUPTSTADT_VON, Deutschland) → "Berlin ist die Hauptstadt von Deutschland"
        - (Einstein, GEBOREN_IN, Ulm) → "Einstein wurde in Ulm geboren"
        - (Q1, HAS_ANSWER, Paris) → "Die Antwort ist Paris"
        """
        subject_name = triple.subject.name
        predicate = triple.predicate.upper()
        object_name = triple.object.name

        # Prädikats-spezifische Templates
        templates = {
            "HAUPTSTADT_VON": f"{subject_name} ist die Hauptstadt von {object_name}",
            "GEBOREN_IN": f"{subject_name} wurde in {object_name} geboren",
            "GESTORBEN_IN": f"{subject_name} ist in {object_name} gestorben",
            "ARBEITET_BEI": f"{subject_name} arbeitet bei {object_name}",
            "WOHNT_IN": f"{subject_name} wohnt in {object_name}",
            "MITGLIED_VON": f"{subject_name} ist Mitglied von {object_name}",
            "TEIL_VON": f"{subject_name} ist Teil von {object_name}",
            "HAS_ANSWER": f"Die Antwort auf die Frage ist {object_name}",
            "ANSWERS": f"Die Antwort lautet {object_name}",
            "RELATED_TO": f"{subject_name} steht in Beziehung zu {object_name}",
            "SUPPORTS": f"{subject_name} unterstützt {object_name}",
        }

        # Verwende Template oder generisches Format
        if predicate in templates:
            return templates[predicate]

        # Generisches Format: Prädikat als Verb
        predicate_readable = predicate.replace("_", " ").lower()
        return f"{subject_name} {predicate_readable} {object_name}"

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Berechnet Embedding für einen Text."""
        if self.embedding_model is None:
            return None

        # Cache Check
        if self.config.enable_cache and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Berechne Embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        # Cache speichern
        if self.config.enable_cache:
            self._embedding_cache[text] = embedding

        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Berechnet Cosine Similarity zwischen zwei Vektoren."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def verify(self, triple: Triple, source_text: str = None) -> SourceVerificationResult:
        """
        Verifiziert ob die Quelle den Triple-Claim unterstützt.

        Args:
            triple: Das zu verifizierende Triple
            source_text: Optionaler Quelltext (falls nicht in triple.source_text)

        Returns:
            SourceVerificationResult mit Similarity und Confidence-Faktor
        """
        # Quelltext ermitteln
        text = source_text or triple.source_text

        # Kein Quelltext verfügbar
        if not text or len(text.strip()) == 0:
            self.skipped_no_source += 1
            return SourceVerificationResult(
                is_verified=False,
                similarity_score=0.0,
                support_level="none",
                confidence_factor=self.config.no_support_penalty,
                claim_text=self.triple_to_claim(triple),
                source_text_snippet="",
                source_text_available=False,
            )

        # Quelltext zu kurz
        if len(text.strip()) < self.config.min_source_text_length:
            self.skipped_no_source += 1
            return SourceVerificationResult(
                is_verified=False,
                similarity_score=0.0,
                support_level="none",
                confidence_factor=self.config.low_support_penalty,
                claim_text=self.triple_to_claim(triple),
                source_text_snippet=text[:200],
                source_text_available=True,
                source_text_too_short=True,
            )

        # Claim generieren
        claim = self.triple_to_claim(triple)

        # Embeddings berechnen
        claim_embedding = self._get_embedding(claim)
        source_embedding = self._get_embedding(text)

        if claim_embedding is None or source_embedding is None:
            logger.warning("Embedding-Berechnung fehlgeschlagen")
            return SourceVerificationResult(
                is_verified=False,
                similarity_score=0.0,
                support_level="unknown",
                confidence_factor=1.0,  # Neutral wenn keine Prüfung möglich
                claim_text=claim,
                source_text_snippet=text[:200],
                source_text_available=True,
            )

        # Similarity berechnen
        similarity = self._cosine_similarity(claim_embedding, source_embedding)

        # Support Level und Confidence Factor bestimmen
        if similarity >= self.config.high_support_threshold:
            support_level = "high"
            confidence_factor = self.config.high_support_bonus
            is_verified = True
        elif similarity >= self.config.medium_support_threshold:
            support_level = "medium"
            confidence_factor = self.config.medium_support_bonus
            is_verified = True
        elif similarity >= self.config.low_support_threshold:
            support_level = "low"
            confidence_factor = self.config.low_support_penalty
            is_verified = False
        else:
            support_level = "none"
            confidence_factor = self.config.no_support_penalty
            is_verified = False

        # Statistiken
        self.total_verified += 1
        if is_verified:
            self.verified_supported += 1
        else:
            self.verified_unsupported += 1

        logger.debug(
            f"Source Verification: claim='{claim[:50]}...' "
            f"similarity={similarity:.3f} support={support_level}"
        )

        return SourceVerificationResult(
            is_verified=is_verified,
            similarity_score=similarity,
            support_level=support_level,
            confidence_factor=confidence_factor,
            claim_text=claim,
            source_text_snippet=text[:200],
            source_text_available=True,
        )

    def verify_batch(
        self,
        triples: List[Triple],
        source_texts: List[str] = None
    ) -> List[SourceVerificationResult]:
        """Verifiziert mehrere Triples."""
        results = []
        texts = source_texts or [None] * len(triples)

        for triple, text in zip(triples, texts):
            results.append(self.verify(triple, text))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        return {
            "total_verified": self.total_verified,
            "verified_supported": self.verified_supported,
            "verified_unsupported": self.verified_unsupported,
            "skipped_no_source": self.skipped_no_source,
            "support_rate": (
                self.verified_supported / self.total_verified
                if self.total_verified > 0 else 0.0
            ),
            "cache_size": len(self._embedding_cache),
        }

    def clear_cache(self):
        """Leert den Embedding-Cache."""
        self._embedding_cache.clear()
