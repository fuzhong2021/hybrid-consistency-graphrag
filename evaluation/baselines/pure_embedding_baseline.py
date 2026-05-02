#!/usr/bin/env python3
# evaluation/baselines/pure_embedding_baseline.py
"""
Pure-Embedding-Baseline: NUR Cosine-Similarity für Duplikaterkennung.

Keine Provenance, keine Anomalie-Erkennung, keine Graph-basierte
Widerspruchsprüfung, kein Relationstyp-Matching. Nur:
  1. Encode Triple als Sentence-Embedding
  2. Vergleiche mit bestehenden Graph-Triples via Cosine-Similarity
  3. Wenn Similarity >= threshold → Duplikat → REJECT
  4. Sonst → ACCEPT

Zeigt den isolierten Beitrag von Embedding-Similarity ohne die
zusätzlichen Prüfungen, die im EmbeddingValidator enthalten sind.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.models.entities import Triple, ValidationStatus

logger = logging.getLogger(__name__)


@dataclass
class PureEmbeddingConfig:
    similarity_threshold: float = 0.85
    acceptance_threshold: float = 0.5


@dataclass
class BaselineResult:
    accepted: bool
    confidence: float
    processing_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class PureEmbeddingBaseline:
    """
    Reine Embedding-Duplikaterkennung ohne jede weitere Prüfung.

    Vergleicht eingehende Triples per Cosine-Similarity mit dem bestehenden
    Graph. Erkennt nur semantische Duplikate — keine Kardinalität, keine
    Provenance, keine Anomalien, keine Widersprüche.
    """

    def __init__(self, config: PureEmbeddingConfig = None, embedding_model: Any = None):
        self.config = config or PureEmbeddingConfig()
        self.embedding_model = embedding_model
        self.stats = {
            "total": 0, "accepted": 0, "rejected": 0,
            "total_time_ms": 0.0, "duplicates_found": 0,
        }

    def _triple_to_text(self, triple: Triple) -> str:
        subj = triple.subject.name if hasattr(triple.subject, 'name') else str(triple.subject)
        obj = triple.object.name if hasattr(triple.object, 'name') else str(triple.object)
        return f"{subj} {triple.predicate} {obj}"

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if self.embedding_model is None:
            return None
        try:
            emb = self.embedding_model.encode(text, convert_to_numpy=True)
            return emb / (np.linalg.norm(emb) + 1e-9)
        except Exception as e:
            logger.debug(f"Encoding failed: {e}")
            return None

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def validate(self, triple: Triple, graph_repo: Any = None) -> BaselineResult:
        start_time = time.time()

        if self.embedding_model is None or graph_repo is None:
            self.stats["total"] += 1
            self.stats["accepted"] += 1
            triple.validation_status = ValidationStatus.ACCEPTED
            return BaselineResult(accepted=True, confidence=self.config.acceptance_threshold,
                                  processing_time_ms=0.0, details={"reason": "no_model_or_graph"})

        new_emb = self._encode(self._triple_to_text(triple))
        if new_emb is None:
            self.stats["total"] += 1
            self.stats["accepted"] += 1
            triple.validation_status = ValidationStatus.ACCEPTED
            return BaselineResult(accepted=True, confidence=self.config.acceptance_threshold,
                                  processing_time_ms=0.0, details={"reason": "encoding_failed"})

        is_duplicate = False
        max_sim = 0.0

        existing_triples = getattr(graph_repo, 'get_all_triples',
                                   lambda: getattr(graph_repo, 'triples', []))()
        if not callable(existing_triples):
            existing_triples = getattr(graph_repo, 'triples', [])

        for existing in existing_triples:
            if existing.validation_status != ValidationStatus.ACCEPTED:
                continue
            ex_emb = self._encode(self._triple_to_text(existing))
            if ex_emb is None:
                continue
            sim = self._cosine_sim(new_emb, ex_emb)
            max_sim = max(max_sim, sim)
            if sim >= self.config.similarity_threshold:
                is_duplicate = True
                break

        processing_time_ms = (time.time() - start_time) * 1000

        if is_duplicate:
            accepted = False
            confidence = 1.0 - max_sim
            self.stats["duplicates_found"] += 1
        else:
            accepted = True
            confidence = max(self.config.acceptance_threshold, 1.0 - max_sim)

        self.stats["total"] += 1
        if accepted:
            self.stats["accepted"] += 1
            triple.validation_status = ValidationStatus.ACCEPTED
        else:
            self.stats["rejected"] += 1
            triple.validation_status = ValidationStatus.REJECTED

        return BaselineResult(
            accepted=accepted,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            details={
                "baseline": "pure_embedding",
                "is_duplicate": is_duplicate,
                "max_similarity": max_sim,
            }
        )

    def validate_batch(self, triples: List[Triple], graph_repo: Any = None) -> List[BaselineResult]:
        return [self.validate(t, graph_repo) for t in triples]

    def get_statistics(self) -> Dict[str, Any]:
        total = self.stats["total"]
        return {
            "baseline_type": "pure_embedding",
            "total_processed": total,
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "acceptance_rate": self.stats["accepted"] / total if total > 0 else 0,
            "avg_time_ms": self.stats["total_time_ms"] / total if total > 0 else 0,
            "duplicates_found": self.stats["duplicates_found"],
        }

    def reset(self):
        self.stats = {
            "total": 0, "accepted": 0, "rejected": 0,
            "total_time_ms": 0.0, "duplicates_found": 0,
        }
