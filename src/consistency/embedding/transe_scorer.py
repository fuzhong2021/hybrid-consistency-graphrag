# src/consistency/embedding/transe_scorer.py
"""
TransE-basierte Plausibilitätsbewertung für Knowledge Graph Triples.

TransE (Translating Embeddings) Prinzip:
- Jede Entity und Relation wird als Vektor im Embedding-Raum repräsentiert
- Für ein korrektes Triple (h, r, t) gilt: h + r ≈ t
- Score: ||h + r - t|| (niedrig = plausibel, hoch = unplausibel)

Wissenschaftliche Grundlage:
- Bordes et al. (2013): "Translating Embeddings for Modeling Multi-relational Data"
- Sun et al. (2019): "RotatE: Knowledge Graph Embedding by Relational Rotation"

Anwendung für Konsistenzprüfung:
- Niedriger TransE-Score → Triple ist plausibel
- Hoher TransE-Score → Triple könnte falsch sein (Anomalie)
"""

import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from src.models.entities import Triple, ConflictSet, ConflictType

logger = logging.getLogger(__name__)


@dataclass
class TransEConfig:
    """Konfiguration für TransE Scoring."""

    # Embedding-Dimension (wird automatisch erkannt wenn Modell vorhanden)
    embedding_dim: int = 384

    # Anomalie-Schwellenwerte
    anomaly_threshold: float = 0.7  # Score > threshold = Anomalie
    warning_threshold: float = 0.5  # Score > warning = Warnung

    # Normalisierung
    normalize_embeddings: bool = True

    # Cache-Größe für Relation-Embeddings
    max_cache_size: int = 10000

    # Fallback wenn keine Embeddings verfügbar
    use_name_similarity_fallback: bool = True


class TransEScorer:
    """
    Bewertet Triple-Plausibilität mit TransE-Prinzip.

    Nutzt vorhandene Entity-Embeddings und lernt Relation-Translationen.
    """

    def __init__(
        self,
        embedding_model: Any = None,
        config: TransEConfig = None
    ):
        """
        Args:
            embedding_model: Embedding-Modell mit embed_query() Methode
            config: TransE-Konfiguration
        """
        self.embedding_model = embedding_model
        self.config = config or TransEConfig()

        # Caches
        self._entity_embeddings: Dict[str, np.ndarray] = {}
        self._relation_embeddings: Dict[str, np.ndarray] = {}

        # Statistiken für Relation-Embeddings
        self._relation_samples: Dict[str, List[np.ndarray]] = defaultdict(list)

        logger.info(f"TransEScorer initialisiert (dim={self.config.embedding_dim})")

    def score_triple(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Berechnet TransE-Score für ein Triple.

        Returns:
            Tuple von (score, details)
            - score: 0.0-1.0 (0 = perfekt plausibel, 1 = sehr anomal)
            - details: Dict mit zusätzlichen Informationen
        """
        if not self.embedding_model:
            return 0.5, {"error": "no_embedding_model", "method": "fallback"}

        try:
            # Entity-Embeddings holen
            head_emb = self._get_entity_embedding(triple.subject.name)
            tail_emb = self._get_entity_embedding(triple.object.name)

            if head_emb is None or tail_emb is None:
                return 0.5, {"error": "embedding_failed", "method": "fallback"}

            # Relation-Embedding holen oder lernen
            rel_emb = self._get_relation_embedding(
                triple.predicate,
                head_emb,
                tail_emb,
                graph_repo
            )

            # TransE Score berechnen: ||h + r - t||
            if rel_emb is not None:
                translation = head_emb + rel_emb
                distance = np.linalg.norm(translation - tail_emb)

                # Normalisiere zu 0-1 (heuristisch)
                # Typische Distanzen liegen zwischen 0.5 und 2.0
                score = min(1.0, distance / 2.0)

                return score, {
                    "method": "transe",
                    "distance": float(distance),
                    "head_norm": float(np.linalg.norm(head_emb)),
                    "tail_norm": float(np.linalg.norm(tail_emb)),
                    "relation": triple.predicate,
                }
            else:
                # Fallback: Cosine Similarity zwischen Head und Tail
                similarity = self._cosine_similarity(head_emb, tail_emb)
                # Konvertiere Similarity zu "Anomalie-Score"
                # Sehr hohe Similarity (≈1) = möglicherweise Duplikat
                # Sehr niedrige Similarity (≈0) = ungewöhnliche Kombination
                score = 1.0 - similarity if similarity > 0.5 else similarity

                return score, {
                    "method": "cosine_fallback",
                    "similarity": float(similarity),
                }

        except Exception as e:
            logger.warning(f"TransE Scoring fehlgeschlagen: {e}")
            return 0.5, {"error": str(e), "method": "error_fallback"}

    def check_anomaly(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> Optional[ConflictSet]:
        """
        Prüft ob ein Triple eine TransE-Anomalie ist.

        Returns:
            ConflictSet wenn Anomalie erkannt, sonst None
        """
        score, details = self.score_triple(triple, graph_repo)

        if score >= self.config.anomaly_threshold:
            return ConflictSet(
                conflict_type=ConflictType.CONTRADICTORY_RELATION,
                description=f"TransE Anomalie: Triple '{triple.subject.name}' --[{triple.predicate}]--> "
                           f"'{triple.object.name}' hat hohen Anomalie-Score ({score:.2f}). "
                           f"Methode: {details.get('method', 'unknown')}",
                severity=0.6 + (score - self.config.anomaly_threshold) * 0.4
            )

        if score >= self.config.warning_threshold:
            return ConflictSet(
                conflict_type=ConflictType.CONTRADICTORY_RELATION,
                description=f"TransE Warnung: Triple hat erhöhten Anomalie-Score ({score:.2f})",
                severity=0.4
            )

        return None

    def learn_relation(
        self,
        predicate: str,
        examples: List[Tuple[str, str]],
    ):
        """
        Lernt Relation-Embedding aus Beispielen.

        Args:
            predicate: Relationstyp
            examples: Liste von (head_name, tail_name) Paaren
        """
        if not self.embedding_model:
            return

        translations = []

        for head_name, tail_name in examples:
            head_emb = self._get_entity_embedding(head_name)
            tail_emb = self._get_entity_embedding(tail_name)

            if head_emb is not None and tail_emb is not None:
                # r = t - h (TransE Prinzip)
                translation = tail_emb - head_emb
                translations.append(translation)

        if translations:
            # Durchschnitt als Relation-Embedding
            rel_emb = np.mean(translations, axis=0)
            if self.config.normalize_embeddings:
                norm = np.linalg.norm(rel_emb)
                if norm > 0:
                    rel_emb = rel_emb / norm

            self._relation_embeddings[predicate.upper()] = rel_emb
            logger.debug(f"Relation-Embedding gelernt: {predicate} (aus {len(translations)} Beispielen)")

    def learn_from_graph(self, graph_repo: Any):
        """
        Lernt Relation-Embeddings aus dem gesamten Graph.
        """
        if not graph_repo or not self.embedding_model:
            return

        logger.info("Lerne Relation-Embeddings aus Graph...")

        # Sammle Beispiele pro Relation
        relation_examples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        try:
            # Iteriere über alle Relationen im Graph
            for rel in graph_repo._relations.values():
                rel_type = rel.relation_type.upper()
                source = graph_repo._entities.get(rel.source_id)
                target = graph_repo._entities.get(rel.target_id)

                if source and target:
                    relation_examples[rel_type].append((source.name, target.name))

            # Lerne Embeddings
            for rel_type, examples in relation_examples.items():
                if len(examples) >= 3:  # Mindestens 3 Beispiele
                    self.learn_relation(rel_type, examples)

            logger.info(f"Relation-Embeddings gelernt für {len(self._relation_embeddings)} Relationstypen")

        except Exception as e:
            logger.warning(f"Fehler beim Lernen aus Graph: {e}")

    def _get_entity_embedding(self, name: str) -> Optional[np.ndarray]:
        """Holt oder berechnet Entity-Embedding."""
        if name in self._entity_embeddings:
            return self._entity_embeddings[name]

        if not self.embedding_model:
            return None

        try:
            emb = np.array(self.embedding_model.embed_query(name))
            if self.config.normalize_embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

            # Cache (mit Größenbeschränkung)
            if len(self._entity_embeddings) < self.config.max_cache_size:
                self._entity_embeddings[name] = emb

            return emb

        except Exception as e:
            logger.debug(f"Entity-Embedding fehlgeschlagen für '{name}': {e}")
            return None

    def _get_relation_embedding(
        self,
        predicate: str,
        head_emb: np.ndarray,
        tail_emb: np.ndarray,
        graph_repo: Any = None
    ) -> Optional[np.ndarray]:
        """Holt oder lernt Relation-Embedding."""
        predicate_upper = predicate.upper()

        # Bereits gelernt?
        if predicate_upper in self._relation_embeddings:
            return self._relation_embeddings[predicate_upper]

        # Versuche aus Graph zu lernen
        if graph_repo:
            examples = self._collect_relation_examples(predicate_upper, graph_repo)
            if len(examples) >= 2:
                self.learn_relation(predicate_upper, examples)
                if predicate_upper in self._relation_embeddings:
                    return self._relation_embeddings[predicate_upper]

        # Fallback: Verwende den Text des Prädikats als Embedding
        if self.embedding_model:
            try:
                label = predicate.replace("_", " ").lower()
                rel_emb = np.array(self.embedding_model.embed_query(label))
                if self.config.normalize_embeddings:
                    norm = np.linalg.norm(rel_emb)
                    if norm > 0:
                        rel_emb = rel_emb / norm
                return rel_emb
            except Exception:
                pass

        return None

    def _collect_relation_examples(
        self,
        predicate: str,
        graph_repo: Any
    ) -> List[Tuple[str, str]]:
        """Sammelt Beispiele für eine Relation aus dem Graph."""
        examples = []

        try:
            for rel in graph_repo._relations.values():
                if rel.relation_type.upper() == predicate:
                    source = graph_repo._entities.get(rel.source_id)
                    target = graph_repo._entities.get(rel.target_id)
                    if source and target:
                        examples.append((source.name, target.name))
                        if len(examples) >= 50:  # Max 50 Beispiele
                            break
        except Exception:
            pass

        return examples

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Berechnet Cosine Similarity."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über gelernte Embeddings zurück."""
        return {
            "entity_embeddings_cached": len(self._entity_embeddings),
            "relation_embeddings_learned": len(self._relation_embeddings),
            "relations": list(self._relation_embeddings.keys()),
        }
