#!/usr/bin/env python3
# evaluation/streaming/shuffle_strategy.py
"""
Streaming Shuffle Strategien für realistische Evaluation.

Simuliert verschiedene Streaming-Szenarien:
1. Random Shuffle: Zufällige Reihenfolge (seed=42 für Reproduzierbarkeit)
2. Interleaved: Abwechselnd SUPPORTS/REFUTES
3. Temporal: Basierend auf simulierten Zeitstempeln
4. Clustered: Ähnliche Triples zusammen (simuliert Burst-Traffic)

Wissenschaftliche Referenz:
- Heist & Paulheim (2019): Streaming KG Construction
"""

import random
import logging
from typing import List, Iterator, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .triple_generator import AnnotatedTriple, TripleCategory

logger = logging.getLogger(__name__)


class ShuffleStrategy(Enum):
    """Verfügbare Shuffle-Strategien."""
    RANDOM = "random"           # Komplett zufällig
    INTERLEAVED = "interleaved" # Abwechselnd positive/negative
    TEMPORAL = "temporal"       # Nach simuliertem Zeitstempel
    CLUSTERED = "clustered"     # Nach Entity-Cluster
    ADVERSARIAL = "adversarial" # Konflikte direkt nach Original


@dataclass
class StreamingShuffler:
    """
    Implementiert verschiedene Shuffle-Strategien für Streaming-Simulation.

    Attribute:
        strategy: Die verwendete Shuffle-Strategie
        seed: Random seed für Reproduzierbarkeit
        batch_size: Größe der Streaming-Batches
    """
    strategy: ShuffleStrategy = ShuffleStrategy.RANDOM
    seed: int = 42
    batch_size: int = 10

    def __post_init__(self):
        random.seed(self.seed)

    def shuffle(
        self,
        triples: List[AnnotatedTriple]
    ) -> List[AnnotatedTriple]:
        """
        Shuffelt Triples nach der gewählten Strategie.

        Args:
            triples: Liste von AnnotatedTriples

        Returns:
            Gemischte Liste
        """
        if self.strategy == ShuffleStrategy.RANDOM:
            return self._random_shuffle(triples)
        elif self.strategy == ShuffleStrategy.INTERLEAVED:
            return self._interleaved_shuffle(triples)
        elif self.strategy == ShuffleStrategy.TEMPORAL:
            return self._temporal_shuffle(triples)
        elif self.strategy == ShuffleStrategy.CLUSTERED:
            return self._clustered_shuffle(triples)
        elif self.strategy == ShuffleStrategy.ADVERSARIAL:
            return self._adversarial_shuffle(triples)
        else:
            return self._random_shuffle(triples)

    def _random_shuffle(
        self,
        triples: List[AnnotatedTriple]
    ) -> List[AnnotatedTriple]:
        """
        Komplett zufälliges Mischen (mit seed für Reproduzierbarkeit).

        Dies ist der Default für wissenschaftliche Evaluation.
        """
        result = triples.copy()
        random.shuffle(result)
        logger.info(f"Random Shuffle: {len(result)} Triples (seed={self.seed})")
        return result

    def _interleaved_shuffle(
        self,
        triples: List[AnnotatedTriple]
    ) -> List[AnnotatedTriple]:
        """
        Abwechselnd positive und negative Triples.

        Strategie:
        - Zuerst ein SUPPORTS Triple
        - Dann ein REFUTES Triple
        - Dann ein ENTITY_VARIANT oder CROSS_DOC
        - Wiederholen

        Das testet die Fähigkeit bei schnellem Kontextwechsel.
        """
        # Kategorisiere
        supports = [t for t in triples if t.category == TripleCategory.SUPPORTS]
        refutes = [t for t in triples if t.category == TripleCategory.REFUTES]
        variants = [t for t in triples if t.category == TripleCategory.ENTITY_VARIANT]
        cross_doc = [t for t in triples if t.category == TripleCategory.CROSS_DOC]

        # Shuffle innerhalb der Kategorien
        random.shuffle(supports)
        random.shuffle(refutes)
        random.shuffle(variants)
        random.shuffle(cross_doc)

        # Interleave
        result = []
        iters = [iter(supports), iter(refutes), iter(variants), iter(cross_doc)]

        # Round-robin
        while any(True for _ in []):  # Dummy, replaced below
            added = False
            for it in iters:
                try:
                    result.append(next(it))
                    added = True
                except StopIteration:
                    continue
            if not added:
                break

        # Tatsächliche Round-Robin Implementierung
        result = []
        queues = [list(supports), list(refutes), list(variants), list(cross_doc)]
        indices = [0, 0, 0, 0]

        while True:
            added = False
            for i, queue in enumerate(queues):
                if indices[i] < len(queue):
                    result.append(queue[indices[i]])
                    indices[i] += 1
                    added = True
            if not added:
                break

        logger.info(f"Interleaved Shuffle: {len(result)} Triples")
        return result

    def _temporal_shuffle(
        self,
        triples: List[AnnotatedTriple]
    ) -> List[AnnotatedTriple]:
        """
        Sortiert nach simulierten Zeitstempeln.

        Annahme: Fakten werden über Zeit extrahiert.
        - Unterstützende Fakten kommen zuerst (etablieren Baseline)
        - Widersprüche kommen später (als Updates)
        - Entity-Varianten sind verteilt (unterschiedliche Quellen)
        """
        # Weise jedem Triple einen simulierten Zeitstempel zu
        timestamped = []

        for t in triples:
            if t.category == TripleCategory.SUPPORTS:
                # Frühe Zeitstempel für SUPPORTS
                timestamp = random.uniform(0.0, 0.4)
            elif t.category == TripleCategory.ENTITY_VARIANT:
                # Mittlere Zeitstempel für Varianten
                timestamp = random.uniform(0.2, 0.7)
            elif t.category == TripleCategory.REFUTES:
                # Späte Zeitstempel für REFUTES
                timestamp = random.uniform(0.5, 0.9)
            else:  # CROSS_DOC
                # Sehr späte Zeitstempel für Cross-Doc
                timestamp = random.uniform(0.7, 1.0)

            timestamped.append((timestamp, t))

        # Sortiere nach Zeitstempel
        timestamped.sort(key=lambda x: x[0])
        result = [t for _, t in timestamped]

        logger.info(f"Temporal Shuffle: {len(result)} Triples")
        return result

    def _clustered_shuffle(
        self,
        triples: List[AnnotatedTriple]
    ) -> List[AnnotatedTriple]:
        """
        Gruppiert Triples nach Subject-Entity.

        Simuliert Burst-Traffic wo viele Triples zur selben Entity kommen.
        Das testet die Entity-Resolution und Konflikt-Erkennung.
        """
        # Gruppiere nach Subject
        clusters: Dict[str, List[AnnotatedTriple]] = {}
        for t in triples:
            key = t.triple.subject.name.lower()
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(t)

        # Shuffle Cluster-Reihenfolge
        cluster_keys = list(clusters.keys())
        random.shuffle(cluster_keys)

        # Baue Ergebnis
        result = []
        for key in cluster_keys:
            cluster = clusters[key]
            random.shuffle(cluster)  # Shuffle innerhalb des Clusters
            result.extend(cluster)

        logger.info(f"Clustered Shuffle: {len(result)} Triples in {len(clusters)} Clustern")
        return result

    def _adversarial_shuffle(
        self,
        triples: List[AnnotatedTriple]
    ) -> List[AnnotatedTriple]:
        """
        Adversarial Shuffle: Konflikte direkt nach Original.

        Strategie:
        - Platziere REFUTES Triple direkt nach dem zugehörigen SUPPORTS
        - Das testet sofortige Konflikt-Erkennung

        WICHTIG: Nur möglich wenn Konflikte auf SUPPORTS basieren.
        """
        # Erstelle Mapping von claim_id zu SUPPORTS Triple
        supports_by_id: Dict[str, AnnotatedTriple] = {}
        other_triples = []

        for t in triples:
            if t.category == TripleCategory.SUPPORTS:
                supports_by_id[t.original_claim_id] = t
            else:
                other_triples.append(t)

        # Paare SUPPORTS mit zugehörigen Konflikten
        pairs = []
        remaining_conflicts = []

        for t in other_triples:
            if t.original_claim_id in supports_by_id:
                # Paare mit SUPPORTS
                pairs.append((supports_by_id[t.original_claim_id], t))
                # Entferne aus dict um doppelte Nutzung zu vermeiden
                del supports_by_id[t.original_claim_id]
            else:
                remaining_conflicts.append(t)

        # Baue Ergebnis: Erst Paare, dann verbleibende SUPPORTS, dann Rest
        result = []

        # Shuffle Paare
        random.shuffle(pairs)
        for support, conflict in pairs:
            result.append(support)
            result.append(conflict)

        # Verbleibende SUPPORTS
        remaining_supports = list(supports_by_id.values())
        random.shuffle(remaining_supports)
        result.extend(remaining_supports)

        # Verbleibende Konflikte
        random.shuffle(remaining_conflicts)
        result.extend(remaining_conflicts)

        logger.info(f"Adversarial Shuffle: {len(result)} Triples, {len(pairs)} Paare")
        return result

    def stream(
        self,
        triples: List[AnnotatedTriple],
        shuffle_first: bool = True
    ) -> Iterator[AnnotatedTriple]:
        """
        Streamt Triples einzeln (Generator).

        Args:
            triples: Liste von Triples
            shuffle_first: Wenn True, wird vor dem Streaming geshuffelt

        Yields:
            Einzelne AnnotatedTriples
        """
        if shuffle_first:
            triples = self.shuffle(triples)

        for triple in triples:
            yield triple

    def stream_batches(
        self,
        triples: List[AnnotatedTriple],
        shuffle_first: bool = True
    ) -> Iterator[List[AnnotatedTriple]]:
        """
        Streamt Triples in Batches.

        Args:
            triples: Liste von Triples
            shuffle_first: Wenn True, wird vor dem Streaming geshuffelt

        Yields:
            Listen von AnnotatedTriples (Batches)
        """
        if shuffle_first:
            triples = self.shuffle(triples)

        for i in range(0, len(triples), self.batch_size):
            yield triples[i:i + self.batch_size]

    def get_statistics(
        self,
        triples: List[AnnotatedTriple]
    ) -> Dict[str, Any]:
        """Statistiken über die Triple-Verteilung."""
        if not triples:
            return {"total": 0}

        categories = {}
        for t in triples:
            cat = t.category.value
            categories[cat] = categories.get(cat, 0) + 1

        ground_truth = {
            "should_accept": sum(1 for t in triples if t.should_accept),
            "should_reject": sum(1 for t in triples if t.should_reject),
            "should_merge": sum(1 for t in triples if t.should_merge),
        }

        # Berechne Position-basierte Statistiken
        positions = {cat: [] for cat in TripleCategory}
        for i, t in enumerate(triples):
            positions[t.category].append(i)

        position_stats = {}
        for cat, pos_list in positions.items():
            if pos_list:
                position_stats[cat.value] = {
                    "first": min(pos_list),
                    "last": max(pos_list),
                    "mean": sum(pos_list) / len(pos_list),
                }

        return {
            "total": len(triples),
            "categories": categories,
            "ground_truth": ground_truth,
            "position_stats": position_stats,
            "strategy": self.strategy.value,
            "seed": self.seed,
        }


def create_shuffler(
    strategy: str = "random",
    seed: int = 42,
    batch_size: int = 10
) -> StreamingShuffler:
    """Factory-Funktion für StreamingShuffler."""
    try:
        strat = ShuffleStrategy(strategy.lower())
    except ValueError:
        logger.warning(f"Unbekannte Strategie '{strategy}', nutze 'random'")
        strat = ShuffleStrategy.RANDOM

    return StreamingShuffler(
        strategy=strat,
        seed=seed,
        batch_size=batch_size
    )
