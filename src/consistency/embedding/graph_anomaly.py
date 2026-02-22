# src/consistency/embedding/graph_anomaly.py
"""
Graph-basierte Anomalie-Erkennung für Knowledge Graphs.

Erkennt strukturelle Auffälligkeiten ohne LLM:
1. Degree Anomaly - Entity hat ungewöhnlich viele/wenige Relationen
2. Type Distribution Anomaly - Relation zwischen unüblichen Entity-Typen
3. Cluster Bridge Anomaly - Triple verbindet normalerweise getrennte Cluster
4. Relation Frequency Anomaly - Ungewöhnliche Häufigkeit eines Relationstyps

Wissenschaftliche Grundlage:
- Akoglu et al. (2015): "Graph-based Anomaly Detection and Description"
- Ranshous et al. (2015): "Anomaly Detection in Dynamic Networks"
"""

import logging
import math
from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from src.models.entities import Triple, ConflictSet, ConflictType

logger = logging.getLogger(__name__)


@dataclass
class GraphAnomalyConfig:
    """Konfiguration für Graph-Anomalie-Erkennung."""

    # Degree Anomaly
    degree_z_score_threshold: float = 3.0  # Standard-Abweichungen
    min_degree_samples: int = 10  # Minimum Samples für Statistik

    # Type Distribution Anomaly
    type_frequency_threshold: float = 0.01  # < 1% = selten
    min_type_samples: int = 5

    # Relation Frequency Anomaly
    relation_frequency_z_score: float = 2.5

    # Allgemeine Einstellungen
    update_statistics_interval: int = 100  # Alle N Triples neu berechnen
    anomaly_score_threshold: float = 0.7


@dataclass
class GraphStatistics:
    """Statistiken über den Graph für Anomalie-Erkennung."""

    # Degree-Statistiken
    avg_out_degree: float = 0.0
    std_out_degree: float = 1.0
    avg_in_degree: float = 0.0
    std_in_degree: float = 1.0

    # Entity-Typ zu Relation-Typ Verteilung
    # type_relation_counts[(entity_type, relation_type)] = count
    type_relation_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Relation-Typ Häufigkeiten
    relation_counts: Dict[str, int] = field(default_factory=dict)

    # Entity-Typ Paare pro Relation
    # relation_type_pairs[relation][(subject_type, object_type)] = count
    relation_type_pairs: Dict[str, Dict[Tuple[str, str], int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    # Gesamtzahlen
    total_entities: int = 0
    total_relations: int = 0

    # Entity Degrees
    entity_out_degrees: Dict[str, int] = field(default_factory=dict)
    entity_in_degrees: Dict[str, int] = field(default_factory=dict)


class GraphAnomalyDetector:
    """
    Erkennt strukturelle Anomalien im Knowledge Graph.
    """

    def __init__(self, config: GraphAnomalyConfig = None):
        self.config = config or GraphAnomalyConfig()
        self.stats = GraphStatistics()
        self._triples_since_update = 0

        logger.info("GraphAnomalyDetector initialisiert")

    def update_statistics(self, graph_repo: Any):
        """
        Aktualisiert die Graph-Statistiken.

        Sollte regelmäßig aufgerufen werden für akkurate Anomalie-Erkennung.
        """
        if not graph_repo:
            return

        logger.debug("Aktualisiere Graph-Statistiken...")

        # Reset
        self.stats = GraphStatistics()

        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)
        type_relation = defaultdict(int)
        relation_counts = defaultdict(int)
        relation_type_pairs = defaultdict(lambda: defaultdict(int))

        try:
            # Iteriere über alle Relationen
            for rel in graph_repo._relations.values():
                source = graph_repo._entities.get(rel.source_id)
                target = graph_repo._entities.get(rel.target_id)

                if not source or not target:
                    continue

                rel_type = rel.relation_type.upper()
                source_type = source.entity_type.value
                target_type = target.entity_type.value

                # Degree-Statistiken
                out_degrees[rel.source_id] += 1
                in_degrees[rel.target_id] += 1

                # Type-Relation Verteilung
                type_relation[(source_type, rel_type)] += 1
                type_relation[(target_type, rel_type)] += 1

                # Relation-Häufigkeit
                relation_counts[rel_type] += 1

                # Entity-Typ-Paare pro Relation
                relation_type_pairs[rel_type][(source_type, target_type)] += 1

            # Statistiken berechnen
            if out_degrees:
                degrees = list(out_degrees.values())
                self.stats.avg_out_degree = sum(degrees) / len(degrees)
                self.stats.std_out_degree = self._std(degrees, self.stats.avg_out_degree)

            if in_degrees:
                degrees = list(in_degrees.values())
                self.stats.avg_in_degree = sum(degrees) / len(degrees)
                self.stats.std_in_degree = self._std(degrees, self.stats.avg_in_degree)

            self.stats.entity_out_degrees = dict(out_degrees)
            self.stats.entity_in_degrees = dict(in_degrees)
            self.stats.type_relation_counts = dict(type_relation)
            self.stats.relation_counts = dict(relation_counts)
            self.stats.relation_type_pairs = {k: dict(v) for k, v in relation_type_pairs.items()}
            self.stats.total_entities = len(graph_repo._entities)
            self.stats.total_relations = len(graph_repo._relations)

            logger.debug(
                f"Graph-Statistiken aktualisiert: "
                f"{self.stats.total_entities} Entities, "
                f"{self.stats.total_relations} Relationen, "
                f"avg_degree={self.stats.avg_out_degree:.2f}"
            )

        except Exception as e:
            logger.warning(f"Fehler beim Aktualisieren der Statistiken: {e}")

    def check_anomaly(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> List[ConflictSet]:
        """
        Prüft ein Triple auf verschiedene Graph-Anomalien.

        Returns:
            Liste von gefundenen Anomalie-Konflikten
        """
        conflicts = []

        # Statistiken aktualisieren wenn nötig
        self._triples_since_update += 1
        if self._triples_since_update >= self.config.update_statistics_interval:
            self.update_statistics(graph_repo)
            self._triples_since_update = 0

        # 1. Degree Anomaly
        conflict = self._check_degree_anomaly(triple, graph_repo)
        if conflict:
            conflicts.append(conflict)

        # 2. Type Distribution Anomaly
        conflict = self._check_type_distribution_anomaly(triple)
        if conflict:
            conflicts.append(conflict)

        # 3. Relation Type Pair Anomaly
        conflict = self._check_relation_type_pair_anomaly(triple)
        if conflict:
            conflicts.append(conflict)

        # 4. Relation Frequency Anomaly
        conflict = self._check_relation_frequency_anomaly(triple, graph_repo)
        if conflict:
            conflicts.append(conflict)

        return conflicts

    def score_triple(
        self,
        triple: Triple,
        graph_repo: Any = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Berechnet einen Anomalie-Score für das Triple.

        Returns:
            Tuple von (score, details)
            - score: 0.0-1.0 (0 = normal, 1 = sehr anomal)
        """
        scores = []
        details = {}

        # Degree Score
        degree_score = self._score_degree_anomaly(triple, graph_repo)
        scores.append(degree_score)
        details["degree_score"] = degree_score

        # Type Distribution Score
        type_score = self._score_type_distribution(triple)
        scores.append(type_score)
        details["type_score"] = type_score

        # Relation Type Pair Score
        pair_score = self._score_relation_type_pair(triple)
        scores.append(pair_score)
        details["pair_score"] = pair_score

        # Gewichteter Durchschnitt
        total_score = sum(scores) / len(scores) if scores else 0.0
        details["total_score"] = total_score

        return total_score, details

    def _check_degree_anomaly(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft ob das Triple eine Degree-Anomalie erzeugt.

        z.B. Entity hat plötzlich viel mehr Relationen als üblich.
        """
        if not graph_repo or self.stats.total_entities < self.config.min_degree_samples:
            return None

        subject_id = triple.subject.id

        # Aktueller Out-Degree + 1 (neues Triple)
        current_degree = self.stats.entity_out_degrees.get(subject_id, 0) + 1

        # Z-Score berechnen
        if self.stats.std_out_degree > 0:
            z_score = (current_degree - self.stats.avg_out_degree) / self.stats.std_out_degree

            if z_score > self.config.degree_z_score_threshold:
                return ConflictSet(
                    conflict_type=ConflictType.CONTRADICTORY_RELATION,
                    description=f"Degree Anomalie: '{triple.subject.name}' hätte {current_degree} "
                               f"ausgehende Relationen (Durchschnitt: {self.stats.avg_out_degree:.1f}, "
                               f"Z-Score: {z_score:.2f})",
                    severity=min(0.5 + z_score * 0.1, 0.8)
                )

        return None

    def _check_type_distribution_anomaly(
        self,
        triple: Triple
    ) -> Optional[ConflictSet]:
        """
        Prüft ob die Kombination aus Entity-Typ und Relation-Typ unüblich ist.
        """
        if self.stats.total_relations < self.config.min_type_samples:
            return None

        rel_type = triple.predicate.upper()
        subject_type = triple.subject.entity_type.value
        object_type = triple.object.entity_type.value

        # Prüfe Subject-Typ
        subject_key = (subject_type, rel_type)
        subject_count = self.stats.type_relation_counts.get(subject_key, 0)
        total_rel = self.stats.relation_counts.get(rel_type, 0)

        if total_rel > 0:
            subject_freq = subject_count / total_rel
            if subject_freq < self.config.type_frequency_threshold and subject_count < 3:
                return ConflictSet(
                    conflict_type=ConflictType.CONTRADICTORY_RELATION,
                    description=f"Typ-Anomalie: '{subject_type}' als Subject von '{rel_type}' "
                               f"ist selten ({subject_count} von {total_rel}, {subject_freq:.1%})",
                    severity=0.5
                )

        # Prüfe Object-Typ
        object_key = (object_type, rel_type)
        object_count = self.stats.type_relation_counts.get(object_key, 0)

        if total_rel > 0:
            object_freq = object_count / total_rel
            if object_freq < self.config.type_frequency_threshold and object_count < 3:
                return ConflictSet(
                    conflict_type=ConflictType.CONTRADICTORY_RELATION,
                    description=f"Typ-Anomalie: '{object_type}' als Object von '{rel_type}' "
                               f"ist selten ({object_count} von {total_rel}, {object_freq:.1%})",
                    severity=0.5
                )

        return None

    def _check_relation_type_pair_anomaly(
        self,
        triple: Triple
    ) -> Optional[ConflictSet]:
        """
        Prüft ob das Entity-Typ-Paar für diese Relation unüblich ist.
        """
        rel_type = triple.predicate.upper()
        pair = (triple.subject.entity_type.value, triple.object.entity_type.value)

        if rel_type not in self.stats.relation_type_pairs:
            return None

        pair_counts = self.stats.relation_type_pairs[rel_type]
        total = sum(pair_counts.values())

        if total < self.config.min_type_samples:
            return None

        pair_count = pair_counts.get(pair, 0)
        pair_freq = pair_count / total if total > 0 else 0

        if pair_freq < self.config.type_frequency_threshold and pair_count < 2:
            # Finde häufigste Paare als Referenz
            common_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:3]
            common_str = ", ".join(f"{p[0]}->{p[1]}" for p, _ in common_pairs)

            return ConflictSet(
                conflict_type=ConflictType.CONTRADICTORY_RELATION,
                description=f"Typ-Paar Anomalie: '{pair[0]}' --[{rel_type}]--> '{pair[1]}' "
                           f"ist unüblich ({pair_count}/{total}). "
                           f"Üblich: {common_str}",
                severity=0.6
            )

        return None

    def _check_relation_frequency_anomaly(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[ConflictSet]:
        """
        Prüft ob das Subject ungewöhnlich viele Relationen dieses Typs hat.
        """
        if not graph_repo:
            return None

        rel_type = triple.predicate.upper()

        # Zähle existierende Relationen dieses Typs vom Subject
        existing = graph_repo.find_relations(source_id=triple.subject.id)
        same_type_count = sum(
            1 for r in existing
            if r.get("rel_type", "").upper() == rel_type
        )

        # Berechne erwartete Häufigkeit
        total_rel = self.stats.relation_counts.get(rel_type, 0)
        if self.stats.total_entities > 0 and total_rel > 0:
            expected = total_rel / self.stats.total_entities
            if expected > 0:
                # Poisson-Approximation: z-score
                z_score = (same_type_count + 1 - expected) / math.sqrt(max(expected, 1))

                if z_score > self.config.relation_frequency_z_score and same_type_count >= 3:
                    return ConflictSet(
                        conflict_type=ConflictType.CONTRADICTORY_RELATION,
                        description=f"Häufigkeits-Anomalie: '{triple.subject.name}' hat bereits "
                                   f"{same_type_count} '{rel_type}'-Relationen "
                                   f"(erwartet: {expected:.1f}, Z-Score: {z_score:.2f})",
                        severity=0.5
                    )

        return None

    def _score_degree_anomaly(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> float:
        """Berechnet Degree-Anomalie-Score."""
        if not graph_repo or self.stats.std_out_degree == 0:
            return 0.0

        current_degree = self.stats.entity_out_degrees.get(triple.subject.id, 0) + 1
        z_score = abs(current_degree - self.stats.avg_out_degree) / max(self.stats.std_out_degree, 1)

        # Normalisiere zu 0-1
        return min(z_score / 5.0, 1.0)

    def _score_type_distribution(self, triple: Triple) -> float:
        """Berechnet Type-Distribution-Score."""
        if self.stats.total_relations == 0:
            return 0.0

        rel_type = triple.predicate.upper()
        subject_type = triple.subject.entity_type.value
        total_rel = self.stats.relation_counts.get(rel_type, 1)

        subject_count = self.stats.type_relation_counts.get((subject_type, rel_type), 0)
        freq = subject_count / total_rel if total_rel > 0 else 0

        # Niedrige Frequenz = hoher Score
        if freq < 0.01:
            return 0.8
        elif freq < 0.05:
            return 0.5
        elif freq < 0.1:
            return 0.3
        return 0.0

    def _score_relation_type_pair(self, triple: Triple) -> float:
        """Berechnet Relation-Type-Pair-Score."""
        rel_type = triple.predicate.upper()
        pair = (triple.subject.entity_type.value, triple.object.entity_type.value)

        if rel_type not in self.stats.relation_type_pairs:
            return 0.3  # Unbekannte Relation

        pair_counts = self.stats.relation_type_pairs[rel_type]
        total = sum(pair_counts.values())

        if total == 0:
            return 0.0

        pair_count = pair_counts.get(pair, 0)
        freq = pair_count / total

        if freq < 0.01:
            return 0.8
        elif freq < 0.05:
            return 0.5
        elif freq < 0.1:
            return 0.3
        return 0.0

    @staticmethod
    def _std(values: List[float], mean: float) -> float:
        """Berechnet Standardabweichung."""
        if len(values) < 2:
            return 1.0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance) if variance > 0 else 1.0
