#!/usr/bin/env python3
# evaluation/streaming/comprehensive_conflict_generator.py
"""
Umfassender Konflikt-Generator für alle 10 Taxonomie-Kategorien.

Generiert wissenschaftlich fundierte Testfälle für jeden Konflikt-Typ:
1. Faktische Widersprüche
2. Temporale Konflikte
3. Granularitäts-Differenzen
4. Entity-Varianten
5. Implizite Widersprüche
6. Negations-Konflikte
7. Modalitäts-Konflikte
8. Source-Qualitäts-Konflikte
9. Schema-Heterogenität
10. Numerische Präzision

Jeder Generator produziert:
- Annotierte Triples mit Ground Truth
- Metadata für Reproduzierbarkeit
- Difficulty-Labels für differenzierte Evaluation

Autor: Masterarbeit GraphRAG Konsistenzprüfung
"""

import re
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from src.models.entities import Entity, EntityType, Triple

from .conflict_taxonomy import (
    ConflictType,
    GroundTruthAction,
    DetectionMethod,
    get_category,
)
from .triple_generator import AnnotatedTriple, TripleCategory

logger = logging.getLogger(__name__)


# =============================================================================
# ERWEITERTE DATENKLASSEN
# =============================================================================

@dataclass
class ConflictAnnotation:
    """Erweiterte Annotation für wissenschaftlich vollständige Evaluation."""
    conflict_type: ConflictType
    ground_truth_action: GroundTruthAction
    detection_methods: List[DetectionMethod]
    difficulty: str  # easy, medium, hard
    requires_world_knowledge: bool
    requires_temporal_data: bool
    requires_ontology: bool
    original_triple_id: Optional[str] = None
    conflict_description: str = ""
    expected_confidence_delta: float = 0.0  # Erwartete Konfidenz-Änderung


@dataclass
class AnnotatedConflictTriple:
    """Triple mit vollständiger Konflikt-Annotation."""
    triple: Triple
    annotation: ConflictAnnotation
    paired_triple_id: Optional[str] = None  # ID des konfliktierenden Triples

    @property
    def conflict_type(self) -> ConflictType:
        return self.annotation.conflict_type

    @property
    def ground_truth_action(self) -> GroundTruthAction:
        return self.annotation.ground_truth_action

    @property
    def should_accept(self) -> bool:
        return self.annotation.ground_truth_action == GroundTruthAction.ACCEPT

    @property
    def should_reject(self) -> bool:
        return self.annotation.ground_truth_action == GroundTruthAction.REJECT

    @property
    def should_merge(self) -> bool:
        return self.annotation.ground_truth_action == GroundTruthAction.MERGE


# =============================================================================
# ABSTRACT BASE GENERATOR
# =============================================================================

class ConflictGenerator(ABC):
    """Abstract Base Class für Konflikt-Generatoren."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self._entity_cache: Dict[str, Entity] = {}

    @property
    @abstractmethod
    def conflict_type(self) -> ConflictType:
        """Der Konflikt-Typ den dieser Generator produziert."""
        pass

    @abstractmethod
    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Konflikte aus Basis-Triples."""
        pass

    def _create_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.CONCEPT,
        source_doc: Optional[str] = None
    ) -> Entity:
        """Erstellt oder holt eine Entity aus dem Cache."""
        cache_key = f"{name.lower().strip()}_{entity_type.value}"
        if cache_key not in self._entity_cache:
            self._entity_cache[cache_key] = Entity(
                name=name.strip(),
                entity_type=entity_type,
                source_document=source_doc
            )
        return self._entity_cache[cache_key]

    def _create_annotation(
        self,
        conflict_description: str = "",
        expected_confidence_delta: float = 0.0,
        original_id: Optional[str] = None
    ) -> ConflictAnnotation:
        """Erstellt eine Annotation für den Konflikt-Typ."""
        category = get_category(self.conflict_type)
        return ConflictAnnotation(
            conflict_type=self.conflict_type,
            ground_truth_action=category.ground_truth_action,
            detection_methods=category.detection_methods,
            difficulty=category.difficulty,
            requires_world_knowledge=category.requires_world_knowledge,
            requires_temporal_data=category.requires_temporal_data,
            requires_ontology=category.requires_ontology,
            original_triple_id=original_id,
            conflict_description=conflict_description,
            expected_confidence_delta=expected_confidence_delta,
        )


# =============================================================================
# 1. FAKTISCHE WIDERSPRÜCHE
# =============================================================================

class FactualConflictGenerator(ConflictGenerator):
    """
    Generator für faktische Widersprüche (Kategorie 1).

    Erzeugt direkte Widersprüche durch Austausch des Objekts.
    """

    FACTUAL_ALTERNATIVES = {
        # Orte
        "Ulm": ["Munich", "Berlin", "Vienna", "Hamburg"],
        "Germany": ["Austria", "France", "Switzerland"],
        "Berlin": ["Munich", "Hamburg", "Frankfurt"],
        "Paris": ["Lyon", "Marseille", "Nice"],
        "London": ["Manchester", "Liverpool", "Edinburgh"],
        "New York": ["Los Angeles", "Chicago", "Boston"],

        # Jahreszahlen (innerhalb 5 Jahre)
        "1879": ["1878", "1880", "1881"],
        "1955": ["1954", "1956", "1957"],
        "1960": ["1959", "1961", "1962"],
        "2000": ["1999", "2001", "2002"],

        # Berufe
        "physicist": ["chemist", "mathematician", "biologist"],
        "actor": ["director", "producer", "writer"],
        "politician": ["businessman", "lawyer", "diplomat"],

        # Nationalitäten
        "German": ["Austrian", "Swiss", "American"],
        "American": ["British", "Canadian", "Australian"],
        "British": ["American", "Irish", "Scottish"],
    }

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.FACTUAL

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert faktische Widersprüche."""
        conflicts = []
        selected = random.sample(base_triples, min(num_conflicts, len(base_triples)))

        for triple in selected:
            conflict = self._generate_factual_conflict(triple)
            if conflict:
                conflicts.append(conflict)

        logger.info(f"[FACTUAL] Generiert: {len(conflicts)} Konflikte")
        return conflicts

    def _generate_factual_conflict(
        self,
        triple: Triple
    ) -> Optional[AnnotatedConflictTriple]:
        """Generiert einen faktischen Widerspruch für ein Triple."""
        original_obj = triple.object.name

        # Suche Alternative
        alternative = None
        for key, alts in self.FACTUAL_ALTERNATIVES.items():
            if key.lower() in original_obj.lower():
                alternative = random.choice(alts)
                break

        if not alternative:
            # Generische Alternative
            alternative = f"not-{original_obj}"

        # Erstelle konfliktierendes Triple
        conflict_entity = self._create_entity(
            alternative,
            triple.object.entity_type,
            f"factual_conflict_{triple.source_document_id}"
        )

        conflict_triple = Triple(
            subject=triple.subject,
            predicate=triple.predicate,
            object=conflict_entity,
            source_text=f"[FACTUAL CONFLICT] {triple.source_text}".replace(
                original_obj, alternative
            ),
            source_document_id=f"conflict_{triple.source_document_id}",
            extraction_confidence=0.85,
        )

        annotation = self._create_annotation(
            conflict_description=f"Factual: {original_obj} vs {alternative}",
            original_id=triple.source_document_id,
        )

        return AnnotatedConflictTriple(
            triple=conflict_triple,
            annotation=annotation,
            paired_triple_id=triple.source_document_id,
        )


# =============================================================================
# 2. TEMPORALE KONFLIKTE
# =============================================================================

class TemporalConflictGenerator(ConflictGenerator):
    """
    Generator für temporale Konflikte (Kategorie 2).

    Erzeugt Aussagen die zu verschiedenen Zeitpunkten gültig sind.
    """

    TEMPORAL_PATTERNS = [
        # (Subjekt-Muster, Prädikat, Objekt1, Zeit1, Objekt2, Zeit2)
        ("Einstein", "workedAt", "University of Berlin", "[1914-1932]",
         "Institute for Advanced Study", "[1933-1955]"),
        ("Germany", "hasCapital", "Bonn", "[1949-1990]",
         "Berlin", "[1990-present]"),
        ("Apple", "CEO", "Steve Jobs", "[1997-2011]",
         "Tim Cook", "[2011-present]"),
    ]

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.TEMPORAL

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert temporale Konflikte."""
        conflicts = []

        # Generiere aus Patterns
        for pattern in self.TEMPORAL_PATTERNS[:num_conflicts]:
            conflict_pair = self._generate_temporal_pair(pattern)
            conflicts.extend(conflict_pair)

        # Generiere aus Basis-Triples
        remaining = num_conflicts - len(conflicts)
        if remaining > 0 and base_triples:
            selected = random.sample(base_triples, min(remaining, len(base_triples)))
            for triple in selected:
                conflict = self._generate_temporal_variant(triple)
                if conflict:
                    conflicts.append(conflict)

        logger.info(f"[TEMPORAL] Generiert: {len(conflicts)} Konflikte")
        return conflicts

    def _generate_temporal_pair(
        self,
        pattern: Tuple[str, str, str, str, str, str]
    ) -> List[AnnotatedConflictTriple]:
        """Generiert ein Paar temporaler Aussagen."""
        subj, pred, obj1, time1, obj2, time2 = pattern

        subject = self._create_entity(subj, EntityType.PERSON)
        object1 = self._create_entity(obj1, EntityType.ORGANIZATION)
        object2 = self._create_entity(obj2, EntityType.ORGANIZATION)

        triple1 = Triple(
            subject=subject,
            predicate=pred,
            object=object1,
            source_text=f"{subj} {pred} {obj1} {time1}",
            source_document_id=f"temporal_1_{subj}",
            extraction_confidence=0.9,
        )

        triple2 = Triple(
            subject=subject,
            predicate=pred,
            object=object2,
            source_text=f"{subj} {pred} {obj2} {time2}",
            source_document_id=f"temporal_2_{subj}",
            extraction_confidence=0.9,
        )

        # Beide sollten AKZEPTIERT werden (disjunkte Zeiträume)
        annotation = self._create_annotation(
            conflict_description=f"Temporal: {time1} vs {time2}",
        )
        # Override: ACCEPT für disjunkte Zeiträume
        annotation.ground_truth_action = GroundTruthAction.ACCEPT

        return [
            AnnotatedConflictTriple(triple=triple1, annotation=annotation),
            AnnotatedConflictTriple(triple=triple2, annotation=annotation),
        ]

    def _generate_temporal_variant(
        self,
        triple: Triple
    ) -> Optional[AnnotatedConflictTriple]:
        """Generiert eine temporale Variante eines Triples."""
        # Füge Zeitangabe hinzu
        time_suffixes = ["[2010-2015]", "[2015-2020]", "[2020-present]"]
        time_suffix = random.choice(time_suffixes)

        temporal_triple = Triple(
            subject=triple.subject,
            predicate=triple.predicate,
            object=triple.object,
            source_text=f"{triple.source_text} {time_suffix}",
            source_document_id=f"temporal_{triple.source_document_id}",
            extraction_confidence=triple.extraction_confidence,
        )

        annotation = self._create_annotation(
            conflict_description=f"Temporal variant: {time_suffix}",
            original_id=triple.source_document_id,
        )

        return AnnotatedConflictTriple(
            triple=temporal_triple,
            annotation=annotation,
        )


# =============================================================================
# 3. GRANULARITÄTS-DIFFERENZEN
# =============================================================================

class GranularityConflictGenerator(ConflictGenerator):
    """
    Generator für Granularitäts-Differenzen (Kategorie 3).

    Erzeugt Aussagen auf verschiedenen Abstraktionsebenen.
    """

    GRANULARITY_HIERARCHY = {
        # fine -> coarse
        "Ulm": ["Baden-Württemberg", "Germany", "Europe"],
        "Berlin": ["Germany", "Europe"],
        "Paris": ["Île-de-France", "France", "Europe"],
        "New York City": ["New York State", "United States", "North America"],
        "Tokyo": ["Kantō", "Japan", "Asia"],

        # Zeitgranularität
        "March 14, 1879": ["March 1879", "1879", "1870s", "19th century"],
        "2020-03-15": ["March 2020", "2020", "early 2020s", "21st century"],
    }

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.GRANULARITY

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Granularitäts-Differenzen."""
        conflicts = []
        selected = random.sample(base_triples, min(num_conflicts, len(base_triples)))

        for triple in selected:
            conflict = self._generate_granularity_variant(triple)
            if conflict:
                conflicts.append(conflict)

        logger.info(f"[GRANULARITY] Generiert: {len(conflicts)} Konflikte")
        return conflicts

    def _generate_granularity_variant(
        self,
        triple: Triple
    ) -> Optional[AnnotatedConflictTriple]:
        """Generiert eine gröbere Granularität."""
        original_obj = triple.object.name

        # Suche gröbere Granularität
        coarse_version = None
        for fine, coarse_list in self.GRANULARITY_HIERARCHY.items():
            if fine.lower() in original_obj.lower():
                coarse_version = random.choice(coarse_list)
                break

        if not coarse_version:
            return None

        coarse_entity = self._create_entity(
            coarse_version,
            triple.object.entity_type,
            f"granularity_{triple.source_document_id}"
        )

        granularity_triple = Triple(
            subject=triple.subject,
            predicate=triple.predicate,
            object=coarse_entity,
            source_text=f"[COARSE] {triple.source_text}".replace(original_obj, coarse_version),
            source_document_id=f"granularity_{triple.source_document_id}",
            extraction_confidence=0.80,  # Etwas niedriger weil weniger spezifisch
        )

        annotation = self._create_annotation(
            conflict_description=f"Granularity: {original_obj} (fine) vs {coarse_version} (coarse)",
            original_id=triple.source_document_id,
        )

        return AnnotatedConflictTriple(
            triple=granularity_triple,
            annotation=annotation,
            paired_triple_id=triple.source_document_id,
        )


# =============================================================================
# 4. ENTITY-VARIANTEN (bereits implementiert, hier erweitert)
# =============================================================================

class EntityVariantConflictGenerator(ConflictGenerator):
    """
    Generator für Entity-Varianten (Kategorie 4).

    Erweiterte Version mit mehr Varianten-Typen.
    """

    VARIANT_PATTERNS = {
        "person": [
            lambda n: f"Dr. {n}",
            lambda n: f"Prof. {n}",
            lambda n: n.split()[0][0] + ". " + n.split()[-1] if len(n.split()) > 1 else n,
            lambda n: n.split()[-1] + ", " + n.split()[0] if len(n.split()) > 1 else n,
            lambda n: n.split()[-1] if len(n.split()) > 1 else n,
        ],
        "organization": [
            lambda n: n.replace(" Inc.", "").replace(" Corp.", "").replace(" Ltd.", ""),
            lambda n: "The " + n if not n.startswith("The") else n[4:],
            lambda n: "".join(w[0] for w in n.split() if w[0].isupper()),  # Acronym
        ],
        "location": [
            lambda n: n.replace("City", "").strip(),
            lambda n: n.replace("United States", "U.S."),
            lambda n: n.replace("United Kingdom", "U.K."),
        ],
    }

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.ENTITY_VARIANT

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Entity-Varianten."""
        conflicts = []
        selected = random.sample(base_triples, min(num_conflicts, len(base_triples)))

        for triple in selected:
            variants = self._generate_variants(triple)
            conflicts.extend(variants)

        logger.info(f"[ENTITY_VARIANT] Generiert: {len(conflicts)} Varianten")
        return conflicts[:num_conflicts]  # Limit

    def _generate_variants(
        self,
        triple: Triple
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Varianten für Subject und Object."""
        variants = []

        # Subject-Variante
        subj_variant = self._create_variant(
            triple.subject.name,
            triple.subject.entity_type
        )
        if subj_variant and subj_variant != triple.subject.name:
            variant_entity = self._create_entity(
                subj_variant,
                triple.subject.entity_type,
                f"variant_{triple.source_document_id}"
            )
            variant_triple = Triple(
                subject=variant_entity,
                predicate=triple.predicate,
                object=triple.object,
                source_text=f"[VARIANT] {triple.source_text}".replace(
                    triple.subject.name, subj_variant
                ),
                source_document_id=f"variant_{triple.source_document_id}",
                extraction_confidence=0.85,
            )
            annotation = self._create_annotation(
                conflict_description=f"Entity variant: {triple.subject.name} -> {subj_variant}",
                original_id=triple.source_document_id,
            )
            variants.append(AnnotatedConflictTriple(
                triple=variant_triple,
                annotation=annotation,
                paired_triple_id=triple.source_document_id,
            ))

        return variants

    def _create_variant(
        self,
        name: str,
        entity_type: EntityType
    ) -> Optional[str]:
        """Erstellt eine Variante eines Entity-Namens."""
        type_key = entity_type.value.lower()
        if type_key not in self.VARIANT_PATTERNS:
            type_key = "person"  # Default

        patterns = self.VARIANT_PATTERNS.get(type_key, [])
        if not patterns:
            return None

        pattern = random.choice(patterns)
        try:
            return pattern(name)
        except Exception:
            return None


# =============================================================================
# 5. IMPLIZITE WIDERSPRÜCHE
# =============================================================================

class ImplicitConflictGenerator(ConflictGenerator):
    """
    Generator für implizite Widersprüche (Kategorie 5).

    Erzeugt Widersprüche die Weltwissen/Inferenz erfordern.
    """

    IMPLICIT_PATTERNS = [
        # (Aussage1, Aussage2, Erklärung)
        ("had no children", "was the father of", "Contradiction: no children vs has child"),
        ("was never married", "divorced from", "Contradiction: never married vs divorce"),
        ("is a vegetarian", "regularly eats meat", "Contradiction: vegetarian vs eats meat"),
        ("died in 1955", "published a paper in 1960", "Temporal: dead person cannot publish"),
        ("is blind", "watched the movie", "Physical: blind person cannot watch"),
        ("cannot swim", "swam across the channel", "Ability: cannot swim vs swam"),
    ]

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.IMPLICIT

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert implizite Widersprüche."""
        conflicts = []

        # Nutze Patterns
        for i, (stmt1, stmt2, explanation) in enumerate(self.IMPLICIT_PATTERNS[:num_conflicts]):
            conflict_pair = self._generate_implicit_pair(
                f"Person_{i}",
                stmt1, stmt2, explanation
            )
            conflicts.extend(conflict_pair)

        logger.info(f"[IMPLICIT] Generiert: {len(conflicts)} Konflikte")
        return conflicts

    def _generate_implicit_pair(
        self,
        subject_name: str,
        statement1: str,
        statement2: str,
        explanation: str
    ) -> List[AnnotatedConflictTriple]:
        """Generiert ein Paar implizit widersprüchlicher Aussagen."""
        subject = self._create_entity(subject_name, EntityType.PERSON)

        # Triple 1 (sollte akzeptiert werden)
        triple1 = Triple(
            subject=subject,
            predicate="STATES",
            object=self._create_entity(statement1, EntityType.CONCEPT),
            source_text=f"{subject_name} {statement1}",
            source_document_id=f"implicit_1_{subject_name}",
            extraction_confidence=0.85,
        )

        # Triple 2 (sollte rejected werden - impliziter Widerspruch)
        triple2 = Triple(
            subject=subject,
            predicate="STATES",
            object=self._create_entity(statement2, EntityType.CONCEPT),
            source_text=f"{subject_name} {statement2}",
            source_document_id=f"implicit_2_{subject_name}",
            extraction_confidence=0.85,
        )

        annotation1 = self._create_annotation(
            conflict_description=f"Implicit (base): {statement1}",
        )
        annotation1.ground_truth_action = GroundTruthAction.ACCEPT

        annotation2 = self._create_annotation(
            conflict_description=f"Implicit conflict: {explanation}",
        )
        annotation2.ground_truth_action = GroundTruthAction.REJECT

        return [
            AnnotatedConflictTriple(triple=triple1, annotation=annotation1),
            AnnotatedConflictTriple(
                triple=triple2,
                annotation=annotation2,
                paired_triple_id=triple1.source_document_id
            ),
        ]


# =============================================================================
# 6. NEGATIONS-KONFLIKTE
# =============================================================================

class NegationConflictGenerator(ConflictGenerator):
    """
    Generator für Negations-Konflikte (Kategorie 6).

    Erzeugt direkte Negationen von Aussagen.
    """

    NEGATION_PATTERNS = [
        ("won", "did not win", "never won"),
        ("is", "is not"),
        ("was", "was not", "was never"),
        ("has", "does not have", "has no"),
        ("supports", "does not support", "opposes"),
        ("achieved", "failed to achieve", "never achieved"),
    ]

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.NEGATION

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Negations-Konflikte."""
        conflicts = []
        selected = random.sample(base_triples, min(num_conflicts, len(base_triples)))

        for triple in selected:
            conflict = self._generate_negation(triple)
            if conflict:
                conflicts.append(conflict)

        logger.info(f"[NEGATION] Generiert: {len(conflicts)} Konflikte")
        return conflicts

    def _generate_negation(
        self,
        triple: Triple
    ) -> Optional[AnnotatedConflictTriple]:
        """Generiert eine Negation eines Triples."""
        source_text = triple.source_text

        # Suche passendes Negations-Pattern
        negated_text = None
        for pattern in self.NEGATION_PATTERNS:
            positive = pattern[0]
            negatives = pattern[1:]
            if f" {positive} " in source_text.lower():
                negative = random.choice(negatives)
                negated_text = re.sub(
                    f" {positive} ",
                    f" {negative} ",
                    source_text,
                    flags=re.IGNORECASE
                )
                break

        if not negated_text:
            # Fallback: Präfix "NOT"
            negated_text = f"[NOT] {source_text}"

        negation_triple = Triple(
            subject=triple.subject,
            predicate=f"NOT_{triple.predicate}",
            object=triple.object,
            source_text=negated_text,
            source_document_id=f"negation_{triple.source_document_id}",
            extraction_confidence=0.80,
        )

        annotation = self._create_annotation(
            conflict_description=f"Negation of: {triple.source_text[:50]}...",
            original_id=triple.source_document_id,
        )

        return AnnotatedConflictTriple(
            triple=negation_triple,
            annotation=annotation,
            paired_triple_id=triple.source_document_id,
        )


# =============================================================================
# 7. MODALITÄTS-KONFLIKTE
# =============================================================================

class ModalityConflictGenerator(ConflictGenerator):
    """
    Generator für Modalitäts-Konflikte (Kategorie 7).

    Erzeugt Aussagen mit unterschiedlichen Gewissheitsgraden.
    """

    MODALITY_MARKERS = {
        "certain": ["definitely", "certainly", "without doubt", "clearly"],
        "probable": ["probably", "likely", "most likely", "presumably"],
        "possible": ["possibly", "perhaps", "maybe", "might be"],
        "uncertain": ["allegedly", "reportedly", "supposedly", "claimed to be"],
    }

    MODALITY_CONFIDENCE = {
        "certain": 0.95,
        "probable": 0.75,
        "possible": 0.50,
        "uncertain": 0.30,
    }

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.MODALITY

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Modalitäts-Konflikte."""
        conflicts = []
        selected = random.sample(base_triples, min(num_conflicts, len(base_triples)))

        for triple in selected:
            conflict = self._generate_modality_variant(triple)
            if conflict:
                conflicts.append(conflict)

        logger.info(f"[MODALITY] Generiert: {len(conflicts)} Konflikte")
        return conflicts

    def _generate_modality_variant(
        self,
        triple: Triple
    ) -> Optional[AnnotatedConflictTriple]:
        """Generiert eine Modalitäts-Variante."""
        # Wähle zufällige Modalität
        modality = random.choice(list(self.MODALITY_MARKERS.keys()))
        marker = random.choice(self.MODALITY_MARKERS[modality])
        confidence = self.MODALITY_CONFIDENCE[modality]

        modal_text = f"{marker} {triple.source_text}"

        modal_triple = Triple(
            subject=triple.subject,
            predicate=triple.predicate,
            object=triple.object,
            source_text=modal_text,
            source_document_id=f"modal_{triple.source_document_id}",
            extraction_confidence=confidence,
        )

        annotation = self._create_annotation(
            conflict_description=f"Modality: {modality} ({marker})",
            expected_confidence_delta=confidence - triple.extraction_confidence,
            original_id=triple.source_document_id,
        )
        annotation.ground_truth_action = GroundTruthAction.WEIGHT

        return AnnotatedConflictTriple(
            triple=modal_triple,
            annotation=annotation,
            paired_triple_id=triple.source_document_id,
        )


# =============================================================================
# 8. SOURCE-QUALITÄTS-KONFLIKTE
# =============================================================================

class SourceQualityConflictGenerator(ConflictGenerator):
    """
    Generator für Source-Qualitäts-Konflikte (Kategorie 8).

    Erzeugt gleiche Aussagen aus unterschiedlich vertrauenswürdigen Quellen.
    """

    SOURCE_RELIABILITY = {
        "scientific_paper": 0.95,
        "wikipedia": 0.85,
        "news_major": 0.75,
        "news_minor": 0.60,
        "blog": 0.40,
        "social_media": 0.25,
        "anonymous_forum": 0.15,
    }

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.SOURCE_QUALITY

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Source-Qualitäts-Varianten."""
        conflicts = []
        selected = random.sample(base_triples, min(num_conflicts, len(base_triples)))

        for triple in selected:
            variants = self._generate_source_variants(triple)
            conflicts.extend(variants)

        logger.info(f"[SOURCE_QUALITY] Generiert: {len(conflicts)} Varianten")
        return conflicts[:num_conflicts]

    def _generate_source_variants(
        self,
        triple: Triple
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Varianten mit verschiedenen Quellen."""
        variants = []

        # Wähle 2 verschiedene Quellen
        sources = random.sample(list(self.SOURCE_RELIABILITY.items()), 2)

        for source_name, reliability in sources:
            source_triple = Triple(
                subject=triple.subject,
                predicate=triple.predicate,
                object=triple.object,
                source_text=f"[{source_name.upper()}] {triple.source_text}",
                source_document_id=f"{source_name}_{triple.source_document_id}",
                extraction_confidence=reliability,
            )

            annotation = self._create_annotation(
                conflict_description=f"Source: {source_name} (reliability={reliability})",
                expected_confidence_delta=reliability - 0.5,
                original_id=triple.source_document_id,
            )
            annotation.ground_truth_action = GroundTruthAction.WEIGHT

            variants.append(AnnotatedConflictTriple(
                triple=source_triple,
                annotation=annotation,
            ))

        return variants


# =============================================================================
# 9. SCHEMA-HETEROGENITÄT
# =============================================================================

class SchemaConflictGenerator(ConflictGenerator):
    """
    Generator für Schema-Heterogenität (Kategorie 9).

    Erzeugt semantisch äquivalente Aussagen mit verschiedenen Schemata.
    """

    SCHEMA_EQUIVALENCES = {
        "birthPlace": ["placeOfBirth", "bornIn", "nativePlaceOf", "P19"],
        "deathPlace": ["placeOfDeath", "diedIn", "P20"],
        "spouse": ["marriedTo", "hasSpouse", "P26"],
        "worksAt": ["employedBy", "affiliatedWith", "P108"],
        "capital": ["capitalOf", "hasCapital", "P36"],
        "creator": ["createdBy", "author", "madeBy", "P170"],
        "locatedIn": ["location", "isIn", "partOf", "P131"],
    }

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.SCHEMA

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert Schema-Varianten."""
        conflicts = []
        selected = random.sample(base_triples, min(num_conflicts, len(base_triples)))

        for triple in selected:
            variant = self._generate_schema_variant(triple)
            if variant:
                conflicts.append(variant)

        logger.info(f"[SCHEMA] Generiert: {len(conflicts)} Varianten")
        return conflicts

    def _generate_schema_variant(
        self,
        triple: Triple
    ) -> Optional[AnnotatedConflictTriple]:
        """Generiert eine Schema-Variante."""
        original_pred = triple.predicate

        # Suche äquivalente Prädikate
        equivalent = None
        for canonical, alternatives in self.SCHEMA_EQUIVALENCES.items():
            all_forms = [canonical] + alternatives
            if original_pred.lower() in [f.lower() for f in all_forms]:
                # Wähle eine andere Form
                other_forms = [f for f in all_forms if f.lower() != original_pred.lower()]
                if other_forms:
                    equivalent = random.choice(other_forms)
                    break

        if not equivalent:
            # Fallback: Präfix hinzufügen
            equivalent = f"schema:{original_pred}"

        schema_triple = Triple(
            subject=triple.subject,
            predicate=equivalent,
            object=triple.object,
            source_text=f"[SCHEMA: {equivalent}] {triple.source_text}",
            source_document_id=f"schema_{triple.source_document_id}",
            extraction_confidence=triple.extraction_confidence,
        )

        annotation = self._create_annotation(
            conflict_description=f"Schema: {original_pred} -> {equivalent}",
            original_id=triple.source_document_id,
        )

        return AnnotatedConflictTriple(
            triple=schema_triple,
            annotation=annotation,
            paired_triple_id=triple.source_document_id,
        )


# =============================================================================
# 10. NUMERISCHE PRÄZISION
# =============================================================================

class NumericalConflictGenerator(ConflictGenerator):
    """
    Generator für numerische Präzisions-Konflikte (Kategorie 10).

    Erzeugt numerische Werte mit unterschiedlicher Präzision.
    """

    @property
    def conflict_type(self) -> ConflictType:
        return ConflictType.NUMERICAL

    def generate(
        self,
        base_triples: List[Triple],
        num_conflicts: int = 10
    ) -> List[AnnotatedConflictTriple]:
        """Generiert numerische Präzisions-Varianten."""
        conflicts = []

        # Generiere Beispiel-Paare
        numerical_examples = [
            ("1879", "late 1870s", "the 1870s"),
            ("100 km", "approximately 100 km", "about 100 kilometers"),
            ("3.14159", "3.14", "roughly 3"),
            ("1,000,000", "about 1 million", "around a million"),
            ("50%", "about half", "roughly 50 percent"),
        ]

        for precise, approx1, approx2 in numerical_examples[:num_conflicts]:
            pair = self._generate_numerical_pair(precise, [approx1, approx2])
            conflicts.extend(pair)

        logger.info(f"[NUMERICAL] Generiert: {len(conflicts)} Varianten")
        return conflicts

    def _generate_numerical_pair(
        self,
        precise_value: str,
        approximate_values: List[str]
    ) -> List[AnnotatedConflictTriple]:
        """Generiert ein Paar mit unterschiedlicher Präzision."""
        pairs = []
        subject = self._create_entity("Measurement", EntityType.CONCEPT)

        # Präziser Wert
        precise_triple = Triple(
            subject=subject,
            predicate="hasValue",
            object=self._create_entity(precise_value, EntityType.CONCEPT),
            source_text=f"The value is exactly {precise_value}",
            source_document_id=f"numerical_precise_{precise_value}",
            extraction_confidence=0.95,
        )

        annotation = self._create_annotation(
            conflict_description=f"Numerical (precise): {precise_value}",
        )
        annotation.ground_truth_action = GroundTruthAction.ACCEPT

        pairs.append(AnnotatedConflictTriple(triple=precise_triple, annotation=annotation))

        # Approximative Werte
        for approx in approximate_values:
            approx_triple = Triple(
                subject=subject,
                predicate="hasValue",
                object=self._create_entity(approx, EntityType.CONCEPT),
                source_text=f"The value is {approx}",
                source_document_id=f"numerical_approx_{approx.replace(' ', '_')}",
                extraction_confidence=0.75,
            )

            approx_annotation = self._create_annotation(
                conflict_description=f"Numerical (approx): {approx}",
            )
            approx_annotation.ground_truth_action = GroundTruthAction.MERGE

            pairs.append(AnnotatedConflictTriple(
                triple=approx_triple,
                annotation=approx_annotation,
                paired_triple_id=precise_triple.source_document_id,
            ))

        return pairs


# =============================================================================
# COMPREHENSIVE GENERATOR (Hauptklasse)
# =============================================================================

class ComprehensiveConflictGenerator:
    """
    Haupt-Generator der alle 10 Konflikt-Kategorien orchestriert.

    Verwendet alle spezialisierten Generatoren um ein wissenschaftlich
    vollständiges Testset zu erstellen.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

        # Initialisiere alle Generatoren
        self.generators: Dict[ConflictType, ConflictGenerator] = {
            ConflictType.FACTUAL: FactualConflictGenerator(seed),
            ConflictType.TEMPORAL: TemporalConflictGenerator(seed),
            ConflictType.GRANULARITY: GranularityConflictGenerator(seed),
            ConflictType.ENTITY_VARIANT: EntityVariantConflictGenerator(seed),
            ConflictType.IMPLICIT: ImplicitConflictGenerator(seed),
            ConflictType.NEGATION: NegationConflictGenerator(seed),
            ConflictType.MODALITY: ModalityConflictGenerator(seed),
            ConflictType.SOURCE_QUALITY: SourceQualityConflictGenerator(seed),
            ConflictType.SCHEMA: SchemaConflictGenerator(seed),
            ConflictType.NUMERICAL: NumericalConflictGenerator(seed),
        }

    def generate_all(
        self,
        base_triples: List[Triple],
        conflicts_per_type: int = 10,
        types: Optional[List[ConflictType]] = None
    ) -> Dict[ConflictType, List[AnnotatedConflictTriple]]:
        """
        Generiert Konflikte für alle (oder ausgewählte) Kategorien.

        Args:
            base_triples: Basis-Triples für Konflikt-Generierung
            conflicts_per_type: Anzahl Konflikte pro Kategorie
            types: Optional, nur diese Typen generieren

        Returns:
            Dict mit Konflikt-Typ als Key und Liste von Konflikten als Value
        """
        if types is None:
            types = list(ConflictType)

        results = {}
        for conflict_type in types:
            generator = self.generators.get(conflict_type)
            if generator:
                conflicts = generator.generate(base_triples, conflicts_per_type)
                results[conflict_type] = conflicts
                logger.info(f"  {conflict_type.value}: {len(conflicts)} Konflikte")

        return results

    def generate_flat(
        self,
        base_triples: List[Triple],
        conflicts_per_type: int = 10,
        types: Optional[List[ConflictType]] = None
    ) -> List[AnnotatedConflictTriple]:
        """
        Generiert alle Konflikte als flache Liste.

        Returns:
            Flache Liste aller generierten Konflikte
        """
        all_conflicts = self.generate_all(base_triples, conflicts_per_type, types)
        return [
            conflict
            for conflicts in all_conflicts.values()
            for conflict in conflicts
        ]

    def get_statistics(
        self,
        conflicts: Dict[ConflictType, List[AnnotatedConflictTriple]]
    ) -> Dict[str, Any]:
        """Berechnet Statistiken über generierte Konflikte."""
        stats = {
            "total": sum(len(c) for c in conflicts.values()),
            "per_type": {ct.value: len(c) for ct, c in conflicts.items()},
            "per_difficulty": {"easy": 0, "medium": 0, "hard": 0},
            "per_ground_truth": {action.value: 0 for action in GroundTruthAction},
            "requires_world_knowledge": 0,
            "requires_temporal_data": 0,
            "requires_ontology": 0,
        }

        for conflict_list in conflicts.values():
            for conflict in conflict_list:
                ann = conflict.annotation
                stats["per_difficulty"][ann.difficulty] += 1
                stats["per_ground_truth"][ann.ground_truth_action.value] += 1
                if ann.requires_world_knowledge:
                    stats["requires_world_knowledge"] += 1
                if ann.requires_temporal_data:
                    stats["requires_temporal_data"] += 1
                if ann.requires_ontology:
                    stats["requires_ontology"] += 1

        return stats


# =============================================================================
# MAIN (Test)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test mit Beispiel-Triples
    from src.models.entities import Entity, EntityType, Triple

    base_triples = [
        Triple(
            subject=Entity(name="Albert Einstein", entity_type=EntityType.PERSON),
            predicate="birthPlace",
            object=Entity(name="Ulm", entity_type=EntityType.LOCATION),
            source_text="Albert Einstein was born in Ulm, Germany.",
            source_document_id="test_1",
            extraction_confidence=0.9,
        ),
        Triple(
            subject=Entity(name="The Beatles", entity_type=EntityType.ORGANIZATION),
            predicate="formed",
            object=Entity(name="1960", entity_type=EntityType.CONCEPT),
            source_text="The Beatles were formed in 1960.",
            source_document_id="test_2",
            extraction_confidence=0.85,
        ),
    ]

    generator = ComprehensiveConflictGenerator(seed=42)
    all_conflicts = generator.generate_all(base_triples, conflicts_per_type=3)

    print("\n" + "=" * 60)
    print("GENERIERTE KONFLIKTE")
    print("=" * 60)

    for conflict_type, conflicts in all_conflicts.items():
        print(f"\n{conflict_type.value.upper()}: {len(conflicts)} Konflikte")
        for c in conflicts[:2]:  # Zeige nur erste 2
            print(f"  - {c.annotation.conflict_description}")
            print(f"    Ground Truth: {c.ground_truth_action.value}")

    stats = generator.get_statistics(all_conflicts)
    print("\n" + "=" * 60)
    print("STATISTIKEN")
    print("=" * 60)
    print(f"Gesamt: {stats['total']}")
    print(f"Per Typ: {stats['per_type']}")
    print(f"Per Schwierigkeit: {stats['per_difficulty']}")
    print(f"Per Ground Truth: {stats['per_ground_truth']}")
