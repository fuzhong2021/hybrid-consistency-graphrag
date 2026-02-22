# src/evaluation/hybrid_evaluator.py
"""
Hybrid Evaluation Framework.

Kombiniert zwei komplementäre Evaluationsansätze:
- Teil A: Kontrolliertes Experiment mit synthetischen Konflikten (perfekte Ground Truth)
- Teil B: Realistisches Experiment mit Supporting Facts vs. All Paragraphs (Proxy Ground Truth)

Wissenschaftliche Grundlage:
- Teil A beweist: "Das Modul erkennt Fehler korrekt" (P/R/F1)
- Teil B beweist: "Das Modul verbessert Graph-Qualität in realistischen Szenarien"
"""

import json
import logging
import random
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from copy import deepcopy

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample

logger = logging.getLogger(__name__)


# =============================================================================
# TEIL A: Kontrolliertes Experiment - Synthetische Konflikte
# =============================================================================

class ConflictCategory(Enum):
    """Kategorien synthetischer Konflikte."""
    DUPLICATE = "duplicate"           # Gleiche Entität, anderer Name
    CONTRADICTION = "contradiction"   # Gleiches Subjekt+Prädikat, anderes Objekt
    SCHEMA_VIOLATION = "schema"       # Ungültige Typen oder Self-Loops
    CARDINALITY = "cardinality"       # Mehrfache Werte für single-value Prädikate
    CORRECT = "correct"               # Korrekte Updates (Kontrolle)


@dataclass
class LabeledTriple:
    """Triple mit Ground-Truth Label."""
    triple: Triple
    expected_decision: str  # "ACCEPT" oder "REJECT"
    conflict_category: ConflictCategory
    mutation_details: str = ""  # Beschreibung der Mutation


@dataclass
class ConfusionMatrix:
    """
    Confusion Matrix für FEHLERERKENNUNG.

    Aus der Perspektive "Fehler erkennen":
    - TP: Fehler korrekt erkannt (expected REJECT, actual REJECT)
    - TN: Korrekt als gut erkannt (expected ACCEPT, actual ACCEPT)
    - FP: Fälschlicherweise abgelehnt (expected ACCEPT, actual REJECT)
    - FN: Fehler übersehen (expected REJECT, actual ACCEPT)
    """
    tp: int = 0  # True Positive (Fehler korrekt erkannt)
    tn: int = 0  # True Negative (Korrekt als gut erkannt)
    fp: int = 0  # False Positive (Fälschlicherweise abgelehnt)
    fn: int = 0  # False Negative (Fehler übersehen)

    @property
    def precision(self) -> float:
        """Precision: Wie viele Ablehnungen waren berechtigt?"""
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Recall: Wie viele Fehler wurden erkannt?"""
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total

    @property
    def acceptance_rate(self) -> float:
        """Akzeptanzrate: Wie viele wurden akzeptiert?"""
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tn + self.fn) / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "acceptance_rate": self.acceptance_rate,
        }


class MutationEngine:
    """
    Generiert synthetische Konflikte aus existierenden Triples.

    Mutation-Strategien:
    1. DUPLICATE: Namens-Variationen (Abkürzungen, Titel, etc.)
    2. CONTRADICTION: Falsche Objekte (andere Entität gleichen Typs)
    3. SCHEMA: Self-Loops, ungültige Typen
    4. CARDINALITY: Zweiter Wert für single-value Prädikate
    5. CORRECT: Unveränderte oder leicht modifizierte (aber korrekte) Triples
    """

    # Namens-Variationen für Duplikat-Generierung
    NAME_PREFIXES = ["Mr.", "Mrs.", "Dr.", "Prof.", "Sir", "The"]
    NAME_SUFFIXES = ["Jr.", "Sr.", "III", "PhD", "MD"]

    # Single-value Prädikate (für Kardinalitäts-Konflikte)
    SINGLE_VALUE_PREDICATES = {
        "GEBOREN_IN", "GESTORBEN_IN", "GEBURTSORT", "TODESTAG",
        "born_in", "died_in", "birthplace", "date_of_death",
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.entity_pool: List[Entity] = []  # Für Widerspruchs-Generierung

    def collect_entities(self, triples: List[Triple]) -> None:
        """Sammelt Entitäten für spätere Widerspruchs-Generierung."""
        seen = set()
        for t in triples:
            for entity in [t.subject, t.object]:
                key = (entity.name, entity.entity_type.value)
                if key not in seen:
                    self.entity_pool.append(entity)
                    seen.add(key)

    def generate_mutations(
        self,
        seed_triples: List[Triple],
        distribution: Dict[ConflictCategory, float] = None,
    ) -> List[LabeledTriple]:
        """
        Generiert mutierte Triples mit Ground-Truth Labels.

        Args:
            seed_triples: Original-Triples aus der Seed-Phase
            distribution: Verteilung der Konflikt-Kategorien (summe = 1.0)

        Returns:
            Liste von LabeledTriple mit erwarteten Entscheidungen
        """
        if distribution is None:
            distribution = {
                ConflictCategory.DUPLICATE: 0.20,
                ConflictCategory.CONTRADICTION: 0.20,
                ConflictCategory.SCHEMA_VIOLATION: 0.20,
                ConflictCategory.CARDINALITY: 0.20,
                ConflictCategory.CORRECT: 0.20,
            }

        # Sammle Entitäten für Widerspruchs-Generierung
        self.collect_entities(seed_triples)

        labeled = []

        for triple in seed_triples:
            # Wähle Kategorie basierend auf Verteilung
            category = self._select_category(distribution)

            if category == ConflictCategory.DUPLICATE:
                labeled.extend(self._generate_duplicates(triple))
            elif category == ConflictCategory.CONTRADICTION:
                labeled.extend(self._generate_contradictions(triple))
            elif category == ConflictCategory.SCHEMA_VIOLATION:
                labeled.extend(self._generate_schema_violations(triple))
            elif category == ConflictCategory.CARDINALITY:
                labeled.extend(self._generate_cardinality_violations(triple))
            else:  # CORRECT
                labeled.extend(self._generate_correct_updates(triple))

        return labeled

    def _select_category(self, distribution: Dict[ConflictCategory, float]) -> ConflictCategory:
        """Wählt eine Kategorie basierend auf der Verteilung."""
        r = self.rng.random()
        cumulative = 0.0
        for category, prob in distribution.items():
            cumulative += prob
            if r <= cumulative:
                return category
        return ConflictCategory.CORRECT

    def _generate_duplicates(self, triple: Triple) -> List[LabeledTriple]:
        """Generiert Duplikate durch Namens-Variationen."""
        results = []

        # Variation 1: Abkürzung des Subjekt-Namens
        name = triple.subject.name
        if " " in name:
            parts = name.split()
            abbreviated = f"{parts[0][0]}. {' '.join(parts[1:])}"

            mutated = self._copy_triple_with_new_subject(triple, abbreviated)
            results.append(LabeledTriple(
                triple=mutated,
                expected_decision="REJECT",
                conflict_category=ConflictCategory.DUPLICATE,
                mutation_details=f"Abbreviated: '{name}' → '{abbreviated}'",
            ))

        # Variation 2: Mit Präfix
        if self.rng.random() > 0.5:
            prefix = self.rng.choice(self.NAME_PREFIXES)
            prefixed = f"{prefix} {name}"

            mutated = self._copy_triple_with_new_subject(triple, prefixed)
            results.append(LabeledTriple(
                triple=mutated,
                expected_decision="REJECT",
                conflict_category=ConflictCategory.DUPLICATE,
                mutation_details=f"Prefixed: '{name}' → '{prefixed}'",
            ))

        return results if results else [self._generate_correct_updates(triple)[0]]

    def _generate_contradictions(self, triple: Triple) -> List[LabeledTriple]:
        """Generiert widersprüchliche Triples (gleiches S+P, anderes O)."""
        # Finde alternative Objekte gleichen Typs
        obj_type = triple.object.entity_type
        candidates = [
            e for e in self.entity_pool
            if e.entity_type == obj_type and e.name != triple.object.name
        ]

        if not candidates:
            return self._generate_correct_updates(triple)

        alt_obj = self.rng.choice(candidates)

        mutated = Triple(
            subject=deepcopy(triple.subject),
            predicate=triple.predicate,
            object=Entity(
                name=alt_obj.name,
                entity_type=alt_obj.entity_type,
                source_document=triple.object.source_document,
            ),
            source_text=f"[MUTATED] {triple.source_text}",
            source_document_id=triple.source_document_id,
            extraction_confidence=triple.extraction_confidence,
        )

        return [LabeledTriple(
            triple=mutated,
            expected_decision="REJECT",
            conflict_category=ConflictCategory.CONTRADICTION,
            mutation_details=f"Object changed: '{triple.object.name}' → '{alt_obj.name}'",
        )]

    def _generate_schema_violations(self, triple: Triple) -> List[LabeledTriple]:
        """Generiert Schema-Verletzungen (Self-Loops, ungültige Typen)."""
        results = []

        # Self-Loop (Subjekt = Objekt)
        self_loop = Triple(
            subject=deepcopy(triple.subject),
            predicate=triple.predicate,
            object=deepcopy(triple.subject),  # Same as subject!
            source_text=f"[SELF-LOOP] {triple.source_text}",
            source_document_id=triple.source_document_id,
            extraction_confidence=triple.extraction_confidence,
        )
        results.append(LabeledTriple(
            triple=self_loop,
            expected_decision="REJECT",
            conflict_category=ConflictCategory.SCHEMA_VIOLATION,
            mutation_details=f"Self-loop: {triple.subject.name} → {triple.subject.name}",
        ))

        return results

    def _generate_cardinality_violations(self, triple: Triple) -> List[LabeledTriple]:
        """Generiert Kardinalitäts-Verletzungen (zweiter Wert für single-value)."""
        # Nur für single-value Prädikate relevant
        if triple.predicate.upper() not in self.SINGLE_VALUE_PREDICATES:
            # Generiere stattdessen eine Contradiction
            return self._generate_contradictions(triple)

        # Finde alternatives Objekt
        candidates = [
            e for e in self.entity_pool
            if e.entity_type == triple.object.entity_type
            and e.name != triple.object.name
        ]

        if not candidates:
            return self._generate_schema_violations(triple)

        alt_obj = self.rng.choice(candidates)

        mutated = Triple(
            subject=deepcopy(triple.subject),
            predicate=triple.predicate,
            object=Entity(
                name=alt_obj.name,
                entity_type=alt_obj.entity_type,
                source_document=triple.object.source_document,
            ),
            source_text=f"[CARDINALITY] {triple.source_text}",
            source_document_id=triple.source_document_id,
            extraction_confidence=triple.extraction_confidence,
        )

        return [LabeledTriple(
            triple=mutated,
            expected_decision="REJECT",
            conflict_category=ConflictCategory.CARDINALITY,
            mutation_details=f"Second value for {triple.predicate}: '{alt_obj.name}'",
        )]

    def _generate_correct_updates(self, triple: Triple) -> List[LabeledTriple]:
        """Generiert korrekte Updates (leichte Modifikationen, aber valide)."""
        # Kopiere Triple mit leicht modifizierter source_text
        mutated = Triple(
            subject=deepcopy(triple.subject),
            predicate=triple.predicate,
            object=deepcopy(triple.object),
            source_text=f"[UPDATE] {triple.source_text}",
            source_document_id=f"update_{triple.source_document_id}",
            extraction_confidence=triple.extraction_confidence,
        )

        return [LabeledTriple(
            triple=mutated,
            expected_decision="ACCEPT",
            conflict_category=ConflictCategory.CORRECT,
            mutation_details="Valid update (new source)",
        )]

    def _copy_triple_with_new_subject(self, triple: Triple, new_name: str) -> Triple:
        """Erstellt Triple-Kopie mit neuem Subjekt-Namen."""
        return Triple(
            subject=Entity(
                name=new_name,
                entity_type=triple.subject.entity_type,
                source_document=triple.subject.source_document,
            ),
            predicate=triple.predicate,
            object=deepcopy(triple.object),
            source_text=f"[DUPLICATE] {triple.source_text}",
            source_document_id=triple.source_document_id,
            extraction_confidence=triple.extraction_confidence,
        )


@dataclass
class ControlledExperimentResult:
    """Ergebnis des kontrollierten Experiments (Teil A)."""
    total_mutations: int = 0

    # Confusion Matrices pro Kategorie
    by_category: Dict[str, ConfusionMatrix] = field(default_factory=dict)

    # Aggregierte Metriken
    overall: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    # Timing
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mutations": self.total_mutations,
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "overall": self.overall.to_dict(),
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# TEIL B: Realistisches Experiment - Supporting Facts vs. All
# =============================================================================

@dataclass
class SourceTaggedTriple:
    """Triple mit Quellen-Tag (Supporting vs. Distractor)."""
    triple: Triple
    is_from_supporting: bool
    paragraph_title: str


@dataclass
class RealisticExperimentResult:
    """Ergebnis des realistischen Experiments (Teil B)."""
    # Seed-Phase (nur Supporting Facts)
    seed_triples: int = 0
    seed_accepted: int = 0

    # Update-Phase (alle Paragraphen)
    update_total: int = 0
    update_from_supporting: int = 0
    update_from_distractor: int = 0

    # Entscheidungen
    supporting_accepted: int = 0
    supporting_rejected: int = 0
    distractor_accepted: int = 0
    distractor_rejected: int = 0

    # Timing
    duration_seconds: float = 0.0

    @property
    def signal_preservation_rate(self) -> float:
        """Wie viel Signal (Supporting) wurde erhalten?"""
        total_supporting = self.supporting_accepted + self.supporting_rejected
        if total_supporting == 0:
            return 0.0
        return self.supporting_accepted / total_supporting

    @property
    def noise_rejection_rate(self) -> float:
        """Wie viel Noise (Distractor) wurde gefiltert?"""
        total_distractor = self.distractor_accepted + self.distractor_rejected
        if total_distractor == 0:
            return 0.0
        return self.distractor_rejected / total_distractor

    @property
    def information_precision(self) -> float:
        """Precision: Anteil korrekter unter akzeptierten."""
        total_accepted = self.supporting_accepted + self.distractor_accepted
        if total_accepted == 0:
            return 0.0
        return self.supporting_accepted / total_accepted

    @property
    def information_f1(self) -> float:
        """F1 aus Preservation (Recall) und Precision."""
        p = self.information_precision
        r = self.signal_preservation_rate
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": {
                "triples": self.seed_triples,
                "accepted": self.seed_accepted,
            },
            "update": {
                "total": self.update_total,
                "from_supporting": self.update_from_supporting,
                "from_distractor": self.update_from_distractor,
            },
            "decisions": {
                "supporting_accepted": self.supporting_accepted,
                "supporting_rejected": self.supporting_rejected,
                "distractor_accepted": self.distractor_accepted,
                "distractor_rejected": self.distractor_rejected,
            },
            "metrics": {
                "signal_preservation_rate": self.signal_preservation_rate,
                "noise_rejection_rate": self.noise_rejection_rate,
                "information_precision": self.information_precision,
                "information_f1": self.information_f1,
            },
            "duration_seconds": self.duration_seconds,
        }


class SupportingFactExtractor:
    """
    Extrahiert Triples separat aus Supporting Facts und Distraktoren.

    Nutzt die HotpotQA-Labels um Ground Truth zu etablieren:
    - Supporting Facts → "Signal" (relevant für Antwort)
    - Distraktoren → "Noise" (irrelevant aber thematisch ähnlich)
    """

    def __init__(self, triple_extractor: Any = None):
        """
        Args:
            triple_extractor: Optional custom extractor, sonst Heuristik
        """
        self.extractor = triple_extractor

    def extract_by_source(
        self,
        example: QAExample,
    ) -> Tuple[List[SourceTaggedTriple], List[SourceTaggedTriple]]:
        """
        Extrahiert Triples getrennt nach Quelle.

        Returns:
            (supporting_triples, distractor_triples)
        """
        # Parse supporting facts titles from SupportingFact objects
        sf_titles = set()
        if hasattr(example, 'supporting_facts') and example.supporting_facts:
            for sf in example.supporting_facts:
                # SupportingFact is a dataclass with .title attribute
                if hasattr(sf, 'title'):
                    sf_titles.add(sf.title)
                elif isinstance(sf, dict):
                    sf_titles.add(sf.get('title', ''))

        supporting_triples = []
        distractor_triples = []

        # Extrahiere aus jedem Paragraph (QAExample verwendet context_paragraphs)
        paragraphs = []
        if hasattr(example, 'context_paragraphs') and example.context_paragraphs:
            paragraphs = example.context_paragraphs
        elif hasattr(example, 'paragraphs') and example.paragraphs:
            paragraphs = example.paragraphs

        for para in paragraphs:
            title = para.get('title', '')
            sentences = para.get('sentences', [])
            text = ' '.join(sentences) if isinstance(sentences, list) else str(sentences)

            is_supporting = title in sf_titles

            # Extrahiere Triples aus diesem Paragraph
            triples = self._extract_from_paragraph(title, text, example)

            for triple in triples:
                tagged = SourceTaggedTriple(
                    triple=triple,
                    is_from_supporting=is_supporting,
                    paragraph_title=title,
                )

                if is_supporting:
                    supporting_triples.append(tagged)
                else:
                    distractor_triples.append(tagged)

        return supporting_triples, distractor_triples

    def _extract_from_paragraph(
        self,
        title: str,
        text: str,
        example: QAExample,
    ) -> List[Triple]:
        """Extrahiert Triples aus einem einzelnen Paragraph."""
        if self.extractor is not None:
            # Verwende den bereitgestellten Extractor
            try:
                return self.extractor.extract(text)
            except Exception:
                pass

        # Fallback: Einfache heuristische Extraktion
        return self._heuristic_extraction(title, text, example)

    def _heuristic_extraction(
        self,
        title: str,
        text: str,
        example: QAExample,
    ) -> List[Triple]:
        """
        Heuristische Triple-Extraktion.

        Strategie:
        1. Paragraph-Titel als Hauptentität
        2. Finde andere Paragraphen-Titel im Text
        3. Extrahiere Eigennamen aus dem Text als zusätzliche Entitäten
        """
        import re
        triples = []

        if not title or not text:
            return triples

        # Hauptentität ist der Paragraph-Titel
        main_entity = Entity(
            name=title,
            entity_type=self._infer_entity_type(title),
            source_document=example.id,
        )

        # Sammle alle Ziel-Entitäten
        target_entities = set()

        # 1. Andere Paragraphen-Titel im Text
        paragraphs = example.context_paragraphs if hasattr(example, 'context_paragraphs') else []
        all_titles = [p.get('title', '') for p in paragraphs]
        for other_title in all_titles:
            if other_title and other_title != title and other_title in text:
                target_entities.add(other_title)

        # 2. Extrahiere Eigennamen (Wörter mit Großbuchstaben, nicht am Satzanfang)
        # Pattern: Kapitalisierte Wörter die nicht am Satzanfang stehen
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                # Ignoriere erstes Wort (Satzanfang) und kurze Wörter
                if i == 0 or len(word) < 3:
                    continue
                # Entferne Satzzeichen
                clean_word = re.sub(r'[^\w\s]', '', word)
                # Prüfe ob kapitalisiert und nicht komplett uppercase
                if clean_word and clean_word[0].isupper() and not clean_word.isupper():
                    # Mehrwort-Namen erkennen (z.B. "New York")
                    if i + 1 < len(words):
                        next_word = re.sub(r'[^\w\s]', '', words[i + 1])
                        if next_word and next_word[0].isupper() and not next_word.isupper():
                            multi_word = f"{clean_word} {next_word}"
                            if multi_word != title and len(multi_word) > 3:
                                target_entities.add(multi_word)
                    if clean_word != title and clean_word not in ['The', 'A', 'An', 'In', 'On', 'At', 'For', 'And', 'Or', 'But', 'Is', 'Was', 'Are', 'Were', 'He', 'She', 'It', 'They', 'His', 'Her', 'Their']:
                        target_entities.add(clean_word)

        # Erstelle Triples
        for target_name in target_entities:
            if target_name == title:
                continue

            other_entity = Entity(
                name=target_name,
                entity_type=self._infer_entity_type(target_name),
                source_document=example.id,
            )

            triple = Triple(
                subject=main_entity,
                predicate="RELATED_TO",
                object=other_entity,
                source_text=text[:200],
                source_document_id=example.id,
                extraction_confidence=0.7,
            )
            triples.append(triple)

        return triples

    def _infer_entity_type(self, name: str) -> EntityType:
        """Inferiert Entity-Typ aus Namen."""
        name_lower = name.lower()

        # Location indicators
        location_words = ['city', 'country', 'state', 'river', 'mountain', 'island']
        if any(w in name_lower for w in location_words):
            return EntityType.LOCATION

        # Organization indicators
        org_words = ['university', 'company', 'inc', 'corp', 'organization', 'band', 'team']
        if any(w in name_lower for w in org_words):
            return EntityType.ORGANIZATION

        # Event indicators
        event_words = ['war', 'battle', 'election', 'festival', 'championship']
        if any(w in name_lower for w in event_words):
            return EntityType.EVENT

        # Film/Album/Work indicators (often in parentheses)
        if '(' in name and ')' in name:
            paren_content = name[name.find('(')+1:name.find(')')]
            if paren_content.lower() in ['film', 'album', 'book', 'song', 'tv series']:
                return EntityType.CONCEPT

        # Default to PERSON for proper names
        if name[0].isupper():
            return EntityType.PERSON

        return EntityType.CONCEPT


# =============================================================================
# HYBRID EVALUATOR - Kombiniert Teil A und Teil B
# =============================================================================

@dataclass
class HybridEvaluationResult:
    """Kombiniertes Ergebnis beider Experimente."""
    controlled: ControlledExperimentResult
    realistic: RealisticExperimentResult

    # Konfiguration
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_a_controlled": self.controlled.to_dict(),
            "part_b_realistic": self.realistic.to_dict(),
            "config": self.config,
        }


class HybridEvaluator:
    """
    Führt beide Evaluations-Teile durch.

    Teil A: Kontrolliertes Experiment
    - Seed: Saubere Triples aus HotpotQA
    - Update: Synthetisch mutierte Triples
    - Messung: P/R/F1 pro Konflikt-Kategorie

    Teil B: Realistisches Experiment
    - Seed: Triples nur aus Supporting Facts
    - Update: Triples aus allen Paragraphen
    - Messung: Signal Preservation, Noise Rejection, Information F1
    """

    def __init__(
        self,
        consistency_config: ConsistencyConfig,
        sample_size: int = 100,
        benchmark: str = "hotpotqa",
        llm_client: Any = None,
        seed: int = 42,
    ):
        self.consistency_config = consistency_config
        self.sample_size = sample_size
        self.benchmark = benchmark
        self.llm_client = llm_client
        self.seed = seed

        # Wird während der Ausführung initialisiert
        self.examples: List[QAExample] = []
        self.mutation_engine = MutationEngine(seed=seed)
        self.sf_extractor = SupportingFactExtractor()

    def run(
        self,
        run_part_a: bool = True,
        run_part_b: bool = True,
    ) -> HybridEvaluationResult:
        """Führt die Hybrid-Evaluation durch."""
        logger.info("=" * 60)
        logger.info("HYBRID EVALUATION FRAMEWORK")
        logger.info("=" * 60)
        logger.info(f"Sample Size: {self.sample_size}")
        logger.info(f"Benchmark: {self.benchmark}")
        logger.info(f"Part A (Controlled): {run_part_a}")
        logger.info(f"Part B (Realistic): {run_part_b}")

        # Lade Beispiele
        self._load_examples()

        # Teil A
        if run_part_a:
            logger.info("\n" + "=" * 40)
            logger.info("TEIL A: KONTROLLIERTES EXPERIMENT")
            logger.info("=" * 40)
            controlled_result = self._run_controlled_experiment()
        else:
            controlled_result = ControlledExperimentResult()

        # Teil B
        if run_part_b:
            logger.info("\n" + "=" * 40)
            logger.info("TEIL B: REALISTISCHES EXPERIMENT")
            logger.info("=" * 40)
            realistic_result = self._run_realistic_experiment()
        else:
            realistic_result = RealisticExperimentResult()

        result = HybridEvaluationResult(
            controlled=controlled_result,
            realistic=realistic_result,
            config={
                "sample_size": self.sample_size,
                "benchmark": self.benchmark,
                "seed": self.seed,
            }
        )

        self._print_summary(result)

        return result

    def _load_examples(self):
        """Lädt Benchmark-Beispiele."""
        loader = BenchmarkLoader()

        if self.benchmark.lower() == "hotpotqa":
            self.examples = loader.load_hotpotqa(
                split="validation",
                sample_size=self.sample_size,
            )
        elif self.benchmark.lower() == "musique":
            self.examples = loader.load_musique(
                split="validation",
                sample_size=self.sample_size,
            )
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

        logger.info(f"Loaded {len(self.examples)} examples")

    def _run_controlled_experiment(self) -> ControlledExperimentResult:
        """
        Teil A: Kontrolliertes Experiment mit synthetischen Konflikten.

        1. Extrahiere saubere Triples → Seed Graph
        2. Generiere Mutationen mit bekannten Labels
        3. Validiere Mutationen gegen Seed Graph
        4. Berechne P/R/F1 pro Kategorie
        """
        result = ControlledExperimentResult()
        start_time = time.time()

        # Initialisiere Confusion Matrices
        for cat in ConflictCategory:
            result.by_category[cat.value] = ConfusionMatrix()

        # Phase 1: Seed - Extrahiere und speichere saubere Triples
        logger.info("Phase 1: Extracting seed triples...")
        graph_repo = InMemoryGraphRepository()
        seed_triples = []

        from src.evaluation.comprehensive import EnhancedTripleExtractor
        extractor = EnhancedTripleExtractor()

        for example in self.examples:
            tagged_triples = extractor.extract_all_triples(example)
            for tagged in tagged_triples:
                triple = tagged.triple
                # Nur Self-Loop-freie Triples
                if triple.subject.name != triple.object.name:
                    fresh = self._create_fresh_triple(triple)
                    fresh.validation_status = ValidationStatus.ACCEPTED
                    try:
                        graph_repo.save_triple(fresh)
                        seed_triples.append(fresh)
                    except Exception:
                        pass

        logger.info(f"Seed complete: {len(seed_triples)} triples in graph")

        # Phase 2: Generiere Mutationen
        logger.info("Phase 2: Generating mutations...")
        labeled_mutations = self.mutation_engine.generate_mutations(seed_triples)
        result.total_mutations = len(labeled_mutations)
        logger.info(f"Generated {len(labeled_mutations)} mutations")

        # Phase 3: Initialisiere Orchestrator und validiere
        logger.info("Phase 3: Validating mutations...")
        orchestrator = ConsistencyOrchestrator(
            config=self.consistency_config,
            graph_repo=graph_repo,
            embedding_model=None,
            llm_client=self.llm_client,
            enable_metrics=True,
            always_check_duplicates=True,
        )

        for labeled in labeled_mutations:
            fresh = self._create_fresh_triple(labeled.triple)
            validated = orchestrator.process(fresh)

            actual_decision = "ACCEPT" if validated.validation_status == ValidationStatus.ACCEPTED else "REJECT"
            expected = labeled.expected_decision
            category = labeled.conflict_category.value

            # Update Confusion Matrix (Fehlererkennung-Perspektive)
            # TP = Fehler korrekt erkannt (expected REJECT, actual REJECT)
            # TN = Korrekt als gut erkannt (expected ACCEPT, actual ACCEPT)
            # FP = Fälschlicherweise abgelehnt (expected ACCEPT, actual REJECT)
            # FN = Fehler übersehen (expected REJECT, actual ACCEPT)
            cm = result.by_category[category]
            if expected == "REJECT" and actual_decision == "REJECT":
                cm.tp += 1
                result.overall.tp += 1
            elif expected == "ACCEPT" and actual_decision == "ACCEPT":
                cm.tn += 1
                result.overall.tn += 1
            elif expected == "ACCEPT" and actual_decision == "REJECT":
                cm.fp += 1
                result.overall.fp += 1
            else:  # expected REJECT, actual ACCEPT (missed error)
                cm.fn += 1
                result.overall.fn += 1

        result.duration_seconds = time.time() - start_time

        return result

    def _run_realistic_experiment(self) -> RealisticExperimentResult:
        """
        Teil B: Realistisches Experiment mit Supporting Facts.

        1. Seed: Extrahiere Triples NUR aus Supporting Facts
        2. Update: Extrahiere Triples aus ALLEN Paragraphen
        3. Validiere Update-Triples gegen Seed Graph
        4. Messe Signal Preservation und Noise Rejection
        """
        result = RealisticExperimentResult()
        start_time = time.time()

        # Frischer Graph
        graph_repo = InMemoryGraphRepository()

        # Phase 1: Seed - Nur Supporting Facts
        logger.info("Phase 1: Seeding with supporting facts only...")

        for example in self.examples:
            supporting, _ = self.sf_extractor.extract_by_source(example)

            for tagged in supporting:
                triple = tagged.triple
                if triple.subject.name != triple.object.name:
                    fresh = self._create_fresh_triple(triple)
                    fresh.validation_status = ValidationStatus.ACCEPTED
                    try:
                        graph_repo.save_triple(fresh)
                        result.seed_triples += 1
                        result.seed_accepted += 1
                    except Exception:
                        pass

        logger.info(f"Seed complete: {result.seed_accepted} triples from supporting facts")

        # Phase 2: Update - Alle Paragraphen
        logger.info("Phase 2: Updating with all paragraphs...")

        orchestrator = ConsistencyOrchestrator(
            config=self.consistency_config,
            graph_repo=graph_repo,
            embedding_model=None,
            llm_client=self.llm_client,
            enable_metrics=True,
            always_check_duplicates=True,
        )

        for example in self.examples:
            supporting, distractors = self.sf_extractor.extract_by_source(example)

            # Alle Triples durchgehen
            all_update_triples = supporting + distractors

            for tagged in all_update_triples:
                result.update_total += 1

                if tagged.is_from_supporting:
                    result.update_from_supporting += 1
                else:
                    result.update_from_distractor += 1

                triple = tagged.triple
                if triple.subject.name == triple.object.name:
                    # Self-loop - immer ablehnen
                    if tagged.is_from_supporting:
                        result.supporting_rejected += 1
                    else:
                        result.distractor_rejected += 1
                    continue

                fresh = self._create_fresh_triple(triple)
                validated = orchestrator.process(fresh)

                is_accepted = validated.validation_status == ValidationStatus.ACCEPTED

                if tagged.is_from_supporting:
                    if is_accepted:
                        result.supporting_accepted += 1
                        try:
                            graph_repo.save_triple(validated)
                        except Exception:
                            pass
                    else:
                        result.supporting_rejected += 1
                else:
                    if is_accepted:
                        result.distractor_accepted += 1
                        try:
                            graph_repo.save_triple(validated)
                        except Exception:
                            pass
                    else:
                        result.distractor_rejected += 1

        result.duration_seconds = time.time() - start_time

        return result

    def _create_fresh_triple(self, triple: Triple) -> Triple:
        """Erstellt eine frische Kopie eines Triples."""
        return Triple(
            subject=Entity(
                name=triple.subject.name,
                entity_type=triple.subject.entity_type,
                source_document=triple.subject.source_document,
            ),
            predicate=triple.predicate,
            object=Entity(
                name=triple.object.name,
                entity_type=triple.object.entity_type,
                source_document=triple.object.source_document,
            ),
            source_text=triple.source_text,
            source_document_id=triple.source_document_id,
            extraction_confidence=triple.extraction_confidence,
        )

    def _print_summary(self, result: HybridEvaluationResult):
        """Druckt Zusammenfassung der Ergebnisse."""
        print("\n" + "=" * 70)
        print("HYBRID EVALUATION SUMMARY")
        print("=" * 70)

        # Teil A
        ctrl = result.controlled
        if ctrl.total_mutations > 0:
            print("\nTEIL A: KONTROLLIERTES EXPERIMENT")
            print("-" * 40)
            print(f"Total Mutations: {ctrl.total_mutations}")
            print(f"\nOverall Metrics:")
            print(f"  Precision: {ctrl.overall.precision:.1%}")
            print(f"  Recall:    {ctrl.overall.recall:.1%}")
            print(f"  F1:        {ctrl.overall.f1:.1%}")
            print(f"  Accuracy:  {ctrl.overall.accuracy:.1%}")

            print(f"\nBy Category:")
            for cat, cm in ctrl.by_category.items():
                if cm.tp + cm.tn + cm.fp + cm.fn > 0:
                    print(f"  {cat}:")
                    print(f"    P={cm.precision:.1%} R={cm.recall:.1%} F1={cm.f1:.1%}")

        # Teil B
        real = result.realistic
        if real.update_total > 0:
            print("\nTEIL B: REALISTISCHES EXPERIMENT")
            print("-" * 40)
            print(f"Seed Triples:     {real.seed_accepted}")
            print(f"Update Triples:   {real.update_total}")
            print(f"  - Supporting:   {real.update_from_supporting}")
            print(f"  - Distractor:   {real.update_from_distractor}")

            print(f"\nDecisions:")
            print(f"  Supporting Accepted:  {real.supporting_accepted}")
            print(f"  Supporting Rejected:  {real.supporting_rejected}")
            print(f"  Distractor Accepted:  {real.distractor_accepted}")
            print(f"  Distractor Rejected:  {real.distractor_rejected}")

            print(f"\nKey Metrics:")
            print(f"  Signal Preservation: {real.signal_preservation_rate:.1%}")
            print(f"  Noise Rejection:     {real.noise_rejection_rate:.1%}")
            print(f"  Information Prec:    {real.information_precision:.1%}")
            print(f"  Information F1:      {real.information_f1:.1%}")

        print("=" * 70)


def generate_latex_tables(result: HybridEvaluationResult) -> str:
    """Generiert LaTeX-Tabellen für beide Teile."""
    lines = []

    # Teil A Tabelle
    ctrl = result.controlled
    if ctrl.total_mutations > 0:
        lines.extend([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Teil A: Kontrolliertes Experiment -- Konflikterkennung}",
            r"\label{tab:controlled-experiment}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Kategorie} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{n} \\",
            r"\midrule",
        ])

        for cat, cm in ctrl.by_category.items():
            n = cm.tp + cm.tn + cm.fp + cm.fn
            if n > 0:
                lines.append(
                    f"{cat.capitalize()} & {cm.precision:.1%} & {cm.recall:.1%} & {cm.f1:.1%} & {n} \\\\"
                )

        lines.extend([
            r"\midrule",
            f"\\textbf{{Gesamt}} & {ctrl.overall.precision:.1%} & {ctrl.overall.recall:.1%} & {ctrl.overall.f1:.1%} & {ctrl.total_mutations} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])

    # Teil B Tabelle
    real = result.realistic
    if real.update_total > 0:
        lines.extend([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Teil B: Realistisches Experiment -- Information Quality}",
            r"\label{tab:realistic-experiment}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"\textbf{Metrik} & \textbf{Wert} \\",
            r"\midrule",
            f"Signal Preservation Rate & {real.signal_preservation_rate:.1%} \\\\",
            f"Noise Rejection Rate & {real.noise_rejection_rate:.1%} \\\\",
            f"Information Precision & {real.information_precision:.1%} \\\\",
            f"Information F1 & {real.information_f1:.1%} \\\\",
            r"\midrule",
            f"Seed Triples & {real.seed_accepted} \\\\",
            f"Update Triples & {real.update_total} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    return "\n".join(lines)
