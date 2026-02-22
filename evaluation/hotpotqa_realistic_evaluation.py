#!/usr/bin/env python3
# evaluation/hotpotqa_realistic_evaluation.py
"""
Realistische HotpotQA Evaluation für Knowledge Graph Konsistenzprüfung.

Wissenschaftliches Design:
- Phase 1: Baseline KG aus GOLD Paragraphen
- Phase 2: Korrekte Fakten MIT source_document_id (Corroboration Test)
- Phase 3: Korrekte Fakten OHNE source_document_id (Missing Source Penalty Test)
- Phase 4: DISTRACTOR-basierte Kontradiktionen (echte falsche Fakten aus Dataset!)
- Phase 5: CROSS-QUESTION Confusion (Antworten von anderen Fragen)

Kontradiktionen werden NICHT generiert, sondern aus dem HotpotQA Dataset extrahiert:
- HotpotQA hat 10 Paragraphen pro Frage: 2 GOLD + 8 DISTRACTOR
- Distraktoren sind vom Dataset als "falsch aber ähnlich" markiert
- Das ist perfekt für realistische Konsistenzprüfung!

Autor: Masterarbeit GraphRAG Konsistenzprüfung
"""

import sys
sys.path.insert(0, '.')

import os
import json
import logging
import argparse
import random
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.entities import Entity, EntityType, Triple, ValidationStatus, Relation
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.graph.memory_repository import InMemoryGraphRepository
from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class PhaseMetrics:
    """Metriken für eine Evaluationsphase."""
    phase_name: str
    total_triples: int = 0
    accepted: int = 0
    rejected: int = 0
    conflicts_detected: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    missing_source_warnings: int = 0
    penalties_applied: int = 0
    corroborated_facts: int = 0
    processing_time_seconds: float = 0.0

    # Für Kontradiktions-Analyse
    true_positives: int = 0   # Korrekt als falsch erkannt
    false_negatives: int = 0  # Fälschlicherweise akzeptiert
    true_negatives: int = 0   # Korrekt als richtig erkannt
    false_positives: int = 0  # Fälschlicherweise abgelehnt

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total_triples if self.total_triples > 0 else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.total_triples if self.total_triples > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvaluationResults:
    """Gesamtergebnisse der Evaluation."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset: str = "hotpotqa"
    sample_size: int = 0

    phase1_baseline: Optional[PhaseMetrics] = None
    phase2_correct_with_source: Optional[PhaseMetrics] = None
    phase3_correct_without_source: Optional[PhaseMetrics] = None
    phase4_distractor_contradictions: Optional[PhaseMetrics] = None
    phase5_cross_question_confusion: Optional[PhaseMetrics] = None
    phase6_fake_source_attack: Optional[PhaseMetrics] = None  # NEU: Source Verification

    # Vergleichsmetriken
    missing_source_penalty_effectiveness: float = 0.0
    contradiction_detection_f1: float = 0.0
    cross_question_detection_f1: float = 0.0
    fake_source_detection_f1: float = 0.0  # NEU

    config: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TRIPLE EXTRAKTION MIT GOLD/DISTRACTOR UNTERSCHEIDUNG
# =============================================================================

class RealisticTripleExtractor:
    """
    Extrahiert Triples aus HotpotQA mit Gold/Distractor Unterscheidung.

    Schlüsselidee:
    - GOLD Paragraphen (in supporting_facts) → korrekte Fakten
    - DISTRACTOR Paragraphen (nicht in supporting_facts) → falsche Fakten
    """

    def __init__(self):
        self.entity_cache: Dict[str, Entity] = {}
        self._gold_titles_cache: Dict[str, Set[str]] = {}

    def _get_gold_titles(self, example: QAExample) -> Set[str]:
        """Extrahiert die Titel der GOLD Paragraphen."""
        cache_key = example.id
        if cache_key not in self._gold_titles_cache:
            gold_titles = set(sf.title for sf in example.supporting_facts)
            self._gold_titles_cache[cache_key] = gold_titles
        return self._gold_titles_cache[cache_key]

    def _infer_entity_type(self, name: str) -> EntityType:
        """Heuristik für Entity-Typ."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["city", "country", "river", "state", "mountain", "lake"]):
            return EntityType.LOCATION
        if any(kw in name_lower for kw in ["university", "company", "inc", "ltd", "school", "band", "team"]):
            return EntityType.ORGANIZATION
        if any(kw in name_lower for kw in ["war", "battle", "championship", "election", "festival", "award"]):
            return EntityType.EVENT
        if any(kw in name_lower for kw in ["film", "movie", "album", "song", "book", "novel", "series"]):
            return EntityType.CONCEPT
        return EntityType.PERSON

    def _get_or_create_entity(self, name: str, source_doc: Optional[str] = None) -> Entity:
        """Holt oder erstellt eine Entity."""
        cache_key = name.lower().strip()
        if cache_key not in self.entity_cache:
            entity = Entity(
                name=name.strip(),
                entity_type=self._infer_entity_type(name),
                source_document=source_doc
            )
            self.entity_cache[cache_key] = entity
        return self.entity_cache[cache_key]

    def clear_cache(self):
        """Leert den Entity-Cache für neue Phase."""
        self.entity_cache.clear()

    # =========================================================================
    # PHASE 1: Baseline aus GOLD Paragraphen
    # =========================================================================

    def extract_baseline_triples(
        self,
        example: QAExample,
        with_source: bool = True
    ) -> List[Triple]:
        """
        Phase 1: Extrahiert Baseline-Triples nur aus GOLD Paragraphen.

        Nur Paragraphen die in supporting_facts referenziert werden.
        """
        triples = []
        source_id = example.id if with_source else None
        gold_titles = self._get_gold_titles(example)

        # Erstelle Entitäten aus GOLD Paragraphen
        gold_entities = []
        for para in example.context_paragraphs:
            title = para.get("title", "").strip()
            if title and title in gold_titles:
                entity = self._get_or_create_entity(title, source_id)
                gold_entities.append(entity)

        # Verknüpfe GOLD Entitäten (die in supporting_facts sind)
        for i in range(len(gold_entities) - 1):
            triple = Triple(
                subject=gold_entities[i],
                predicate="RELATED_TO",
                object=gold_entities[i + 1],
                source_text=f"Gold context for: {example.question[:50]}",
                source_document_id=source_id,
                extraction_confidence=0.85
            )
            triples.append(triple)

        # Verknüpfe mit Antwort
        if example.answer and len(example.answer) > 2 and gold_entities:
            answer_entity = self._get_or_create_entity(example.answer, source_id)
            triple = Triple(
                subject=gold_entities[0],
                predicate="HAS_ANSWER",
                object=answer_entity,
                source_text=example.question,
                source_document_id=source_id,
                extraction_confidence=0.9
            )
            triples.append(triple)

        return triples

    # =========================================================================
    # PHASE 2/3: Supporting Facts (korrekte Fakten)
    # =========================================================================

    def extract_supporting_fact_triples(
        self,
        example: QAExample,
        with_source: bool = True
    ) -> List[Triple]:
        """
        Phase 2/3: Extrahiert Triples aus Supporting Facts.

        Diese sollten AKZEPTIERT werden (korrekte Fakten).
        """
        triples = []
        source_id = example.id if with_source else None

        sf_titles = list(set(sf.title for sf in example.supporting_facts if sf.title))
        if len(sf_titles) < 2:
            return triples

        # Verknüpfe Supporting Facts
        for i in range(len(sf_titles) - 1):
            subj = self._get_or_create_entity(sf_titles[i], source_id)
            obj = self._get_or_create_entity(sf_titles[i + 1], source_id)

            triple = Triple(
                subject=subj,
                predicate="SUPPORTS",
                object=obj,
                source_text=example.question,
                source_document_id=source_id,
                extraction_confidence=0.85
            )
            triples.append(triple)

        # Verknüpfe mit Antwort
        if example.answer and len(example.answer) > 2:
            answer_entity = self._get_or_create_entity(example.answer, source_id)
            first_sf = self._get_or_create_entity(sf_titles[0], source_id)

            triple = Triple(
                subject=first_sf,
                predicate="ANSWERS",
                object=answer_entity,
                source_text=example.question,
                source_document_id=source_id,
                extraction_confidence=0.9
            )
            triples.append(triple)

        return triples

    # =========================================================================
    # PHASE 4: KARDINALITÄTS-VERLETZUNGEN (testbar!)
    # =========================================================================

    def extract_cardinality_violation_triples(
        self,
        example: QAExample,
        with_source: bool = True
    ) -> List[Triple]:
        """
        Phase 4: Erzeugt Kardinalitätsverletzungen.

        SCHLÜSSELIDEE für testbare Kontradiktionen:
        - In Phase 1/2 wurde etabliert: "GoldEntity HAS_ANSWER Antwort"
        - Jetzt versuchen wir: "GoldEntity HAS_ANSWER FalscheAntwort"
        - Da HAS_ANSWER Kardinalität=1 hat, ist das ein KONFLIKT!

        Das ist eine realistische Kontradiktion weil:
        - Es nutzt die GLEICHE Gold-Entity (aus supporting_facts)
        - Es nutzt die GLEICHE Relation (HAS_ANSWER)
        - Aber eine ANDERE Antwort (aus Distraktoren)

        Das Konsistenzmodul MUSS das erkennen!
        """
        triples = []
        source_id = example.id if with_source else None
        gold_titles = self._get_gold_titles(example)

        if not example.supporting_facts or not example.answer:
            return triples

        # Finde DISTRACTOR Paragraphen für falsche Antworten
        distractor_titles = []
        for para in example.context_paragraphs:
            title = para.get("title", "").strip()
            if title and title not in gold_titles:
                distractor_titles.append(title)

        if not distractor_titles:
            return triples

        # WICHTIG: Die GLEICHE Gold-Entity wie in Phase 1 verwenden!
        # Phase 1 iteriert über context_paragraphs, nicht supporting_facts
        # Die Reihenfolge kann unterschiedlich sein!
        gold_entities_from_context = []
        for para in example.context_paragraphs:
            title = para.get("title", "").strip()
            if title and title in gold_titles:
                gold_entities_from_context.append(title)

        if not gold_entities_from_context:
            return triples

        # Erste Gold-Entity aus context_paragraphs (wie in extract_baseline_triples)
        gold_title = gold_entities_from_context[0]
        gold_entity = self._get_or_create_entity(gold_title, source_id)

        # Versuche FALSCHE Antworten (Distraktoren) mit HAS_ANSWER zu verknüpfen
        for dist_title in distractor_titles[:2]:  # Max 2 pro Beispiel
            # Der Distraktor-Titel als "falsche Antwort"
            wrong_answer_entity = self._get_or_create_entity(dist_title, source_id)

            # GLEICHE Subject + GLEICHE Relation + ANDERES Object = KARDINALITÄTSKONFLIKT!
            triple = Triple(
                subject=gold_entity,  # GLEICHE Entity wie in Phase 1
                predicate="HAS_ANSWER",  # GLEICHE Relation (Kardinalität=1)
                object=wrong_answer_entity,  # ANDERE Antwort → KONFLIKT!
                source_text=f"[CARDINALITY VIOLATION] {example.question}",
                source_document_id=source_id,
                extraction_confidence=0.8
            )
            triples.append(triple)

        return triples

    # =========================================================================
    # PHASE 5: Cross-Question Kardinalitätsverletzungen
    # =========================================================================

    def extract_cross_question_violation_triples(
        self,
        example: QAExample,
        other_examples: List[QAExample],
        with_source: bool = True
    ) -> List[Triple]:
        """
        Phase 5: Kardinalitätsverletzung mit Antworten von anderen Fragen.

        SCHLÜSSELIDEE:
        - In Phase 1/2 wurde etabliert: "GoldEntity HAS_ANSWER KorrekteAntwort"
        - Jetzt: "GoldEntity HAS_ANSWER AntwortVonAndererFrage"
        - Das ist ein KARDINALITÄTSKONFLIKT!

        Beispiel:
        - Frage 1: "Wer führte Regie bei Inception?" → "Christopher Nolan"
        - Gold: "Inception HAS_ANSWER Christopher Nolan" (etabliert in Phase 1)

        - Frage 2: "Wer schrieb Harry Potter?" → "J.K. Rowling"

        Cross-Confusion:
        - "Inception HAS_ANSWER J.K. Rowling" → KARDINALITÄTSKONFLIKT!
        """
        triples = []
        source_id = example.id if with_source else None

        if not example.supporting_facts or not other_examples:
            return triples

        # Wähle eine zufällige andere Antwort
        other_answers = [
            ex.answer for ex in other_examples
            if ex.answer and ex.answer != example.answer and len(ex.answer) > 2
        ]

        if not other_answers:
            return triples

        wrong_answer = random.choice(other_answers)
        wrong_answer_entity = self._get_or_create_entity(wrong_answer, source_id)

        # WICHTIG: Die GLEICHE Gold-Entity wie in Phase 1 verwenden!
        # Phase 1 iteriert über context_paragraphs, nicht supporting_facts
        gold_titles = self._get_gold_titles(example)
        gold_entities_from_context = []
        for para in example.context_paragraphs:
            title = para.get("title", "").strip()
            if title and title in gold_titles:
                gold_entities_from_context.append(title)

        if not gold_entities_from_context:
            return triples

        gold_title = gold_entities_from_context[0]
        gold_entity = self._get_or_create_entity(gold_title, source_id)

        # KARDINALITÄTSKONFLIKT: Gleiche Entity + Gleiche Relation + Andere Antwort
        triple = Triple(
            subject=gold_entity,  # GLEICHE Entity
            predicate="HAS_ANSWER",  # GLEICHE Relation (Kardinalität=1)
            object=wrong_answer_entity,  # ANDERE Antwort → KONFLIKT!
            source_text=f"[CROSS-QUESTION VIOLATION] {example.question}",
            source_document_id=source_id,
            extraction_confidence=0.8
        )
        triples.append(triple)

        return triples

    # =========================================================================
    # PHASE 6: FAKE SOURCE ATTACK - Quelle unterstützt Claim nicht
    # =========================================================================

    def extract_fake_source_triples(
        self,
        example: QAExample,
        other_examples: List[QAExample],
        with_source: bool = True
    ) -> List[Triple]:
        """
        Phase 6: Triples mit gefälschter Quelle die den Claim nicht belegt.

        SZENARIO (Angriff):
        - Jemand erstellt ein Triple mit korrektem source_document_id
        - Aber der source_text ist von einem ANDEREN Thema
        - Das System sollte erkennen dass die Quelle den Claim nicht unterstützt

        Beispiel:
        - Claim: "Berlin ist die Hauptstadt von Deutschland"
        - source_document_id: "doc_123" (existiert)
        - source_text: "Der Mount Everest ist 8848m hoch" (thematisch irrelevant!)

        Das Source Verification sollte die niedrige Similarity erkennen!
        """
        triples = []
        source_id = example.id if with_source else None
        gold_titles = self._get_gold_titles(example)

        if not example.supporting_facts or not example.answer:
            return triples

        # Finde einen irrelevanten Text von einem anderen Beispiel
        irrelevant_texts = []
        for other in other_examples:
            for para in other.context_paragraphs:
                text = " ".join(para.get("sentences", []))
                if text and len(text) > 50:
                    # Prüfe dass der Text nichts mit unserem Beispiel zu tun hat
                    if (example.answer.lower() not in text.lower() and
                        not any(sf.title.lower() in text.lower() for sf in example.supporting_facts)):
                        irrelevant_texts.append(text)

        if not irrelevant_texts:
            return triples

        # Wähle einen zufälligen irrelevanten Text
        fake_source_text = random.choice(irrelevant_texts)[:500]

        # Hole Gold-Entity
        gold_entities_from_context = []
        for para in example.context_paragraphs:
            title = para.get("title", "").strip()
            if title and title in gold_titles:
                gold_entities_from_context.append(title)

        if not gold_entities_from_context:
            return triples

        gold_title = gold_entities_from_context[0]
        gold_entity = self._get_or_create_entity(gold_title, source_id)
        answer_entity = self._get_or_create_entity(example.answer, source_id)

        # Erstelle Triple mit KORREKTEM Claim aber FALSCHEM source_text
        triple = Triple(
            subject=gold_entity,
            predicate="CONFIRMS",  # Neue Relation um Kardinalitätskonflikte zu vermeiden
            object=answer_entity,
            # HIER IST DER ANGRIFF: source_text passt nicht zum Claim!
            source_text=fake_source_text,
            source_document_id=source_id,  # Hat eine Quelle (kein Missing Source Penalty)
            extraction_confidence=0.9
        )
        triples.append(triple)

        return triples


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class RealisticHotpotQAEvaluator:
    """Führt die realistische 5-Phasen Evaluation durch."""

    def __init__(
        self,
        use_gpu: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        missing_source_penalty: float = 0.7,
        verbose: bool = True
    ):
        self.use_gpu = use_gpu
        self.embedding_model_name = embedding_model_name
        self.missing_source_penalty = missing_source_penalty
        self.verbose = verbose

        self.extractor = RealisticTripleExtractor()
        self.results = EvaluationResults()

        self.embedding_model = None
        self.orchestrator = None
        self.graph_repo = None

    def setup(self, llm_model: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434"):
        """Initialisiert Modelle und Orchestrator."""
        logger.info("=== SETUP ===")

        # 1. Embedding-Modell
        logger.info(f"Lade Embedding-Modell: {self.embedding_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if self.use_gpu else "cpu"
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=device
            )
            logger.info(f"  ✓ Modell geladen auf {device}")

            if self.use_gpu:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("  ⚠ GPU nicht verfügbar, nutze CPU")
        except Exception as e:
            logger.warning(f"  ⚠ Kein Embedding-Modell: {e}")
            self.embedding_model = None

        # 2. LLM-Client (Ollama/Llama)
        llm_client = None
        if llm_model:
            try:
                from src.llm.ollama_client import OllamaClient
                llm_client = OllamaClient(model=llm_model, base_url=ollama_url)
                logger.info(f"  ✓ LLM-Client: {llm_model} @ {ollama_url}")
            except Exception as e:
                logger.warning(f"  ⚠ Kein LLM-Client (Ollama nicht erreichbar): {e}")
                logger.warning(f"    → Starte Ollama mit: ollama serve")
                llm_client = None
        else:
            logger.info("  ○ LLM deaktiviert (--no-llm)")

        # 3. Graph Repository
        self.graph_repo = InMemoryGraphRepository()
        logger.info("  ✓ Graph Repository initialisiert")

        # 4. Consistency Orchestrator
        config = ConsistencyConfig(
            valid_relation_types=["RELATED_TO", "SUPPORTS", "ANSWERS", "HAS_ANSWER", "CONFIRMS"],
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.7,
            enable_missing_source_penalty=True,
            missing_source_penalty=self.missing_source_penalty,
            enable_provenance_boost=True,
            # Source Verification: Prüft ob source_text den Claim unterstützt
            enable_source_verification=True,
            # LLM-Modell für Stufe 3 (muss mit Ollama-Modell übereinstimmen!)
            llm_model=llm_model if llm_model else "llama3.1:8b",
            # Kardinalitätsregeln: Format {"RELATION": {"max": n}}
            # ANSWERS/HAS_ANSWER darf nur einmal pro Subject vorkommen!
            cardinality_rules={
                "ANSWERS": {"max": 1},
                "HAS_ANSWER": {"max": 1},
            },
        )

        self.orchestrator = ConsistencyOrchestrator(
            config=config,
            graph_repo=self.graph_repo,
            embedding_model=self.embedding_model,
            llm_client=llm_client,  # Ollama/Llama wenn verfügbar
            enable_metrics=True
        )
        logger.info("  ✓ Orchestrator initialisiert")
        if llm_client:
            logger.info(f"    → LLM-Arbitration aktiviert mit {llm_model}")

        self.results.config = {
            "embedding_model": self.embedding_model_name,
            "use_gpu": self.use_gpu,
            "missing_source_penalty": self.missing_source_penalty,
            "cardinality_rules": {"ANSWERS": 1, "HAS_ANSWER": 1},
            "source_verification_enabled": True,
        }

    def _process_triples_batch(
        self,
        triples: List[Triple],
        phase_name: str,
        expect_rejection: bool = False,
        persist_to_graph: bool = False
    ) -> PhaseMetrics:
        """
        Verarbeitet Triples und sammelt Metriken.

        Args:
            triples: Zu verarbeitende Triples
            phase_name: Name der Phase
            expect_rejection: Wenn True, sind Rejections "korrekt" (für Kontradiktionen)
            persist_to_graph: Wenn True, werden akzeptierte Triples im Graph gespeichert
                             (wichtig für Kardinalitätsprüfung in späteren Phasen!)
        """
        import time
        start_time = time.time()

        metrics = PhaseMetrics(phase_name=phase_name)
        confidences = []

        for triple in triples:
            try:
                result = self.orchestrator.process(triple)
                metrics.total_triples += 1

                is_rejected = result.validation_status == ValidationStatus.REJECTED
                is_accepted = result.validation_status == ValidationStatus.ACCEPTED
                is_needs_review = result.validation_status == ValidationStatus.NEEDS_REVIEW

                # Für Kontradiktionserkennung: NEEDS_REVIEW zählt als "Konflikt erkannt"
                # weil das System den Konflikt gefunden hat und zur LLM-Prüfung eskaliert
                is_conflict_detected = is_rejected or (expect_rejection and is_needs_review)

                if is_accepted:
                    metrics.accepted += 1
                    if expect_rejection:
                        metrics.false_negatives += 1  # Hätte abgelehnt werden sollen
                    else:
                        metrics.true_negatives += 1  # Korrekt akzeptiert

                    # WICHTIG: Persistiere akzeptierte Triples im Graph!
                    # Das ermöglicht Kardinalitätsprüfung in späteren Phasen
                    if persist_to_graph and self.graph_repo is not None:
                        try:
                            # Stelle sicher dass Entities existieren
                            self.graph_repo.create_entity(triple.subject)
                            self.graph_repo.create_entity(triple.object)

                            # Erstelle die Relation
                            from src.models.entities import Relation
                            relation = Relation(
                                source_id=triple.subject.id,
                                target_id=triple.object.id,
                                relation_type=triple.predicate,
                                source_document_id=triple.source_document_id,
                                confidence=result.validation_history[-1].get("confidence", 0.5) if result.validation_history else 0.5
                            )
                            self.graph_repo.create_relation(relation)
                        except Exception as persist_error:
                            logger.debug(f"Persist-Fehler: {persist_error}")

                elif is_rejected:
                    metrics.rejected += 1
                    if expect_rejection:
                        metrics.true_positives += 1  # Korrekt abgelehnt
                    else:
                        metrics.false_positives += 1  # Fälschlicherweise abgelehnt

                elif is_needs_review:
                    # NEEDS_REVIEW = Konflikt erkannt, zur manuellen Prüfung eskaliert
                    if expect_rejection:
                        # Für Kontradiktionen: NEEDS_REVIEW = Konflikt korrekt erkannt!
                        metrics.true_positives += 1
                        metrics.conflicts_detected += 1
                    else:
                        # Für korrekte Fakten: NEEDS_REVIEW = false positive
                        metrics.false_positives += 1

                # Analysiere validation_history
                for h in result.validation_history:
                    # Unterstütze sowohl Dicts als auch LLMResolution Objekte
                    if isinstance(h, dict):
                        details = h.get("details", {})
                        validator = h.get("validator", "")
                        passed = h.get("passed", True)
                    else:
                        # LLMResolution oder ähnliches Objekt
                        details = getattr(h, "details", {}) if hasattr(h, "details") else {}
                        validator = getattr(h, "validator", "") if hasattr(h, "validator") else ""
                        passed = getattr(h, "passed", True) if hasattr(h, "passed") else True

                    # Auch details kann ein Objekt sein
                    if not isinstance(details, dict):
                        details = {}

                    if details.get("missing_source_warning"):
                        metrics.missing_source_warnings += 1
                        metrics.penalties_applied += 1

                    if "adjusted_confidence" in details:
                        conf = details["adjusted_confidence"]
                        confidences.append(conf)
                        metrics.min_confidence = min(metrics.min_confidence, conf)
                        metrics.max_confidence = max(metrics.max_confidence, conf)

                    if details.get("corroboration_bonus", 0) > 0:
                        metrics.corroborated_facts += 1

                    # Konflikt-Zählung
                    if validator in ["cardinality", "embedding_based"]:
                        if not passed:
                            metrics.conflicts_detected += 1

            except Exception as e:
                logger.debug(f"Fehler bei Triple-Verarbeitung: {e}")

        if confidences:
            metrics.avg_confidence = sum(confidences) / len(confidences)

        metrics.processing_time_seconds = time.time() - start_time
        return metrics

    def run_evaluation(self, examples: List[QAExample]) -> EvaluationResults:
        """Führt die vollständige 5-Phasen Evaluation durch."""
        self.results.sample_size = len(examples)

        logger.info(f"\n{'='*80}")
        logger.info(f"STARTE REALISTISCHE EVALUATION MIT {len(examples)} BEISPIELEN")
        logger.info(f"{'='*80}")

        # =====================================================================
        # PHASE 1: BASELINE KNOWLEDGE GRAPH (nur GOLD)
        # =====================================================================
        logger.info("\n--- PHASE 1: Baseline KG aus GOLD Paragraphen ---")

        baseline_triples = []
        for example in examples:
            triples = self.extractor.extract_baseline_triples(example, with_source=True)
            baseline_triples.extend(triples)

        logger.info(f"Extrahiert: {len(baseline_triples)} Baseline-Triples (nur GOLD)")

        # WICHTIG: persist_to_graph=True damit Kardinalitätsprüfung später funktioniert!
        self.results.phase1_baseline = self._process_triples_batch(
            baseline_triples, "Phase 1: Baseline (GOLD)", expect_rejection=False,
            persist_to_graph=True  # Speichere im Graph für spätere Konflikte
        )
        self._print_phase_summary(self.results.phase1_baseline)

        # Log: Wie viele Relationen sind jetzt im Graph?
        if self.graph_repo:
            try:
                all_entities = self.graph_repo.find_all_entities()
                logger.info(f"  → Graph enthält jetzt {len(all_entities)} Entities")
            except Exception:
                pass

        # =====================================================================
        # PHASE 2: KORREKTE FAKTEN MIT QUELLE
        # =====================================================================
        logger.info("\n--- PHASE 2: Korrekte Fakten MIT Quelle ---")

        correct_triples_with_source = []
        for example in examples:
            triples = self.extractor.extract_supporting_fact_triples(example, with_source=True)
            correct_triples_with_source.extend(triples)

        logger.info(f"Extrahiert: {len(correct_triples_with_source)} korrekte Triples")

        self.results.phase2_correct_with_source = self._process_triples_batch(
            correct_triples_with_source, "Phase 2: Korrekt MIT Quelle", expect_rejection=False
        )
        self._print_phase_summary(self.results.phase2_correct_with_source)

        # =====================================================================
        # PHASE 3: KORREKTE FAKTEN OHNE QUELLE (Missing Source Penalty Test)
        # =====================================================================
        logger.info("\n--- PHASE 3: Korrekte Fakten OHNE Quelle ---")

        # WICHTIG: Cache speichern vor Phase 3!
        # Phase 3 braucht neue Entities ohne source_document_id
        # Aber Phase 4/5 müssen dieselben Entity-IDs wie Phase 1 verwenden
        saved_cache = dict(self.extractor.entity_cache)
        self.extractor.clear_cache()

        correct_triples_without_source = []
        for example in examples:
            triples = self.extractor.extract_supporting_fact_triples(example, with_source=False)
            correct_triples_without_source.extend(triples)

        logger.info(f"Extrahiert: {len(correct_triples_without_source)} Triples (ohne Quelle)")

        self.results.phase3_correct_without_source = self._process_triples_batch(
            correct_triples_without_source, "Phase 3: Korrekt OHNE Quelle", expect_rejection=False
        )
        self._print_phase_summary(self.results.phase3_correct_without_source)

        # =====================================================================
        # PHASE 4: KARDINALITÄTSVERLETZUNGEN (Distraktoren)
        # =====================================================================
        logger.info("\n--- PHASE 4: Kardinalitätsverletzungen (Distraktor-Antworten) ---")
        logger.info("  (Nutzt GLEICHE Gold-Entity + GLEICHE Relation + ANDERE Antwort)")
        logger.info("  → Sollten wegen Kardinalität=1 ABGELEHNT werden!")

        # WICHTIG: Cache von Phase 1 wiederherstellen!
        # Phase 3 hatte den Cache geleert, aber wir brauchen dieselben Entity-IDs wie Phase 1
        self.extractor.entity_cache = saved_cache
        logger.info(f"  → Entity-Cache wiederhergestellt ({len(saved_cache)} Entities)")

        cardinality_triples = []
        for example in examples:
            triples = self.extractor.extract_cardinality_violation_triples(example, with_source=True)
            cardinality_triples.extend(triples)

        logger.info(f"Extrahiert: {len(cardinality_triples)} Kardinalitäts-Violations")

        self.results.phase4_distractor_contradictions = self._process_triples_batch(
            cardinality_triples, "Phase 4: Kardinalitäts-Violations", expect_rejection=True
        )
        self._print_phase_summary(self.results.phase4_distractor_contradictions, is_contradiction=True)

        # =====================================================================
        # PHASE 5: CROSS-QUESTION KARDINALITÄTSVERLETZUNGEN
        # =====================================================================
        logger.info("\n--- PHASE 5: Cross-Question Kardinalitätsverletzungen ---")
        logger.info("  (Gold-Entity + HAS_ANSWER + Antwort von anderer Frage)")
        logger.info("  → Sollten wegen Kardinalität=1 ABGELEHNT werden!")

        cross_triples = []
        for i, example in enumerate(examples):
            other_examples = examples[:i] + examples[i+1:]
            triples = self.extractor.extract_cross_question_violation_triples(
                example, other_examples, with_source=True
            )
            cross_triples.extend(triples)

        logger.info(f"Extrahiert: {len(cross_triples)} Cross-Question-Violations")

        self.results.phase5_cross_question_confusion = self._process_triples_batch(
            cross_triples, "Phase 5: Cross-Question Violations", expect_rejection=True
        )
        self._print_phase_summary(self.results.phase5_cross_question_confusion, is_contradiction=True)

        # =====================================================================
        # PHASE 6: FAKE SOURCE ATTACK (Source Verification Test)
        # =====================================================================
        logger.info("\n--- PHASE 6: Fake Source Attack (Source Verification) ---")
        logger.info("  (Claim ist korrekt, aber source_text passt nicht zum Claim)")
        logger.info("  → Source Verification sollte die Diskrepanz erkennen!")

        fake_source_triples = []
        for i, example in enumerate(examples):
            other_examples = examples[:i] + examples[i+1:]
            triples = self.extractor.extract_fake_source_triples(
                example, other_examples, with_source=True
            )
            fake_source_triples.extend(triples)

        logger.info(f"Extrahiert: {len(fake_source_triples)} Fake-Source-Triples")

        # Diese Triples sollten niedrigere Konfidenz bekommen wegen Source Verification
        # Sie werden nicht rejected (Claim ist korrekt), aber die Konfidenz sinkt
        self.results.phase6_fake_source_attack = self._process_triples_batch(
            fake_source_triples, "Phase 6: Fake Source Attack", expect_rejection=True
        )
        self._print_phase_summary(self.results.phase6_fake_source_attack, is_contradiction=True)

        # =====================================================================
        # VERGLEICHSMETRIKEN
        # =====================================================================
        self._calculate_comparison_metrics()

        return self.results

    def _calculate_comparison_metrics(self):
        """Berechnet Vergleichsmetriken."""
        p2 = self.results.phase2_correct_with_source
        p3 = self.results.phase3_correct_without_source
        p4 = self.results.phase4_distractor_contradictions
        p5 = self.results.phase5_cross_question_confusion
        p6 = self.results.phase6_fake_source_attack

        # Missing Source Penalty Effectiveness
        if p2 and p3 and p2.avg_confidence > 0:
            self.results.missing_source_penalty_effectiveness = (
                (p2.avg_confidence - p3.avg_confidence) / p2.avg_confidence
            )

        # Contradiction Detection F1
        if p4:
            self.results.contradiction_detection_f1 = p4.f1_score

        # Cross-Question Detection F1
        if p5:
            self.results.cross_question_detection_f1 = p5.f1_score

        # Fake Source Detection
        # Hier messen wir wie stark die Konfidenz sinkt wenn source_text nicht passt
        if p2 and p6 and p2.avg_confidence > 0:
            self.results.fake_source_detection_f1 = (
                (p2.avg_confidence - p6.avg_confidence) / p2.avg_confidence
            )

    def _print_phase_summary(self, metrics: PhaseMetrics, is_contradiction: bool = False):
        """Gibt Phase-Zusammenfassung aus."""
        if not self.verbose:
            return

        if is_contradiction:
            print(f"""
  Triples: {metrics.total_triples}
  Abgelehnt (korrekt): {metrics.true_positives} / {metrics.total_triples}
  Akzeptiert (FEHLER): {metrics.false_negatives}
  Ø Konfidenz: {metrics.avg_confidence:.2%}
  Konflikte erkannt: {metrics.conflicts_detected}
  → Rejection Rate: {metrics.rejection_rate:.1%}
  → F1 Score: {metrics.f1_score:.2%}
  Zeit: {metrics.processing_time_seconds:.2f}s
            """)
        else:
            print(f"""
  Triples: {metrics.total_triples}
  Akzeptiert: {metrics.accepted} ({metrics.acceptance_rate:.1%})
  Abgelehnt: {metrics.rejected} ({metrics.rejection_rate:.1%})
  Ø Konfidenz: {metrics.avg_confidence:.2%}
  Missing Source Warnings: {metrics.missing_source_warnings}
  Zeit: {metrics.processing_time_seconds:.2f}s
            """)

    def print_final_report(self):
        """Gibt den finalen Bericht aus."""
        r = self.results

        print("\n" + "="*80)
        print("FINALER EVALUATIONSBERICHT - REALISTISCHE HOTPOTQA EVALUATION")
        print("="*80)

        print(f"""
Dataset: {r.dataset}
Beispiele: {r.sample_size}
Timestamp: {r.timestamp}
        """)

        print("-"*80)
        print("PHASENVERGLEICH")
        print("-"*80)

        # Header
        print(f"\n{'Phase':<35} {'Triples':>8} {'Accept':>8} {'Reject':>8} {'Ø Conf':>8} {'F1':>8}")
        print("-"*80)

        phases = [
            ("Phase 1: Baseline (GOLD)", r.phase1_baseline, False),
            ("Phase 2: Korrekt MIT Quelle", r.phase2_correct_with_source, False),
            ("Phase 3: Korrekt OHNE Quelle", r.phase3_correct_without_source, False),
            ("Phase 4: Distractor-Kontradiktionen", r.phase4_distractor_contradictions, True),
            ("Phase 5: Cross-Question Confusion", r.phase5_cross_question_confusion, True),
            ("Phase 6: Fake Source Attack", r.phase6_fake_source_attack, True),
        ]

        for name, metrics, is_contradiction in phases:
            if metrics:
                f1 = f"{metrics.f1_score:.1%}" if is_contradiction else "-"
                print(f"{name:<35} {metrics.total_triples:>8} {metrics.accepted:>8} {metrics.rejected:>8} {metrics.avg_confidence:>7.1%} {f1:>8}")

        print("\n" + "-"*80)
        print("KERNMETRIKEN")
        print("-"*80)

        print(f"""
Missing Source Penalty Effectiveness: {r.missing_source_penalty_effectiveness:.1%}
  (Konfidenz-Reduktion ohne Quelle)

Distractor Contradiction Detection:
  - F1 Score: {r.contradiction_detection_f1:.1%}
  - True Positives: {r.phase4_distractor_contradictions.true_positives if r.phase4_distractor_contradictions else 0}
  - False Negatives: {r.phase4_distractor_contradictions.false_negatives if r.phase4_distractor_contradictions else 0}

Cross-Question Confusion Detection:
  - F1 Score: {r.cross_question_detection_f1:.1%}
  - True Positives: {r.phase5_cross_question_confusion.true_positives if r.phase5_cross_question_confusion else 0}
  - False Negatives: {r.phase5_cross_question_confusion.false_negatives if r.phase5_cross_question_confusion else 0}

Fake Source Attack Detection (Source Verification):
  - Konfidenz-Reduktion: {r.fake_source_detection_f1:.1%}
  - Avg Confidence (mit Fake Source): {r.phase6_fake_source_attack.avg_confidence if r.phase6_fake_source_attack else 0:.1%}
  - Rejected: {r.phase6_fake_source_attack.rejected if r.phase6_fake_source_attack else 0}
        """)

        print("-"*80)
        print("INTERPRETATION")
        print("-"*80)

        # Missing Source Penalty
        if r.missing_source_penalty_effectiveness > 0.2:
            print("✓ Missing Source Penalty funktioniert effektiv (>20% Reduktion)")
        elif r.missing_source_penalty_effectiveness > 0.1:
            print("○ Missing Source Penalty zeigt moderate Wirkung (10-20%)")
        else:
            print("⚠ Missing Source Penalty zeigt schwache Wirkung (<10%)")

        # Distractor Detection
        if r.contradiction_detection_f1 > 0.5:
            print("✓ Gute Distractor-Kontradiktionserkennung (F1 > 50%)")
        elif r.contradiction_detection_f1 > 0.2:
            print("○ Moderate Distractor-Erkennung (F1 20-50%)")
        else:
            print("⚠ Schwache Distractor-Erkennung (F1 < 20%)")

        # Cross-Question Detection
        if r.cross_question_detection_f1 > 0.5:
            print("✓ Gute Cross-Question-Erkennung (F1 > 50%)")
        elif r.cross_question_detection_f1 > 0.2:
            print("○ Moderate Cross-Question-Erkennung (F1 20-50%)")
        else:
            print("⚠ Schwache Cross-Question-Erkennung (F1 < 20%)")

        # Fake Source Detection (Source Verification)
        if r.fake_source_detection_f1 > 0.3:
            print("✓ Gute Fake-Source-Erkennung (>30% Konfidenz-Reduktion)")
        elif r.fake_source_detection_f1 > 0.1:
            print("○ Moderate Fake-Source-Erkennung (10-30% Reduktion)")
        else:
            print("⚠ Schwache Fake-Source-Erkennung (<10% Reduktion)")

    def export_results(self, output_path: str):
        """Exportiert Ergebnisse als JSON."""

        def serialize_metrics(m: Optional[PhaseMetrics]) -> Optional[Dict]:
            if m is None:
                return None
            d = asdict(m)
            d["acceptance_rate"] = m.acceptance_rate
            d["rejection_rate"] = m.rejection_rate
            d["precision"] = m.precision
            d["recall"] = m.recall
            d["f1_score"] = m.f1_score
            return d

        results_dict = {
            "timestamp": self.results.timestamp,
            "dataset": self.results.dataset,
            "sample_size": self.results.sample_size,
            "config": self.results.config,
            "phases": {
                "phase1_baseline": serialize_metrics(self.results.phase1_baseline),
                "phase2_correct_with_source": serialize_metrics(self.results.phase2_correct_with_source),
                "phase3_correct_without_source": serialize_metrics(self.results.phase3_correct_without_source),
                "phase4_distractor_contradictions": serialize_metrics(self.results.phase4_distractor_contradictions),
                "phase5_cross_question_confusion": serialize_metrics(self.results.phase5_cross_question_confusion),
                "phase6_fake_source_attack": serialize_metrics(self.results.phase6_fake_source_attack),
            },
            "comparison_metrics": {
                "missing_source_penalty_effectiveness": self.results.missing_source_penalty_effectiveness,
                "contradiction_detection_f1": self.results.contradiction_detection_f1,
                "cross_question_detection_f1": self.results.cross_question_detection_f1,
                "fake_source_detection_effectiveness": self.results.fake_source_detection_f1,
            }
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Ergebnisse exportiert: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Realistische HotpotQA Evaluation für KG Konsistenzprüfung"
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Anzahl HotpotQA-Beispiele (default: 50)"
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        help="HotpotQA Split (default: validation)"
    )
    parser.add_argument(
        "--gpu", action="store_true", default=True,
        help="GPU nutzen (default: True)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="GPU deaktivieren"
    )
    parser.add_argument(
        "--penalty", type=float, default=0.7,
        help="Missing Source Penalty (default: 0.7)"
    )
    parser.add_argument(
        "--output", type=str, default="results/hotpotqa_realistic_evaluation.json",
        help="Output-Pfad"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Weniger Output"
    )
    # LLM-Optionen
    parser.add_argument(
        "--llm-model", type=str, default="llama3.1:8b",
        help="Ollama LLM Modell (default: llama3.1:8b)"
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="LLM deaktivieren (nur Embedding-basierte Prüfung)"
    )

    args = parser.parse_args()

    use_gpu = args.gpu and not args.no_gpu

    # Lade HotpotQA
    logger.info(f"Lade HotpotQA ({args.split}, {args.sample_size} Beispiele)...")
    loader = BenchmarkLoader()
    examples = loader.load_hotpotqa(split=args.split, sample_size=args.sample_size)

    if not examples:
        logger.error("HotpotQA konnte nicht geladen werden!")
        sys.exit(1)

    logger.info(f"✓ {len(examples)} Beispiele geladen")

    # Evaluator
    evaluator = RealisticHotpotQAEvaluator(
        use_gpu=use_gpu,
        missing_source_penalty=args.penalty,
        verbose=not args.quiet
    )

    # Setup mit LLM wenn gewünscht
    if args.no_llm:
        evaluator.setup(llm_model=None, ollama_url=None)
    else:
        evaluator.setup(llm_model=args.llm_model, ollama_url=args.ollama_url)

    evaluator.run_evaluation(examples)
    evaluator.print_final_report()
    evaluator.export_results(args.output)

    print(f"\n✓ Evaluation abgeschlossen: {args.output}")


if __name__ == "__main__":
    main()
