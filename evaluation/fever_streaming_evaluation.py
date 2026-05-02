#!/usr/bin/env python3
# evaluation/fever_streaming_evaluation.py
"""
FEVER-basierte Streaming Evaluation für GraphRAG Konsistenzmodul.

Wissenschaftliches Design:
- 6-Phasen Streaming Evaluation
- KEINE Kardinalitätsregeln für Hauptrelationen
- Hybrid-Vorteil über NLI + Embedding demonstrieren

Phasen:
1. FEVER-SUPPORTS Triples       → Ground Truth: ACCEPT
2. FEVER-REFUTES Triples        → Ground Truth: REJECT (via NLI)
3. Entity-Varianten generieren  → Ground Truth: MERGE
4. Cross-Document Konflikte     → Ground Truth: REJECT (via NLI)
5. SHUFFLE (zufällige Reihenfolge, seed=42)
6. Streaming-Verarbeitung + Metriken

Erwartete Ergebnisse:
| Baseline        | F1 (erwartet) | Grund                           |
|-----------------|---------------|----------------------------------|
| Random          | ~0.50         | Zufällig                        |
| Rules-Only      | ~0.55         | Keine semantischen Regeln!      |
| Embedding-Only  | ~0.67         | Entity-Matching, keine Contradiction |
| NLI-Only        | ~0.77         | Contradiction, kein Entity Matching |
| Hybrid          | ~0.85         | Kombination aller Stärken       |

Wissenschaftliche Referenzen:
- Thorne et al. (2018): FEVER Dataset
- Bowman et al. (2015): SNLI
- Williams et al. (2018): MultiNLI
- Lairgi et al. (2024): iText2KG Entity Resolution
- Heist & Paulheim (2019): Streaming KG Construction

Autor: Masterarbeit GraphRAG Konsistenzprüfung
"""

import sys
sys.path.insert(0, '.')

import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.entities import Entity, EntityType, Triple, ValidationStatus, Relation
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.graph.memory_repository import InMemoryGraphRepository
from src.evaluation.fever_loader import FEVERLoader, FEVERClaim

from evaluation.streaming.triple_generator import FEVERTripleGenerator, AnnotatedTriple, TripleCategory
from evaluation.streaming.entity_variant_generator import EntityVariantGenerator
from evaluation.streaming.cross_doc_generator import CrossDocConflictGenerator
from evaluation.streaming.shuffle_strategy import StreamingShuffler, ShuffleStrategy, create_shuffler

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
    merged: int = 0
    avg_confidence: float = 0.0
    processing_time_seconds: float = 0.0

    # Ground Truth Metriken
    true_positives: int = 0   # Korrekt klassifiziert
    false_positives: int = 0  # Falsch akzeptiert (sollte reject)
    true_negatives: int = 0   # Korrekt rejected
    false_negatives: int = 0  # Falsch rejected (sollte accept)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0


@dataclass
class BaselineMetrics:
    """Metriken für einen Baseline-Vergleich."""
    baseline_name: str
    total_triples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    avg_confidence: float = 0.0
    processing_time_seconds: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0


@dataclass
class StreamingEvaluationResults:
    """Gesamtergebnisse der Streaming Evaluation."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset: str = "fever"
    sample_size: int = 0
    seed: int = 42
    shuffle_strategy: str = "random"

    # Phasen-Metriken
    phase1_supports: Optional[PhaseMetrics] = None
    phase2_refutes: Optional[PhaseMetrics] = None
    phase3_entity_variants: Optional[PhaseMetrics] = None
    phase4_cross_doc: Optional[PhaseMetrics] = None
    phase5_shuffle_stats: Optional[Dict[str, Any]] = None
    phase6_streaming: Optional[PhaseMetrics] = None

    # Baseline-Vergleiche
    baselines: Dict[str, BaselineMetrics] = field(default_factory=dict)

    # Statistische Tests
    mcnemar_results: Optional[Dict[str, Any]] = None
    bootstrap_ci: Optional[Dict[str, Tuple[float, float]]] = None

    # Konfiguration
    config: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HAUPTKLASSE
# =============================================================================

class FEVERStreamingEvaluator:
    """
    Führt die 6-Phasen FEVER Streaming Evaluation durch.

    KRITISCHER UNTERSCHIED zu HotpotQA:
    - KEINE Kardinalitätsregeln für FEVER Relationen
    - Widersprüche müssen über NLI erkannt werden
    - Entity-Varianten müssen über Embedding-Matching erkannt werden

    Das demonstriert den Mehrwert des Hybrid-Systems.
    """

    def __init__(
        self,
        sample_size: int = 200,
        seed: int = 42,
        shuffle_strategy: str = "random",
        use_gpu: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        verbose: bool = True
    ):
        self.sample_size = sample_size
        self.seed = seed
        self.shuffle_strategy = shuffle_strategy
        self.use_gpu = use_gpu
        self.embedding_model_name = embedding_model_name
        self.verbose = verbose

        # Generatoren
        self.triple_generator = FEVERTripleGenerator()
        self.variant_generator = EntityVariantGenerator()
        self.conflict_generator = CrossDocConflictGenerator(seed=seed)
        self.shuffler = create_shuffler(shuffle_strategy, seed)

        # Ergebnisse
        self.results = StreamingEvaluationResults(
            seed=seed,
            shuffle_strategy=shuffle_strategy
        )

        # Modelle (werden in setup() initialisiert)
        self.embedding_model = None
        self.orchestrator = None
        self.graph_repo = None

    def setup(
        self,
        llm_model: Optional[str] = "llama3.1:8b",
        ollama_url: str = "http://localhost:11434"
    ):
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
            logger.info(f"  Modell geladen auf {device}")
        except Exception as e:
            logger.warning(f"  Kein Embedding-Modell: {e}")
            self.embedding_model = None

        # 2. LLM-Client
        llm_client = None
        if llm_model:
            try:
                from src.llm.ollama_client import OllamaClient
                llm_client = OllamaClient(model=llm_model, base_url=ollama_url)
                logger.info(f"  LLM-Client: {llm_model} @ {ollama_url}")
            except Exception as e:
                logger.warning(f"  Kein LLM-Client: {e}")

        # 3. Graph Repository
        self.graph_repo = InMemoryGraphRepository()

        # 4. Consistency Config OHNE Kardinalitätsregeln für FEVER
        # KRITISCH: Das ist der Hauptunterschied zu HotpotQA!
        config = ConsistencyConfig(
            valid_relation_types=[
                "CLAIMS", "STATES", "DESCRIBES", "MENTIONS", "ASSERTS",
                # Keine Kardinalitätsregeln für diese Relationen!
            ],
            cardinality_rules={},  # LEER! Keine Kardinalitätsregeln!
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.7,
            enable_missing_source_penalty=True,
            enable_provenance_boost=True,
            enable_source_verification=True,
        )

        # 5. Orchestrator
        self.orchestrator = ConsistencyOrchestrator(
            config=config,
            graph_repo=self.graph_repo,
            embedding_model=self.embedding_model,
            llm_client=llm_client,
            enable_metrics=True
        )
        logger.info("  Orchestrator initialisiert (OHNE Kardinalitätsregeln)")

        self.results.config = {
            "embedding_model": self.embedding_model_name,
            "use_gpu": self.use_gpu,
            "cardinality_rules": {},  # Explizit leer
            "llm_model": llm_model,
        }

    def load_fever_data(
        self,
        split: str = "dev"
    ) -> List[FEVERClaim]:
        """Lädt FEVER Daten."""
        logger.info(f"Lade FEVER ({split}, {self.sample_size} Beispiele)...")

        loader = FEVERLoader()
        claims = loader.load(
            split=split,
            sample_size=self.sample_size,
            filter_nei=True  # NOT ENOUGH INFO filtern
        )

        if not claims:
            logger.error("Keine FEVER Daten geladen!")
            return []

        logger.info(f"  {len(claims)} Claims geladen")
        self.results.sample_size = len(claims)
        return claims

    def run_evaluation(
        self,
        claims: List[FEVERClaim],
        compare_baselines: bool = True
    ) -> StreamingEvaluationResults:
        """
        Führt die vollständige 6-Phasen Evaluation durch.

        Phasen:
        1. FEVER-SUPPORTS Triples → Ground Truth: ACCEPT
        2. FEVER-REFUTES Triples → Ground Truth: REJECT
        3. Entity-Varianten → Ground Truth: MERGE
        4. Cross-Document Konflikte → Ground Truth: REJECT
        5. SHUFFLE
        6. Streaming-Verarbeitung + Metriken
        """
        logger.info(f"\n{'='*80}")
        logger.info("STARTE 6-PHASEN FEVER STREAMING EVALUATION")
        logger.info(f"{'='*80}")

        all_triples: List[AnnotatedTriple] = []

        # =====================================================================
        # PHASE 1: FEVER-SUPPORTS Triples
        # =====================================================================
        logger.info("\n--- PHASE 1: FEVER-SUPPORTS Triples ---")
        logger.info("  Ground Truth: ACCEPT (unterstützte Fakten)")

        supports_claims = [c for c in claims if c.is_supported]
        supports_triples = self._generate_triples_from_claims(supports_claims, "SUPPORTS")

        self.results.phase1_supports = PhaseMetrics(
            phase_name="Phase 1: SUPPORTS",
            total_triples=len(supports_triples)
        )
        all_triples.extend(supports_triples)

        logger.info(f"  Generiert: {len(supports_triples)} SUPPORTS Triples")

        # =====================================================================
        # PHASE 2: FEVER-REFUTES Triples
        # =====================================================================
        logger.info("\n--- PHASE 2: FEVER-REFUTES Triples ---")
        logger.info("  Ground Truth: REJECT (widerlegte Fakten via NLI)")

        refutes_claims = [c for c in claims if c.is_refuted]
        refutes_triples = self._generate_triples_from_claims(refutes_claims, "REFUTES")

        self.results.phase2_refutes = PhaseMetrics(
            phase_name="Phase 2: REFUTES",
            total_triples=len(refutes_triples)
        )
        all_triples.extend(refutes_triples)

        logger.info(f"  Generiert: {len(refutes_triples)} REFUTES Triples")

        # =====================================================================
        # PHASE 3: Entity-Varianten
        # =====================================================================
        logger.info("\n--- PHASE 3: Entity-Varianten ---")
        logger.info("  Ground Truth: MERGE (verschiedene Namen, gleiche Entity)")

        variant_triples = self.variant_generator.generate_variant_triples(
            supports_triples,
            variants_per_entity=2
        )

        self.results.phase3_entity_variants = PhaseMetrics(
            phase_name="Phase 3: Entity Variants",
            total_triples=len(variant_triples)
        )
        all_triples.extend(variant_triples)

        logger.info(f"  Generiert: {len(variant_triples)} Entity-Varianten Triples")

        # =====================================================================
        # PHASE 4: Cross-Document Konflikte
        # =====================================================================
        logger.info("\n--- PHASE 4: Cross-Document Konflikte ---")
        logger.info("  Ground Truth: REJECT (semantische Widersprüche via NLI)")
        logger.info("  WICHTIG: Keine Kardinalitätsregeln!")

        cross_doc_triples = self.conflict_generator.generate_semantic_contradictions(
            supports_triples,
            num_contradictions=min(50, len(supports_triples) // 2)
        )

        self.results.phase4_cross_doc = PhaseMetrics(
            phase_name="Phase 4: Cross-Doc Conflicts",
            total_triples=len(cross_doc_triples)
        )
        all_triples.extend(cross_doc_triples)

        logger.info(f"  Generiert: {len(cross_doc_triples)} Cross-Doc Konflikte")

        # =====================================================================
        # PHASE 5: SHUFFLE
        # =====================================================================
        logger.info(f"\n--- PHASE 5: SHUFFLE (strategy={self.shuffle_strategy}, seed={self.seed}) ---")

        shuffled_triples = self.shuffler.shuffle(all_triples)
        shuffle_stats = self.shuffler.get_statistics(shuffled_triples)

        self.results.phase5_shuffle_stats = shuffle_stats

        logger.info(f"  Gesamt: {len(shuffled_triples)} Triples")
        logger.info(f"  Kategorien: {shuffle_stats.get('categories', {})}")
        logger.info(f"  Ground Truth: {shuffle_stats.get('ground_truth', {})}")

        # =====================================================================
        # PHASE 6: Streaming-Verarbeitung
        # =====================================================================
        logger.info("\n--- PHASE 6: Streaming-Verarbeitung ---")

        self.results.phase6_streaming = self._process_streaming(shuffled_triples)

        # =====================================================================
        # BASELINE-VERGLEICHE
        # =====================================================================
        if compare_baselines:
            logger.info("\n--- BASELINE-VERGLEICHE ---")
            self._run_baseline_comparisons(shuffled_triples)

        return self.results

    def _generate_triples_from_claims(
        self,
        claims: List[FEVERClaim],
        label: str
    ) -> List[AnnotatedTriple]:
        """Generiert Triples aus FEVER Claims."""
        claim_dicts = []
        for claim in claims:
            # Evidence-Text extrahieren
            evidence_text = ""
            if claim.evidence:
                for ev in claim.evidence:
                    if isinstance(ev, list) and len(ev) >= 3:
                        evidence_text = ev[2]
                        break
                    elif isinstance(ev, dict):
                        evidence_text = ev.get("text", "")
                        if evidence_text:
                            break

            claim_dicts.append({
                "id": claim.id,
                "claim": claim.claim,
                "label": label,
                "evidence": claim.evidence,
            })

        return self.triple_generator.generate_batch(claim_dicts, include_nei=False)

    def _process_streaming(
        self,
        triples: List[AnnotatedTriple]
    ) -> PhaseMetrics:
        """
        Verarbeitet Triples im Streaming-Modus.

        Sammelt Metriken für jedes Triple basierend auf Ground Truth.
        """
        start_time = time.time()
        metrics = PhaseMetrics(phase_name="Phase 6: Streaming")
        confidences = []

        for annotated in triples:
            try:
                result = self.orchestrator.process(annotated.triple)
                metrics.total_triples += 1

                is_accepted = result.validation_status == ValidationStatus.ACCEPTED
                is_rejected = result.validation_status == ValidationStatus.REJECTED
                is_needs_review = result.validation_status == ValidationStatus.NEEDS_REVIEW

                # Ground Truth Vergleich
                if annotated.should_accept:
                    if is_accepted:
                        metrics.true_positives += 1
                        metrics.accepted += 1
                    else:
                        metrics.false_negatives += 1
                        metrics.rejected += 1

                elif annotated.should_reject:
                    if is_rejected or is_needs_review:
                        metrics.true_negatives += 1
                        metrics.rejected += 1
                    else:
                        metrics.false_positives += 1
                        metrics.accepted += 1

                elif annotated.should_merge:
                    # Merge = sollte akzeptiert werden nach Entity Resolution
                    if is_accepted:
                        metrics.true_positives += 1
                        metrics.merged += 1
                    else:
                        metrics.false_negatives += 1

                # Konfidenz sammeln
                if result.validation_history:
                    for h in result.validation_history:
                        if isinstance(h, dict) and "confidence" in h:
                            confidences.append(h["confidence"])

                # Persistiere akzeptierte Triples im Graph
                if is_accepted and self.graph_repo:
                    try:
                        self.graph_repo.create_entity(annotated.triple.subject)
                        self.graph_repo.create_entity(annotated.triple.object)
                        relation = Relation(
                            source_id=annotated.triple.subject.id,
                            target_id=annotated.triple.object.id,
                            relation_type=annotated.triple.predicate,
                            source_document_id=annotated.triple.source_document_id,
                            confidence=annotated.triple.extraction_confidence
                        )
                        self.graph_repo.create_relation(relation)
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"Fehler bei Triple-Verarbeitung: {e}")

        metrics.processing_time_seconds = time.time() - start_time
        if confidences:
            metrics.avg_confidence = sum(confidences) / len(confidences)

        return metrics

    def _run_baseline_comparisons(
        self,
        triples: List[AnnotatedTriple]
    ):
        """Führt Baseline-Vergleiche durch."""

        # Import Baselines
        try:
            from evaluation.baselines.random_baseline import RandomBaseline, RandomBaselineConfig
            from evaluation.baselines.rules_only_baseline import RulesOnlyBaseline
            from evaluation.baselines.embedding_only_baseline import EmbeddingOnlyBaseline
            from evaluation.baselines.nli_baseline import NLIBaseline
        except ImportError as e:
            logger.warning(f"Konnte Baselines nicht importieren: {e}")
            return

        # Config ohne Kardinalität
        config = ConsistencyConfig(
            valid_relation_types=["CLAIMS", "STATES", "DESCRIBES", "MENTIONS", "ASSERTS"],
            cardinality_rules={},
        )

        baselines = {
            "random": RandomBaseline(RandomBaselineConfig(acceptance_rate=0.5, seed=self.seed)),
            "rules_only": RulesOnlyBaseline(
                consistency_config=config,
                embedding_model=self.embedding_model
            ),
            "embedding_only": EmbeddingOnlyBaseline(
                consistency_config=config,
                embedding_model=self.embedding_model
            ),
            "nli_only": NLIBaseline(
                model_name="cross-encoder/nli-deberta-v3-xsmall",
                device="cpu"
            ),
        }

        # Extrahiere Raw Triples und Ground Truth
        raw_triples = [t.triple for t in triples]
        ground_truth = [t.should_accept for t in triples]

        for name, baseline in baselines.items():
            logger.info(f"  Evaluiere {name}...")
            start_time = time.time()

            try:
                # Frischer Graph für jeden Baseline
                fresh_graph = InMemoryGraphRepository()
                results = baseline.validate_batch(raw_triples, fresh_graph)

                # Metriken berechnen
                metrics = BaselineMetrics(baseline_name=name, total_triples=len(results))

                for result, gt, annotated in zip(results, ground_truth, triples):
                    if result.accepted:
                        if gt:
                            metrics.true_positives += 1
                        else:
                            metrics.false_positives += 1
                    else:
                        if gt:
                            metrics.false_negatives += 1
                        else:
                            metrics.true_negatives += 1

                metrics.processing_time_seconds = time.time() - start_time
                self.results.baselines[name] = metrics

                logger.info(f"    F1: {metrics.f1_score:.2%}, "
                           f"Precision: {metrics.precision:.2%}, "
                           f"Recall: {metrics.recall:.2%}")

            except Exception as e:
                logger.warning(f"  Fehler bei {name}: {e}")

    def print_final_report(self):
        """Gibt den finalen Bericht aus."""
        r = self.results

        print("\n" + "="*80)
        print("FINALER EVALUATIONSBERICHT - FEVER STREAMING EVALUATION")
        print("="*80)

        print(f"""
Dataset: {r.dataset}
Sample Size: {r.sample_size}
Seed: {r.seed}
Shuffle Strategy: {r.shuffle_strategy}
Timestamp: {r.timestamp}
        """)

        print("-"*80)
        print("PHASEN-ÜBERSICHT")
        print("-"*80)

        phases = [
            ("Phase 1: SUPPORTS", r.phase1_supports),
            ("Phase 2: REFUTES", r.phase2_refutes),
            ("Phase 3: Entity Variants", r.phase3_entity_variants),
            ("Phase 4: Cross-Doc Conflicts", r.phase4_cross_doc),
        ]

        for name, metrics in phases:
            if metrics:
                print(f"  {name}: {metrics.total_triples} Triples")

        if r.phase5_shuffle_stats:
            print(f"\n  Phase 5: Shuffle ({r.shuffle_strategy})")
            print(f"    Kategorien: {r.phase5_shuffle_stats.get('categories', {})}")

        print("-"*80)
        print("STREAMING-ERGEBNISSE (Phase 6)")
        print("-"*80)

        if r.phase6_streaming:
            m = r.phase6_streaming
            print(f"""
  Gesamt: {m.total_triples} Triples
  Accepted: {m.accepted}
  Rejected: {m.rejected}
  Merged: {m.merged}

  Precision: {m.precision:.2%}
  Recall: {m.recall:.2%}
  F1 Score: {m.f1_score:.2%}
  Accuracy: {m.accuracy:.2%}

  True Positives: {m.true_positives}
  False Positives: {m.false_positives}
  True Negatives: {m.true_negatives}
  False Negatives: {m.false_negatives}

  Avg Confidence: {m.avg_confidence:.2%}
  Processing Time: {m.processing_time_seconds:.2f}s
            """)

        print("-"*80)
        print("BASELINE-VERGLEICH")
        print("-"*80)

        print(f"\n{'Baseline':<20} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Accuracy':>10}")
        print("-"*60)

        # Full System zuerst
        if r.phase6_streaming:
            m = r.phase6_streaming
            print(f"{'HYBRID (Full)':*<20} {m.f1_score:>9.2%} {m.precision:>10.2%} "
                  f"{m.recall:>10.2%} {m.accuracy:>10.2%}")

        # Baselines
        for name, metrics in sorted(r.baselines.items()):
            print(f"{name:<20} {metrics.f1_score:>9.2%} {metrics.precision:>10.2%} "
                  f"{metrics.recall:>10.2%} {metrics.accuracy:>10.2%}")

        print("-"*80)
        print("INTERPRETATION")
        print("-"*80)

        if r.phase6_streaming and r.baselines:
            hybrid_f1 = r.phase6_streaming.f1_score
            rules_f1 = r.baselines.get("rules_only", BaselineMetrics("")).f1_score

            if hybrid_f1 > rules_f1:
                improvement = (hybrid_f1 - rules_f1) / rules_f1 * 100 if rules_f1 > 0 else 0
                print(f"  Hybrid-System verbessert Rules-Only um {improvement:.1f}%")
                print("  Der Hybrid-Vorteil kommt von:")
                print("    - NLI-basierte Widerspruchserkennung (keine Kardinalität!)")
                print("    - Embedding-basiertes Entity Matching")
            else:
                print("  WARNUNG: Hybrid performt nicht besser als Rules-Only")
                print("  Mögliche Gründe:")
                print("    - Zu wenige semantische Konflikte im Testset")
                print("    - NLI-Modell nicht sensitiv genug")

    def export_results(self, output_path: str):
        """Exportiert Ergebnisse als JSON."""

        def serialize_metrics(m):
            if m is None:
                return None
            if isinstance(m, dict):
                return m
            d = asdict(m)
            if hasattr(m, 'precision'):
                d['precision'] = m.precision
            if hasattr(m, 'recall'):
                d['recall'] = m.recall
            if hasattr(m, 'f1_score'):
                d['f1_score'] = m.f1_score
            if hasattr(m, 'accuracy'):
                d['accuracy'] = m.accuracy
            return d

        results_dict = {
            "timestamp": self.results.timestamp,
            "dataset": self.results.dataset,
            "sample_size": self.results.sample_size,
            "seed": self.results.seed,
            "shuffle_strategy": self.results.shuffle_strategy,
            "config": self.results.config,
            "phases": {
                "phase1_supports": serialize_metrics(self.results.phase1_supports),
                "phase2_refutes": serialize_metrics(self.results.phase2_refutes),
                "phase3_entity_variants": serialize_metrics(self.results.phase3_entity_variants),
                "phase4_cross_doc": serialize_metrics(self.results.phase4_cross_doc),
                "phase5_shuffle_stats": self.results.phase5_shuffle_stats,
                "phase6_streaming": serialize_metrics(self.results.phase6_streaming),
            },
            "baselines": {
                name: serialize_metrics(m) for name, m in self.results.baselines.items()
            },
            "mcnemar_results": self.results.mcnemar_results,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Ergebnisse exportiert: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FEVER-basierte Streaming Evaluation für GraphRAG Konsistenzprüfung"
    )
    parser.add_argument(
        "--sample-size", type=int, default=200,
        help="Anzahl FEVER Claims (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random Seed (default: 42)"
    )
    parser.add_argument(
        "--shuffle", type=str, default="random",
        choices=["random", "interleaved", "temporal", "clustered", "adversarial"],
        help="Shuffle-Strategie (default: random)"
    )
    parser.add_argument(
        "--split", type=str, default="dev",
        help="FEVER Split (default: dev)"
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
        "--output", type=str, default="results/fever_streaming_evaluation.json",
        help="Output-Pfad"
    )
    parser.add_argument(
        "--compare-baselines", action="store_true", default=True,
        help="Baseline-Vergleiche durchführen (default: True)"
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Baseline-Vergleiche deaktivieren"
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="Ollama LLM Modell (default: None - kein LLM)"
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama API URL"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Weniger Output"
    )

    args = parser.parse_args()

    use_gpu = args.gpu and not args.no_gpu
    compare_baselines = args.compare_baselines and not args.no_baselines

    # Evaluator
    evaluator = FEVERStreamingEvaluator(
        sample_size=args.sample_size,
        seed=args.seed,
        shuffle_strategy=args.shuffle,
        use_gpu=use_gpu,
        verbose=not args.quiet
    )

    # Setup
    evaluator.setup(
        llm_model=args.llm_model,
        ollama_url=args.ollama_url
    )

    # Lade FEVER Daten
    claims = evaluator.load_fever_data(split=args.split)

    if not claims:
        logger.error("Keine FEVER Daten - Abbruch")
        sys.exit(1)

    # Evaluation
    evaluator.run_evaluation(claims, compare_baselines=compare_baselines)

    # Report
    evaluator.print_final_report()
    evaluator.export_results(args.output)

    print(f"\n Evaluation abgeschlossen: {args.output}")


if __name__ == "__main__":
    main()
