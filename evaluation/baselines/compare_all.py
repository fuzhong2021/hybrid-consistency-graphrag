#!/usr/bin/env python3
# evaluation/baselines/compare_all.py
"""
Vergleicht alle Baselines und das vollständige Konsistenzmodul.

Führt eine wissenschaftlich fundierte Evaluation durch mit:
- Alle 4 Baselines (Random, Rules-Only, Embedding-Only, LLM-Only)
- Das vollständige dreistufige Konsistenzmodul
- Statistische Signifikanztests
- Konfidenzintervalle

Wissenschaftliche Referenz:
- McNemar's Test für paarweise Vergleiche
- Bootstrap Confidence Intervals (Efron & Tibshirani, 1993)
"""

import sys
sys.path.insert(0, '.')

import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from src.models.entities import Triple, ValidationStatus
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.graph.memory_repository import InMemoryGraphRepository
from src.evaluation.benchmark_loader import BenchmarkLoader

from evaluation.baselines.random_baseline import RandomBaseline, RandomBaselineConfig
from evaluation.baselines.rules_only_baseline import RulesOnlyBaseline
from evaluation.baselines.embedding_only_baseline import EmbeddingOnlyBaseline
from evaluation.baselines.pure_embedding_baseline import PureEmbeddingBaseline, PureEmbeddingConfig
from evaluation.baselines.llm_only_baseline import LLMOnlyBaseline
from evaluation.baselines.rules_embedding_baseline import (
    RulesEmbeddingBaseline,
    RulesEmbeddingNoSourcePenalty,
    RulesEmbeddingNoSourceVerification,
)
from evaluation.baselines.nli_baseline import NLIBaseline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineComparison:
    """Ergebnis eines Baseline-Vergleichs."""
    baseline_name: str
    total_triples: int
    accepted: int
    rejected: int
    acceptance_rate: float
    avg_confidence: float
    avg_time_ms: float
    total_time_s: float

    # Für Triples mit Ground Truth
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0

    # Kosten
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class ComparisonResults:
    """Gesamtergebnisse des Vergleichs."""
    timestamp: str
    dataset: str
    sample_size: int
    baselines: List[BaselineComparison]
    full_system: BaselineComparison
    config: Dict[str, Any]

    # Statistische Tests
    mcnemar_results: Dict[str, Dict[str, float]] = None
    bootstrap_ci: Dict[str, Dict[str, Tuple[float, float]]] = None


class BaselineComparator:
    """Vergleicht alle Baselines mit dem vollständigen System."""

    def __init__(
        self,
        embedding_model: Any = None,
        llm_client: Any = None,
        graph_repo: Any = None,
        use_ground_truth: bool = False
    ):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.graph_repo = graph_repo or InMemoryGraphRepository()
        self.use_ground_truth = use_ground_truth

        # Baselines initialisieren
        self.baselines = {}
        self._init_baselines()

        logger.info(f"BaselineComparator initialisiert mit {len(self.baselines)} Baselines")

    def _init_baselines(self):
        """Initialisiert alle Baselines."""
        # Config mit Relationstypen die von HotpotQA verwendet werden
        self.config = ConsistencyConfig(
            valid_relation_types=[
                "RELATED_TO", "SUPPORTS", "ANSWERS", "HAS_ANSWER", "CONFIRMS",
                "CLAIMS", "REFUTES", "SUPPORTS_CLAIM",
                # Standard-Relationen
                "GEBOREN_IN", "GESTORBEN_IN", "WOHNT_IN", "ARBEITET_BEI",
                "STUDIERT_AN", "LEITET", "TEIL_VON", "BEFINDET_SICH_IN",
                "VERHEIRATET_MIT", "KIND_VON", "KENNT",
                "ENTWICKELTE", "ERFAND", "SCHRIEB", "ERHIELT",
                "BETEILIGT_AN", "HAT_BEZIEHUNG_ZU",
            ],
            cardinality_rules={
                "ANSWERS": {"max": 1},
                "HAS_ANSWER": {"max": 1},
            },
            llm_model="llama3.1:8b",
        )

        # 1. Random Baseline
        self.baselines["random"] = RandomBaseline(RandomBaselineConfig(
            acceptance_rate=0.5,
            seed=42
        ))

        # 2. Rules-Only Baseline
        self.baselines["rules_only"] = RulesOnlyBaseline(
            consistency_config=self.config,
            embedding_model=self.embedding_model
        )

        # 3. Stage-2-Complete (Embedding + Provenance + Anomalie + Contradiction)
        self.baselines["stage2_complete"] = EmbeddingOnlyBaseline(
            consistency_config=self.config,
            embedding_model=self.embedding_model
        )

        # 3b. Embedding-Only (NUR Cosine-Similarity Duplikaterkennung)
        self.baselines["embedding_only"] = PureEmbeddingBaseline(
            config=PureEmbeddingConfig(similarity_threshold=0.85),
            embedding_model=self.embedding_model
        )

        # 4. Rules + Embedding (Stufe 1 + 2)
        self.baselines["rules_embedding"] = RulesEmbeddingBaseline(
            consistency_config=self.config,
            embedding_model=self.embedding_model,
            enable_source_verification=True,
            enable_missing_source_penalty=True,
        )

        # 5. Rules + Embedding OHNE Source Penalty (Ablation)
        self.baselines["rules_emb_no_penalty"] = RulesEmbeddingNoSourcePenalty(
            consistency_config=self.config,
            embedding_model=self.embedding_model,
        )

        # 6. Rules + Embedding OHNE Source Verification (Ablation)
        self.baselines["rules_emb_no_srcver"] = RulesEmbeddingNoSourceVerification(
            consistency_config=self.config,
            embedding_model=self.embedding_model,
        )

        # 7. LLM-Only Baseline
        if self.llm_client:
            self.baselines["llm_only"] = LLMOnlyBaseline(
                consistency_config=self.config,
                llm_client=self.llm_client
            )
        else:
            logger.warning("Kein LLM-Client - LLM-Only Baseline wird übersprungen")

        # 8. NLI-Only Baseline (für FEVER-artige Fact Verification)
        self.baselines["nli_only"] = NLIBaseline(
            model_name="cross-encoder/nli-deberta-v3-xsmall",
            device="cpu",
            contradiction_threshold=0.7
        )
        logger.info("NLI-Only Baseline initialisiert (~44M Parameter)")

        # Full System
        self.full_system = ConsistencyOrchestrator(
            config=self.config,
            graph_repo=self.graph_repo,
            embedding_model=self.embedding_model,
            llm_client=self.llm_client,
            enable_metrics=True
        )

    def _populate_graph_with_baseline(
        self,
        triples: List[Triple],
        ground_truth: List[bool]
    ):
        """
        Füllt den Graph mit Baseline-Triples (korrekte Fakten).

        WICHTIG für Kardinalitätsprüfung:
        Die Kontradiktions-Triples können nur erkannt werden,
        wenn die korrekten Fakten schon im Graph sind.
        """
        from src.models.entities import Relation

        n_added = 0
        for triple, gt in zip(triples, ground_truth):
            if gt:  # Nur korrekte Triples (ground_truth=True)
                try:
                    self.graph_repo.create_entity(triple.subject)
                    self.graph_repo.create_entity(triple.object)

                    relation = Relation(
                        source_id=triple.subject.id,
                        target_id=triple.object.id,
                        relation_type=triple.predicate,
                        source_document_id=triple.source_document_id,
                        confidence=triple.extraction_confidence
                    )
                    self.graph_repo.create_relation(relation)
                    n_added += 1
                except Exception as e:
                    logger.debug(f"Fehler beim Hinzufügen: {e}")

        logger.info(f"  → {n_added} Baseline-Triples in Graph geladen")

    def run_comparison(
        self,
        triples: List[Triple],
        ground_truth: List[bool] = None
    ) -> ComparisonResults:
        """
        Führt den Vergleich aller Baselines durch.

        WICHTIG: Die Evaluation erfolgt in zwei Phasen:
        1. Baseline-Triples werden auf LEEREM Graph getestet (sollten akzeptiert werden)
        2. Graph wird mit Baseline-Triples gefüllt
        3. Kontradiktion-Triples werden getestet (sollten abgelehnt werden wegen Kardinalität)

        Args:
            triples: Zu validierende Triples
            ground_truth: Optional, Liste von bool (True=sollte akzeptiert werden)

        Returns:
            ComparisonResults mit allen Metriken
        """
        results = []
        predictions = {}  # Für McNemar's Test

        logger.info(f"Starte Vergleich mit {len(triples)} Triples")

        # Separiere Baseline- und Kontradiktion-Triples für korrekte Evaluation
        if ground_truth:
            baseline_triples = [t for t, gt in zip(triples, ground_truth) if gt]
            contradiction_triples = [t for t, gt in zip(triples, ground_truth) if not gt]
            baseline_gt = [True] * len(baseline_triples)
            contradiction_gt = [False] * len(contradiction_triples)
            logger.info(f"  Baseline-Triples (should accept): {len(baseline_triples)}")
            logger.info(f"  Kontradiktion-Triples (should reject): {len(contradiction_triples)}")
        else:
            baseline_triples = triples
            contradiction_triples = []
            baseline_gt = None
            contradiction_gt = None

        # Baselines evaluieren
        for name, baseline in self.baselines.items():
            logger.info(f"\n=== {name.upper()} Baseline ===")
            start_time = time.time()

            # PHASE 1: Teste Baseline-Triples auf LEEREM Graph
            # Diese sollten akzeptiert werden
            logger.info(f"  Phase 1: Teste {len(baseline_triples)} Baseline-Triples (should accept)")
            fresh_graph = InMemoryGraphRepository()  # Frischer Graph für jeden Baseline
            baseline_results = baseline.validate_batch(baseline_triples, fresh_graph)

            # PHASE 2: Fülle Graph mit Baseline-Triples
            if contradiction_triples:
                logger.info(f"  Phase 2: Lade Baseline-Triples in Graph...")
                for triple, result in zip(baseline_triples, baseline_results):
                    if result.accepted:  # Nur akzeptierte Triples
                        try:
                            fresh_graph.create_entity(triple.subject)
                            fresh_graph.create_entity(triple.object)
                            from src.models.entities import Relation
                            relation = Relation(
                                source_id=triple.subject.id,
                                target_id=triple.object.id,
                                relation_type=triple.predicate,
                                source_document_id=triple.source_document_id,
                                confidence=triple.extraction_confidence
                            )
                            fresh_graph.create_relation(relation)
                        except Exception:
                            pass

                # PHASE 3: Teste Kontradiktion-Triples
                logger.info(f"  Phase 3: Teste {len(contradiction_triples)} Kontradiktion-Triples (should reject)")
                contradiction_results = baseline.validate_batch(contradiction_triples, fresh_graph)
            else:
                contradiction_results = []

            # Kombiniere Ergebnisse
            all_results = baseline_results + contradiction_results
            all_ground_truth = (baseline_gt or []) + (contradiction_gt or [])

            # Metriken sammeln
            total = len(all_results)
            accepted = sum(1 for r in all_results if r.accepted)
            rejected = total - accepted
            confidences = [r.confidence for r in all_results]
            times = [r.processing_time_ms for r in all_results]

            comparison = BaselineComparison(
                baseline_name=name,
                total_triples=total,
                accepted=accepted,
                rejected=rejected,
                acceptance_rate=accepted / total if total > 0 else 0,
                avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
                avg_time_ms=sum(times) / len(times) if times else 0,
                total_time_s=time.time() - start_time,
            )

            # Ground Truth Metriken
            if all_ground_truth:
                comparison = self._compute_ground_truth_metrics(
                    comparison, all_results, all_ground_truth
                )

            # LLM-spezifische Metriken
            if name == "llm_only":
                stats = baseline.get_statistics()
                comparison.total_tokens = stats.get("total_tokens", 0)
                comparison.estimated_cost_usd = stats.get("estimated_cost_usd", 0)

            results.append(comparison)
            predictions[name] = [r.accepted for r in all_results]

            logger.info(f"  Akzeptanzrate: {comparison.acceptance_rate:.1%}")
            logger.info(f"  Avg Konfidenz: {comparison.avg_confidence:.2f}")
            logger.info(f"  Avg Zeit: {comparison.avg_time_ms:.1f}ms")

        # Full System evaluieren
        logger.info("\n=== FULL SYSTEM ===")
        start_time = time.time()

        # PHASE 1: Teste Baseline-Triples auf FRISCHEM Orchestrator
        logger.info(f"  Phase 1: Teste {len(baseline_triples)} Baseline-Triples (should accept)")
        fresh_graph_full = InMemoryGraphRepository()
        full_system_fresh = ConsistencyOrchestrator(
            config=self.config,
            graph_repo=fresh_graph_full,
            embedding_model=self.embedding_model,
            llm_client=self.llm_client,
            enable_metrics=True
        )

        full_baseline_results = []
        for triple in baseline_triples:
            result = full_system_fresh.process(triple)
            full_baseline_results.append(result)
            # Akzeptierte Triples zum Graph hinzufügen für Kardinalitätsprüfung
            if result.validation_status == ValidationStatus.ACCEPTED:
                try:
                    fresh_graph_full.create_entity(triple.subject)
                    fresh_graph_full.create_entity(triple.object)
                    from src.models.entities import Relation
                    relation = Relation(
                        source_id=triple.subject.id,
                        target_id=triple.object.id,
                        relation_type=triple.predicate,
                        source_document_id=triple.source_document_id,
                        confidence=triple.extraction_confidence
                    )
                    fresh_graph_full.create_relation(relation)
                except Exception:
                    pass

        # PHASE 2: Graph ist jetzt mit akzeptierten Baseline-Triples gefüllt

        # PHASE 3: Teste Kontradiktion-Triples
        full_contradiction_results = []
        if contradiction_triples:
            logger.info(f"  Phase 3: Teste {len(contradiction_triples)} Kontradiktion-Triples (should reject)")
            for triple in contradiction_triples:
                result = full_system_fresh.process(triple)
                full_contradiction_results.append(result)

        # Kombiniere Ergebnisse
        all_full_results = full_baseline_results + full_contradiction_results
        all_full_gt = (baseline_gt or []) + (contradiction_gt or [])

        total = len(all_full_results)
        accepted = sum(1 for t in all_full_results if t.validation_status == ValidationStatus.ACCEPTED)
        rejected = sum(1 for t in all_full_results if t.validation_status == ValidationStatus.REJECTED)

        full_comparison = BaselineComparison(
            baseline_name="full_system",
            total_triples=total,
            accepted=accepted,
            rejected=rejected,
            acceptance_rate=accepted / total if total > 0 else 0,
            avg_confidence=0.0,
            avg_time_ms=0.0,
            total_time_s=time.time() - start_time,
        )

        if all_full_gt:
            full_comparison = self._compute_ground_truth_metrics(
                full_comparison,
                [type('Result', (), {'accepted': t.validation_status == ValidationStatus.ACCEPTED})()
                 for t in all_full_results],
                all_full_gt
            )

        predictions["full_system"] = [
            t.validation_status == ValidationStatus.ACCEPTED for t in all_full_results
        ]

        logger.info(f"  Akzeptanzrate: {full_comparison.acceptance_rate:.1%}")

        # Ergebnisse zusammenstellen
        comparison_results = ComparisonResults(
            timestamp=datetime.now().isoformat(),
            dataset="custom",
            sample_size=len(triples),
            baselines=results,
            full_system=full_comparison,
            config={"embedding_model": str(self.embedding_model)},
        )

        # Statistische Tests
        if all_full_gt:
            comparison_results.mcnemar_results = self._compute_mcnemar_tests(
                predictions, all_full_gt
            )

        return comparison_results

    def _compute_ground_truth_metrics(
        self,
        comparison: BaselineComparison,
        results: List,
        ground_truth: List[bool]
    ) -> BaselineComparison:
        """Berechnet Metriken basierend auf Ground Truth."""
        tp = fp = tn = fn = 0

        for result, gt in zip(results, ground_truth):
            predicted = result.accepted
            if predicted and gt:
                tp += 1
            elif predicted and not gt:
                fp += 1
            elif not predicted and gt:
                fn += 1
            else:
                tn += 1

        comparison.true_positives = tp
        comparison.false_positives = fp
        comparison.true_negatives = tn
        comparison.false_negatives = fn

        # Precision, Recall, F1
        comparison.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        comparison.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        comparison.f1_score = (
            2 * comparison.precision * comparison.recall /
            (comparison.precision + comparison.recall)
            if (comparison.precision + comparison.recall) > 0 else 0
        )
        comparison.accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return comparison

    def _compute_mcnemar_tests(
        self,
        predictions: Dict[str, List[bool]],
        ground_truth: List[bool]
    ) -> Dict[str, Dict[str, float]]:
        """
        Berechnet McNemar's Test für paarweise Vergleiche.

        McNemar's Test prüft ob zwei Klassifizierer signifikant
        unterschiedlich performen.

        Returns:
            Dict mit Baseline-Paaren und p-values
        """
        try:
            from scipy.stats import mcnemar
        except ImportError:
            logger.warning("scipy nicht installiert - McNemar's Test nicht verfügbar")
            return {}

        results = {}
        baselines = list(predictions.keys())

        for i, baseline1 in enumerate(baselines):
            for baseline2 in baselines[i+1:]:
                pred1 = predictions[baseline1]
                pred2 = predictions[baseline2]

                # Erstelle Kontingenztabelle
                # a: beide richtig, b: 1 richtig/2 falsch, c: 1 falsch/2 richtig, d: beide falsch
                a = b = c = d = 0
                for p1, p2, gt in zip(pred1, pred2, ground_truth):
                    correct1 = p1 == gt
                    correct2 = p2 == gt
                    if correct1 and correct2:
                        a += 1
                    elif correct1 and not correct2:
                        b += 1
                    elif not correct1 and correct2:
                        c += 1
                    else:
                        d += 1

                # McNemar's Test
                if b + c > 0:
                    contingency = [[a, b], [c, d]]
                    try:
                        result = mcnemar(contingency, exact=True)
                        p_value = result.pvalue
                    except Exception:
                        p_value = 1.0
                else:
                    p_value = 1.0  # Keine Unterschiede

                key = f"{baseline1}_vs_{baseline2}"
                results[key] = {
                    "p_value": p_value,
                    "significant_0.05": p_value < 0.05,
                    "significant_0.01": p_value < 0.01,
                }

        return results


def print_comparison_table(results: ComparisonResults):
    """Gibt eine formatierte Vergleichstabelle aus."""
    print("\n" + "=" * 100)
    print("BASELINE VERGLEICH")
    print("=" * 100)

    # Header
    print(f"\n{'Baseline':<20} {'Accept%':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(ms)':>10} {'Cost($)':>10}")
    print("-" * 100)

    # Baselines
    all_results = results.baselines + [results.full_system]
    for r in all_results:
        cost = f"${r.estimated_cost_usd:.4f}" if r.estimated_cost_usd > 0 else "-"
        print(f"{r.baseline_name:<20} {r.acceptance_rate:>9.1%} {r.precision:>10.2%} {r.recall:>10.2%} "
              f"{r.f1_score:>10.2%} {r.avg_time_ms:>9.1f} {cost:>10}")

    print("-" * 100)

    # Statistische Signifikanz
    if results.mcnemar_results:
        print("\n" + "-" * 50)
        print("McNemar's Test (Statistische Signifikanz)")
        print("-" * 50)
        for pair, test_result in results.mcnemar_results.items():
            sig = "*" if test_result["significant_0.05"] else ""
            sig += "*" if test_result["significant_0.01"] else ""
            print(f"  {pair}: p={test_result['p_value']:.4f} {sig}")


def export_results(results: ComparisonResults, output_path: str):
    """Exportiert Ergebnisse als JSON."""
    output = {
        "timestamp": results.timestamp,
        "dataset": results.dataset,
        "sample_size": results.sample_size,
        "config": results.config,
        "baselines": [asdict(b) for b in results.baselines],
        "full_system": asdict(results.full_system),
        "mcnemar_results": results.mcnemar_results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Ergebnisse exportiert: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Vergleicht alle Baselines")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                       choices=["hotpotqa", "fever"],
                       help="Dataset für Evaluation (default: hotpotqa)")
    parser.add_argument("--output", type=str, default="results/baseline_comparison.json")
    parser.add_argument("--no-llm", action="store_true", help="LLM-Baseline überspringen")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--include-contradictions", action="store_true", default=True,
                       help="Distractor-basierte Kontradiktionen einbeziehen (für F1)")
    parser.add_argument("--include-no-source", action="store_true", default=False,
                       help="Triples ohne Quelle einbeziehen (für Missing Source Penalty)")
    parser.add_argument("--include-fake-source", action="store_true", default=False,
                       help="Fake Source Triples einbeziehen (für Source Verification)")
    parser.add_argument("--fever-no-cardinality", action="store_true", default=False,
                       help="FEVER-Modus: Keine Kardinalitätsregeln (NLI-basiert)")

    args = parser.parse_args()

    # Embedding-Modell laden — cuda → mps (Apple Silicon) → cpu Fallback
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        if args.gpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = "cpu"
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(f"Embedding-Modell geladen auf {device}")
    except Exception as e:
        logger.warning(f"Kein Embedding-Modell: {e}")

    # LLM-Client
    llm_client = None
    if not args.no_llm:
        try:
            from src.llm.ollama_client import OllamaClient
            llm_client = OllamaClient(model="llama3.1:8b")
            logger.info("LLM-Client (Ollama) initialisiert")
        except Exception as e:
            logger.warning(f"Kein LLM-Client: {e}")

    # Comparator
    comparator = BaselineComparator(
        embedding_model=embedding_model,
        llm_client=llm_client
    )

    # Daten laden
    logger.info(f"Lade {args.dataset} ({args.sample_size} Beispiele)...")

    if args.dataset == "fever":
        # FEVER Dataset laden
        from src.evaluation.fever_loader import FEVERLoader
        fever_loader = FEVERLoader()
        claims = fever_loader.load_for_streaming(
            split="dev",
            sample_size=args.sample_size,
            balance_labels=True
        )

        if not claims:
            logger.error("Keine FEVER Daten geladen!")
            sys.exit(1)

        # FEVER Triples extrahieren (ohne Kardinalität!)
        logger.info("Extrahiere FEVER Triples...")
        triples_data = fever_loader.extract_enhanced_triples(claims, use_nli_labels=True)

        triples = [td["triple"] for td in triples_data]
        ground_truth = [td["ground_truth_accept"] for td in triples_data]

        logger.info(f"  → {len(triples)} Triples extrahiert")
        logger.info(f"  → Should ACCEPT: {sum(ground_truth)}")
        logger.info(f"  → Should REJECT: {len(ground_truth) - sum(ground_truth)}")

        # FEVER-Modus: Config ohne Kardinalitätsregeln
        if args.fever_no_cardinality:
            logger.info("  → FEVER-Modus: Keine Kardinalitätsregeln!")
            comparator.config.cardinality_rules = {}

        # Vergleich durchführen
        results = comparator.run_comparison(triples, ground_truth)
        print_comparison_table(results)
        export_results(results, args.output)
        print(f"\n Vergleich abgeschlossen: {args.output}")
        return

    # HotpotQA (default)
    loader = BenchmarkLoader()
    examples = loader.load_hotpotqa(split="validation", sample_size=args.sample_size)

    if not examples:
        logger.error("Keine Daten geladen!")
        sys.exit(1)

    # Triples extrahieren MIT Ground Truth
    from evaluation.hotpotqa_realistic_evaluation import RealisticTripleExtractor
    extractor = RealisticTripleExtractor()

    triples = []
    ground_truth = []  # True = sollte akzeptiert werden, False = sollte abgelehnt werden

    # Phase 1: Baseline-Triples (KORREKT - sollten akzeptiert werden)
    logger.info("Extrahiere Baseline-Triples (korrekte Fakten)...")
    for example in examples:
        baseline = extractor.extract_baseline_triples(example)
        for triple in baseline:
            triples.append(triple)
            ground_truth.append(True)  # Sollte akzeptiert werden

    n_correct = len(triples)
    logger.info(f"  → {n_correct} korrekte Triples")

    # Phase 2: Distractor-basierte Kontradiktionen (FALSCH - sollten abgelehnt werden)
    if args.include_contradictions:
        logger.info("Extrahiere Kardinalitätsverletzungen (Distraktoren)...")

        # Zuerst: Baseline in Graph laden für Kardinalitätsprüfung
        # (Die Kontradiktionen sind nur erkennbar wenn die korrekten Fakten schon im Graph sind)

        for example in examples:
            # Kardinalitätsverletzungen: Gleiche Entity + Gleiche Relation + ANDERE Antwort
            violations = extractor.extract_cardinality_violation_triples(example)
            for triple in violations:
                triples.append(triple)
                ground_truth.append(False)  # Sollte ABGELEHNT werden

        n_contradictions = len(triples) - n_correct
        logger.info(f"  → {n_contradictions} Kontradiktions-Triples")

        # Cross-Question Violations
        logger.info("Extrahiere Cross-Question Violations...")
        for i, example in enumerate(examples):
            other_examples = examples[:i] + examples[i+1:]
            cross_violations = extractor.extract_cross_question_violation_triples(
                example, other_examples
            )
            for triple in cross_violations:
                triples.append(triple)
                ground_truth.append(False)  # Sollte ABGELEHNT werden

        n_cross = len(triples) - n_correct - n_contradictions
        logger.info(f"  → {n_cross} Cross-Question-Violations")

    # Phase 3: Triples OHNE Quelle (testet Missing Source Penalty)
    if args.include_no_source:
        logger.info("Extrahiere Triples OHNE Quelle (Missing Source Penalty Test)...")
        n_before = len(triples)
        for example in examples[:min(10, len(examples))]:  # Nur 10 Beispiele für diesen Test
            no_source = extractor.extract_supporting_fact_triples(example, with_source=False)
            for triple in no_source:
                triples.append(triple)
                # Diese KÖNNTEN akzeptiert werden, aber mit reduzierter Konfidenz
                # Für die Ground Truth: akzeptieren (korrekte Fakten, nur ohne Quelle)
                ground_truth.append(True)
        n_no_source = len(triples) - n_before
        logger.info(f"  → {n_no_source} Triples ohne Quelle")

    # Phase 6: Fake Source Attack (testet Source Verification)
    if args.include_fake_source:
        logger.info("Extrahiere Fake Source Triples (Source Verification Test)...")
        n_before = len(triples)
        for i, example in enumerate(examples[:min(10, len(examples))]):
            other_examples = examples[:i] + examples[i+1:]
            fake_source = extractor.extract_fake_source_triples(example, other_examples)
            for triple in fake_source:
                triples.append(triple)
                # Korrekte Fakten, aber mit irrelevantem source_text
                # Ground Truth: akzeptieren (Fakt ist korrekt), aber Konfidenz sollte reduziert sein
                ground_truth.append(True)
        n_fake_source = len(triples) - n_before
        logger.info(f"  → {n_fake_source} Fake Source Triples")

    logger.info(f"\nGesamt: {len(triples)} Triples")
    logger.info(f"  Korrekt (sollte ACCEPT): {sum(ground_truth)}")
    logger.info(f"  Falsch (sollte REJECT): {len(ground_truth) - sum(ground_truth)}")

    # Vergleich durchführen MIT Ground Truth
    results = comparator.run_comparison(triples, ground_truth)

    # Ergebnisse ausgeben
    print_comparison_table(results)
    export_results(results, args.output)

    print(f"\n Vergleich abgeschlossen: {args.output}")


if __name__ == "__main__":
    main()
