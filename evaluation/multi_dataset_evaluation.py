#!/usr/bin/env python3
# evaluation/multi_dataset_evaluation.py
"""
Multi-Dataset Evaluation für Konsistenzprüfung.

Evaluiert das System auf mehreren Datasets für bessere Generalisierung:
- HotpotQA: Multi-hop QA
- FEVER: Fact Verification
- MuSiQue: Multi-step Reasoning

Wissenschaftliche Motivation:
- Ein Dataset allein reicht nicht für robuste Evaluation
- Cross-Dataset Generalisierung zeigt echte Leistungsfähigkeit
- Verschiedene Aufgabentypen decken verschiedene Fähigkeiten ab
"""

import sys
sys.path.insert(0, '.')

import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from src.models.entities import Triple, ValidationStatus
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.graph.memory_repository import InMemoryGraphRepository
from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample
from src.evaluation.fever_loader import FEVERLoader
from src.evaluation.musique_loader import MuSiQueLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetrics:
    """Metriken für ein einzelnes Dataset."""
    dataset_name: str
    total_triples: int = 0
    accepted: int = 0
    rejected: int = 0
    needs_review: int = 0
    acceptance_rate: float = 0.0
    avg_confidence: float = 0.0
    processing_time_s: float = 0.0

    # Mit Ground Truth
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0


@dataclass
class MultiDatasetResults:
    """Gesamtergebnisse der Multi-Dataset Evaluation."""
    timestamp: str
    datasets: List[DatasetMetrics]
    config: Dict[str, Any]

    # Aggregierte Metriken
    avg_f1: float = 0.0
    avg_accuracy: float = 0.0
    generalization_score: float = 0.0  # Std-Abweichung (niedriger = besser)


class MultiDatasetEvaluator:
    """
    Evaluiert das Konsistenzmodul auf mehreren Datasets.

    Unterstützte Datasets:
    - hotpotqa: Multi-hop QA
    - fever: Fact Verification
    - musique: Multi-step Reasoning
    """

    def __init__(
        self,
        embedding_model: Any = None,
        llm_client: Any = None,
        config: ConsistencyConfig = None,
        enable_nli: bool = True,
        llm_model: str = "llama3.1:8b"
    ):
        self.embedding_model = embedding_model
        self.llm_client = llm_client

        # Default Config mit allen benötigten Relationstypen
        if config is None:
            config = ConsistencyConfig(
                valid_relation_types=[
                    # HotpotQA Relationen
                    "RELATED_TO", "SUPPORTS", "ANSWERS", "HAS_ANSWER", "CONFIRMS",
                    # FEVER Relationen
                    "CLAIMS", "REFUTES", "SUPPORTS_CLAIM",
                    # Standard-Relationen
                    "GEBOREN_IN", "GESTORBEN_IN", "WOHNT_IN", "ARBEITET_BEI",
                    "STUDIERT_AN", "LEITET", "TEIL_VON", "BEFINDET_SICH_IN",
                    "VERHEIRATET_MIT", "KIND_VON", "KENNT",
                ],
                cardinality_rules={
                    "ANSWERS": {"max": 1},
                    "HAS_ANSWER": {"max": 1},
                },
                # NLI für semantische Widerspruchserkennung (besonders für FEVER)
                enable_nli=enable_nli,
                nli_model="cross-encoder/nli-deberta-v3-xsmall",
                # LLM-Modell für Stage 3 (Ollama)
                llm_model=llm_model,
            )
        self.config = config

        self.loaders = {
            "hotpotqa": BenchmarkLoader(),
            "fever": FEVERLoader(),
            "musique": MuSiQueLoader(),
        }

        logger.info("MultiDatasetEvaluator initialisiert")

    def load_dataset(
        self,
        dataset_name: str,
        sample_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Lädt ein Dataset und konvertiert zu Triple-Format.

        Args:
            dataset_name: "hotpotqa", "fever", oder "musique"
            sample_size: Anzahl der Beispiele

        Returns:
            Liste von Dictionaries mit "triple" und "ground_truth"
        """
        if dataset_name == "hotpotqa":
            return self._load_hotpotqa(sample_size)
        elif dataset_name == "fever":
            return self._load_fever(sample_size)
        elif dataset_name == "musique":
            return self._load_musique(sample_size)
        else:
            raise ValueError(f"Unbekanntes Dataset: {dataset_name}")

    def _load_hotpotqa(self, sample_size: int) -> List[Dict[str, Any]]:
        """Lädt HotpotQA und konvertiert zu Triples."""
        from evaluation.hotpotqa_realistic_evaluation import RealisticTripleExtractor

        examples = self.loaders["hotpotqa"].load_hotpotqa(
            split="validation",
            sample_size=sample_size
        )

        if not examples:
            return []

        extractor = RealisticTripleExtractor()
        data = []

        for example in examples:
            # Baseline Triples (sollten akzeptiert werden)
            baseline = extractor.extract_baseline_triples(example)
            for triple in baseline:
                data.append({
                    "triple": triple,
                    "ground_truth_accept": True,
                    "is_contradiction": False,
                })

            # Kardinalitätsverletzungen (sollten abgelehnt werden)
            violations = extractor.extract_cardinality_violation_triples(example)
            for triple in violations:
                data.append({
                    "triple": triple,
                    "ground_truth_accept": False,
                    "is_contradiction": True,
                })

        return data

    def _load_fever(self, sample_size: int) -> List[Dict[str, Any]]:
        """Lädt FEVER und konvertiert zu Triples."""
        loader = self.loaders["fever"]
        claims = loader.load(split="dev", sample_size=sample_size, filter_nei=True)

        if not claims:
            return []

        triples_data = loader.convert_to_triples(claims)
        return triples_data

    def _load_musique(self, sample_size: int) -> List[Dict[str, Any]]:
        """Lädt MuSiQue und konvertiert zu Triples."""
        from evaluation.hotpotqa_realistic_evaluation import RealisticTripleExtractor

        loader = self.loaders["musique"]
        examples = loader.load(split="validation", sample_size=sample_size)

        if not examples:
            return []

        # Konvertiere zu QAExample Format
        qa_examples = loader.convert_to_qa_examples(examples)

        # Nutze HotpotQA-Extraktor (gleiche Logik wie bei HotpotQA)
        extractor = RealisticTripleExtractor()
        data = []

        for example in qa_examples:
            # Baseline Triples (sollten akzeptiert werden)
            baseline = extractor.extract_baseline_triples(example)
            for triple in baseline:
                data.append({
                    "triple": triple,
                    "ground_truth_accept": True,
                    "is_contradiction": False,
                })

            # Kardinalitätsverletzungen (sollten abgelehnt werden)
            violations = extractor.extract_cardinality_violation_triples(example)
            for triple in violations:
                data.append({
                    "triple": triple,
                    "ground_truth_accept": False,
                    "is_contradiction": True,
                })

        return data

    def evaluate_dataset(
        self,
        dataset_name: str,
        sample_size: int = 100
    ) -> DatasetMetrics:
        """
        Evaluiert auf einem einzelnen Dataset.

        Args:
            dataset_name: Name des Datasets
            sample_size: Anzahl der Beispiele

        Returns:
            DatasetMetrics mit Evaluationsergebnissen
        """
        logger.info(f"\n=== Evaluiere {dataset_name.upper()} ===")

        # Daten laden
        data = self.load_dataset(dataset_name, sample_size)

        if not data:
            logger.warning(f"Keine Daten für {dataset_name}")
            return DatasetMetrics(dataset_name=dataset_name)

        logger.info(f"  {len(data)} Triples geladen")

        # Graph Repository
        graph_repo = InMemoryGraphRepository()

        # Orchestrator
        orchestrator = ConsistencyOrchestrator(
            config=self.config,
            graph_repo=graph_repo,
            embedding_model=self.embedding_model,
            llm_client=self.llm_client,
            enable_metrics=True
        )

        # Separiere Baseline und Kontradiktion-Triples
        baseline_data = [d for d in data if d["ground_truth_accept"]]
        contradiction_data = [d for d in data if not d["ground_truth_accept"]]

        logger.info(f"  → {len(baseline_data)} Baseline (should accept)")
        logger.info(f"  → {len(contradiction_data)} Kontradiktionen (should reject)")

        # Per-Example JSONL-Logger für McNemar/ECE (Phase 1.3 + 1.4)
        # Verzeichnis aus Env (PER_EXAMPLE_DIR) oder Instanz-Attribut, sonst Default
        per_example_dir = (
            os.environ.get("PER_EXAMPLE_DIR")
            or getattr(self, "per_example_dir", "results/per_example")
        )
        per_example_path = Path(per_example_dir) / f"{dataset_name}.jsonl"
        per_example_path.parent.mkdir(parents=True, exist_ok=True)
        per_example_file = per_example_path.open("w")

        def _extract_last_conf(validation_history):
            if not validation_history:
                return 0.5
            last = validation_history[-1]
            if isinstance(last, dict):
                return float(last.get("confidence", 0.5))
            return float(getattr(last, "confidence", 0.5))

        def _extract_stage(validation_history):
            if not validation_history:
                return "unknown"
            last = validation_history[-1]
            if isinstance(last, dict):
                return str(last.get("stage", last.get("name", "unknown")))
            return str(getattr(last, "stage", getattr(last, "name", "unknown")))

        def _log_per_example(idx: int, triple, result, gold_label: bool) -> None:
            per_example_file.write(json.dumps({
                "triple_id": f"{dataset_name}_{idx}",
                "dataset": dataset_name,
                "y_true": int(gold_label),
                "y_pred": int(result.validation_status == ValidationStatus.ACCEPTED),
                "stage_decided": _extract_stage(result.validation_history),
                "confidence": _extract_last_conf(result.validation_history),
                "subject": getattr(triple.subject, "name", str(triple.subject)),
                "predicate": triple.predicate,
                "object": getattr(triple.object, "name", str(triple.object)),
            }) + "\n")

        # Evaluation
        start_time = time.time()
        results = []
        confidences = []

        # PHASE 1: Baseline-Triples verarbeiten (auf leerem Graph)
        from src.models.entities import Relation
        for idx, item in enumerate(baseline_data):
            triple = item["triple"]
            try:
                result = orchestrator.process(triple)
                results.append({
                    "status": result.validation_status,
                    "ground_truth": True,
                    "is_contradiction": False,
                })
                _log_per_example(idx, triple, result, gold_label=True)

                # Akzeptierte Triples zum Graph hinzufügen für Kardinalitätsprüfung
                if result.validation_status == ValidationStatus.ACCEPTED:
                    try:
                        graph_repo.create_entity(triple.subject)
                        graph_repo.create_entity(triple.object)
                        relation = Relation(
                            source_id=triple.subject.id,
                            target_id=triple.object.id,
                            relation_type=triple.predicate,
                            source_document_id=triple.source_document_id,
                            confidence=triple.extraction_confidence
                        )
                        graph_repo.create_relation(relation)
                    except Exception:
                        pass

                if result.validation_history:
                    last = result.validation_history[-1]
                    if isinstance(last, dict):
                        confidences.append(last.get("confidence", 0.5))
                    else:
                        confidences.append(getattr(last, "confidence", 0.5))
            except Exception as e:
                logger.debug(f"Fehler: {e}")

        # PHASE 2: Kontradiktion-Triples testen (Graph enthält jetzt Baseline)
        offset = len(baseline_data)
        for idx, item in enumerate(contradiction_data):
            triple = item["triple"]
            try:
                result = orchestrator.process(triple)
                results.append({
                    "status": result.validation_status,
                    "ground_truth": False,  # Sollte rejected werden
                    "is_contradiction": True,
                })
                _log_per_example(offset + idx, triple, result, gold_label=False)

                if result.validation_history:
                    last = result.validation_history[-1]
                    if isinstance(last, dict):
                        confidences.append(last.get("confidence", 0.5))
                    else:
                        confidences.append(getattr(last, "confidence", 0.5))
            except Exception as e:
                logger.debug(f"Fehler: {e}")

        per_example_file.close()
        logger.info(f"  Per-example log → {per_example_path}")

        total_time = time.time() - start_time

        # Metriken berechnen
        metrics = DatasetMetrics(dataset_name=dataset_name)
        metrics.total_triples = len(results)
        metrics.processing_time_s = total_time

        for r in results:
            if r["status"] == ValidationStatus.ACCEPTED:
                metrics.accepted += 1
                if r["ground_truth"]:
                    metrics.true_negatives += 1  # Korrekt akzeptiert
                else:
                    metrics.false_negatives += 1  # Fälschlicherweise akzeptiert
            elif r["status"] == ValidationStatus.REJECTED:
                metrics.rejected += 1
                if not r["ground_truth"]:
                    metrics.true_positives += 1  # Korrekt abgelehnt
                else:
                    metrics.false_positives += 1  # Fälschlicherweise abgelehnt
            else:
                metrics.needs_review += 1

        # Raten berechnen
        if metrics.total_triples > 0:
            metrics.acceptance_rate = metrics.accepted / metrics.total_triples

        if confidences:
            metrics.avg_confidence = sum(confidences) / len(confidences)

        # Precision/Recall/F1
        tp, fp, tn, fn = (metrics.true_positives, metrics.false_positives,
                          metrics.true_negatives, metrics.false_negatives)

        if tp + fp > 0:
            metrics.precision = tp / (tp + fp)
        if tp + fn > 0:
            metrics.recall = tp / (tp + fn)
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        if tp + tn + fp + fn > 0:
            metrics.accuracy = (tp + tn) / (tp + tn + fp + fn)

        logger.info(f"  Accepted: {metrics.accepted}/{metrics.total_triples} ({metrics.acceptance_rate:.1%})")
        logger.info(f"  F1: {metrics.f1_score:.3f}, Accuracy: {metrics.accuracy:.3f}")

        return metrics

    def run_full_evaluation(
        self,
        datasets: List[str] = None,
        sample_size: int = 100
    ) -> MultiDatasetResults:
        """
        Führt vollständige Multi-Dataset Evaluation durch.

        Args:
            datasets: Liste von Dataset-Namen
            sample_size: Beispiele pro Dataset

        Returns:
            MultiDatasetResults mit allen Ergebnissen
        """
        if datasets is None:
            datasets = ["hotpotqa", "fever", "musique"]

        logger.info(f"Starte Multi-Dataset Evaluation: {datasets}")

        results = []
        for dataset in datasets:
            try:
                metrics = self.evaluate_dataset(dataset, sample_size)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Fehler bei {dataset}: {e}")

        # Aggregierte Metriken
        if results:
            f1_scores = [r.f1_score for r in results]
            acc_scores = [r.accuracy for r in results]

            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_acc = sum(acc_scores) / len(acc_scores)

            # Generalisierung: niedrige Std = gute Generalisierung
            import numpy as np
            generalization = np.std(f1_scores)
        else:
            avg_f1 = avg_acc = generalization = 0.0

        return MultiDatasetResults(
            timestamp=datetime.now().isoformat(),
            datasets=results,
            config={},
            avg_f1=avg_f1,
            avg_accuracy=avg_acc,
            generalization_score=generalization,
        )


def print_multi_dataset_table(results: MultiDatasetResults):
    """Gibt formatierte Ergebnistabelle aus."""
    print("\n" + "=" * 100)
    print("MULTI-DATASET EVALUATION")
    print("=" * 100)

    print(f"\n{'Dataset':<15} {'Triples':>10} {'Accept%':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(s)':>10}")
    print("-" * 100)

    for d in results.datasets:
        print(f"{d.dataset_name:<15} {d.total_triples:>10} {d.acceptance_rate:>9.1%} "
              f"{d.precision:>10.3f} {d.recall:>10.3f} {d.f1_score:>10.3f} {d.processing_time_s:>9.1f}")

    print("-" * 100)
    print(f"{'DURCHSCHNITT':<15} {'':<10} {'':<10} {'':<10} {'':<10} "
          f"{results.avg_f1:>10.3f} {'':<10}")

    print(f"\nGeneralisierungs-Score (Std F1): {results.generalization_score:.4f}")
    print("(Niedriger = bessere Generalisierung)")


def export_results(results: MultiDatasetResults, output_path: str):
    """Exportiert Ergebnisse als JSON."""
    output = {
        "timestamp": results.timestamp,
        "avg_f1": results.avg_f1,
        "avg_accuracy": results.avg_accuracy,
        "generalization_score": results.generalization_score,
        "datasets": [asdict(d) for d in results.datasets],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Ergebnisse exportiert: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Evaluation")
    parser.add_argument("--datasets", type=str, default="hotpotqa,fever,musique",
                       help="Komma-separierte Liste von Datasets")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/multi_dataset_evaluation.json")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--nli", action=argparse.BooleanOptionalAction, default=True,
                       help="NLI-Stufe (default: on). --no-nli für Ablation.")
    parser.add_argument("--visualize", action="store_true",
                       help="Generiere Thesis-Visualisierungen nach Evaluation")
    parser.add_argument("--figures-dir", type=str, default="results/figures",
                       help="Ausgabeverzeichnis für Visualisierungen")

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    # Embedding-Modell — cuda → mps (Apple Silicon) → cpu Fallback
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
                logger.info("Weder CUDA noch MPS verfügbar, nutze CPU")
        else:
            device = "cpu"

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(f"Embedding-Modell auf {device}")
    except Exception as e:
        logger.warning(f"Kein Embedding-Modell: {e}")
        import traceback
        traceback.print_exc()

    # LLM-Client
    llm_client = None
    if not args.no_llm:
        try:
            from src.llm.ollama_client import OllamaClient
            llm_client = OllamaClient(model="llama3.1:8b")
            logger.info("LLM-Client initialisiert")
        except Exception as e:
            logger.warning(f"Kein LLM: {e}")

    # Evaluator
    evaluator = MultiDatasetEvaluator(
        embedding_model=embedding_model,
        llm_client=llm_client,
        enable_nli=args.nli,
        llm_model="llama3.1:8b"  # Ollama Model
    )

    if args.nli:
        logger.info("NLI-Validator aktiviert für semantische Widerspruchserkennung")

    # Evaluation
    results = evaluator.run_full_evaluation(datasets, args.sample_size)

    # Ausgabe
    print_multi_dataset_table(results)
    export_results(results, args.output)

    # Visualisierungen generieren
    if args.visualize:
        print("\nGeneriere Thesis-Visualisierungen...")
        try:
            from evaluation.visualization import ThesisVisualizer

            visualizer = ThesisVisualizer(output_dir=args.figures_dir)

            # Baseline-Vergleich falls vorhanden
            baseline_path = args.output.replace("multi_dataset", "baseline_comparison")
            if not os.path.exists(baseline_path):
                baseline_path = "results/baseline_comparison.json"

            visualizer.generate_all_from_results(
                evaluation_results_path=args.output,
                baseline_results_path=baseline_path if os.path.exists(baseline_path) else None
            )
        except Exception as e:
            logger.error(f"Visualisierung fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n Evaluation abgeschlossen: {args.output}")
    if args.visualize:
        print(f" Visualisierungen: {args.figures_dir}/")


if __name__ == "__main__":
    main()
