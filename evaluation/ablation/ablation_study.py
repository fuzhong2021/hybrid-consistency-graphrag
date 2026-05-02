#!/usr/bin/env python3
# evaluation/ablation/ablation_study.py
"""
Ablation Study für das Konsistenzmodul.

Systematische Analyse welche Komponenten wie viel zum Gesamtergebnis beitragen.

Wissenschaftliche Grundlage:
- Ablation Studies sind Standard in ML-Forschung
- Zeigt den Beitrag jeder Komponente
- Wichtig für Verständnis und Verbesserung

Varianten:
1. Full System (Baseline)
2. -Stage2: Ohne Embedding-Validierung
3. -Stage3: Ohne LLM-Arbitration
4. -Provenance: Ohne Provenance-Boost
5. -SourceVerif: Ohne Source Verification
6. -SemanticTrig: Ohne Semantischen Trigger
7. Verschiedene Confidence-Kombinations-Methoden
"""

import sys
sys.path.insert(0, '.')

import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from copy import deepcopy

from src.models.entities import Triple, ValidationStatus
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.graph.memory_repository import InMemoryGraphRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AblationVariant:
    """Definition einer Ablation-Variante."""
    name: str
    description: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Ergebnisse
    total_triples: int = 0
    accepted: int = 0
    rejected: int = 0
    needs_review: int = 0
    acceptance_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_time_ms: float = 0.0
    total_time_s: float = 0.0

    # Mit Ground Truth
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0


@dataclass
class AblationResults:
    """Ergebnisse der Ablation Study."""
    timestamp: str
    dataset: str
    sample_size: int
    variants: List[AblationVariant]
    config: Dict[str, Any]

    # Zusammenfassung
    best_variant: str = ""
    baseline_f1: float = 0.0
    component_contributions: Dict[str, float] = field(default_factory=dict)


# Standard-Ablation-Varianten
STANDARD_VARIANTS = [
    AblationVariant(
        name="full_system",
        description="Vollständiges System (Baseline)",
        config_overrides={}
    ),
    AblationVariant(
        name="no_stage2",
        description="Ohne Stage 2 (Embedding-Validierung)",
        config_overrides={"disable_stage_2": True}
    ),
    AblationVariant(
        name="no_stage3",
        description="Ohne Stage 3 (LLM-Arbitration)",
        config_overrides={"disable_stage_3": True}
    ),
    AblationVariant(
        name="no_provenance",
        description="Ohne Provenance-Boost",
        config_overrides={
            "enable_provenance_boost": False,
            "disable_provenance_boost": True
        }
    ),
    AblationVariant(
        name="no_source_verification",
        description="Ohne Source Verification",
        config_overrides={"enable_source_verification": False}
    ),
    AblationVariant(
        name="no_missing_source_penalty",
        description="Ohne Missing Source Penalty",
        config_overrides={"enable_missing_source_penalty": False}
    ),
    AblationVariant(
        name="no_semantic_trigger",
        description="Ohne Semantischen Trigger",
        config_overrides={"enable_semantic_trigger": False}
    ),
    AblationVariant(
        name="conf_multiply",
        description="Konfidenz-Kombination: Multiplikation",
        config_overrides={"confidence_combination_method": "multiply"}
    ),
    AblationVariant(
        name="conf_min",
        description="Konfidenz-Kombination: Minimum (konservativ)",
        config_overrides={"confidence_combination_method": "min"}
    ),
    AblationVariant(
        name="conf_bayesian",
        description="Konfidenz-Kombination: Bayesian",
        config_overrides={"confidence_combination_method": "bayesian"}
    ),
]


class AblationStudy:
    """
    Führt eine systematische Ablation Study durch.

    Analysiert den Beitrag jeder Komponente zum Gesamtergebnis.
    """

    def __init__(
        self,
        base_config: ConsistencyConfig = None,
        embedding_model: Any = None,
        llm_client: Any = None,
        variants: List[AblationVariant] = None
    ):
        self.base_config = base_config or ConsistencyConfig()
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.variants = variants or deepcopy(STANDARD_VARIANTS)

        logger.info(f"AblationStudy initialisiert mit {len(self.variants)} Varianten")

    def _create_config_for_variant(self, variant: AblationVariant) -> ConsistencyConfig:
        """Erstellt eine Config für eine Ablation-Variante."""
        config_dict = {
            k: v for k, v in vars(self.base_config).items()
            if not k.startswith('_')
        }

        # Overrides anwenden
        config_dict.update(variant.config_overrides)

        return ConsistencyConfig(**config_dict)

    def run(
        self,
        triples: List[Triple],
        ground_truth: List[bool] = None
    ) -> AblationResults:
        """
        Führt die Ablation Study durch.

        Args:
            triples: Zu validierende Triples
            ground_truth: Optional, wahre Labels

        Returns:
            AblationResults mit allen Varianten-Ergebnissen
        """
        logger.info(f"Starte Ablation Study mit {len(triples)} Triples")
        logger.info(f"Varianten: {[v.name for v in self.variants]}")

        results = []

        for variant in self.variants:
            logger.info(f"\n=== Variante: {variant.name} ===")
            logger.info(f"    {variant.description}")

            # Config erstellen
            config = self._create_config_for_variant(variant)

            # Graph Repository (frisch für jede Variante)
            graph_repo = InMemoryGraphRepository()

            # Orchestrator erstellen
            try:
                orchestrator = ConsistencyOrchestrator(
                    config=config,
                    graph_repo=graph_repo,
                    embedding_model=self.embedding_model,
                    llm_client=self.llm_client if not config.disable_stage_3 else None,
                    enable_metrics=True
                )
            except Exception as e:
                logger.warning(f"Fehler bei Variante {variant.name}: {e}")
                continue

            # Triples verarbeiten
            start_time = time.time()
            processed_triples = []
            confidences = []

            for triple in triples:
                # Frisches Triple für jede Variante
                fresh_triple = deepcopy(triple)
                try:
                    result = orchestrator.process(fresh_triple)
                    processed_triples.append(result)

                    # Konfidenz aus letztem Validation Event
                    if result.validation_history:
                        last_event = result.validation_history[-1]
                        if isinstance(last_event, dict):
                            conf = last_event.get("confidence", 0.5)
                        else:
                            conf = getattr(last_event, "confidence", 0.5)
                        confidences.append(conf)
                except Exception as e:
                    logger.debug(f"Fehler bei Triple: {e}")

            total_time = time.time() - start_time

            # Metriken berechnen
            variant.total_triples = len(processed_triples)
            variant.accepted = sum(1 for t in processed_triples
                                  if t.validation_status == ValidationStatus.ACCEPTED)
            variant.rejected = sum(1 for t in processed_triples
                                  if t.validation_status == ValidationStatus.REJECTED)
            variant.needs_review = sum(1 for t in processed_triples
                                       if t.validation_status == ValidationStatus.NEEDS_REVIEW)
            variant.acceptance_rate = (
                variant.accepted / variant.total_triples
                if variant.total_triples > 0 else 0
            )
            variant.avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0
            )
            variant.avg_time_ms = (
                (total_time * 1000) / variant.total_triples
                if variant.total_triples > 0 else 0
            )
            variant.total_time_s = total_time

            # Ground Truth Metriken
            if ground_truth:
                predictions = [t.validation_status == ValidationStatus.ACCEPTED
                              for t in processed_triples]
                variant = self._compute_gt_metrics(variant, predictions, ground_truth)

            results.append(variant)

            logger.info(f"    Accepted: {variant.accepted}/{variant.total_triples} "
                       f"({variant.acceptance_rate:.1%})")
            if ground_truth:
                logger.info(f"    F1: {variant.f1_score:.3f}, "
                           f"Precision: {variant.precision:.3f}, "
                           f"Recall: {variant.recall:.3f}")

        # Zusammenfassung
        ablation_results = AblationResults(
            timestamp=datetime.now().isoformat(),
            dataset="custom",
            sample_size=len(triples),
            variants=results,
            config=asdict(self.base_config) if hasattr(self.base_config, '__dataclass_fields__') else {},
        )

        # Beitrag der Komponenten berechnen
        ablation_results = self._compute_contributions(ablation_results)

        return ablation_results

    def _compute_gt_metrics(
        self,
        variant: AblationVariant,
        predictions: List[bool],
        ground_truth: List[bool]
    ) -> AblationVariant:
        """Berechnet Ground-Truth-Metriken."""
        import numpy as np

        pred = np.array(predictions)
        gt = np.array(ground_truth)

        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        tn = np.sum(~pred & ~gt)
        fn = np.sum(~pred & gt)

        variant.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        variant.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        variant.f1_score = (
            2 * variant.precision * variant.recall /
            (variant.precision + variant.recall)
            if (variant.precision + variant.recall) > 0 else 0
        )
        variant.accuracy = (tp + tn) / (tp + tn + fp + fn) if len(predictions) > 0 else 0

        return variant

    def _compute_contributions(self, results: AblationResults) -> AblationResults:
        """Berechnet den Beitrag jeder Komponente."""
        # Finde Baseline (full_system)
        baseline = None
        for v in results.variants:
            if v.name == "full_system":
                baseline = v
                results.baseline_f1 = v.f1_score
                break

        if not baseline:
            return results

        # Berechne Beitrag als Differenz zu Ablation
        contributions = {}
        for v in results.variants:
            if v.name == "full_system":
                continue

            # Positiver Wert = Komponente hilft
            contribution = baseline.f1_score - v.f1_score
            component_name = v.name.replace("no_", "").replace("conf_", "confidence_")
            contributions[component_name] = round(contribution, 4)

        results.component_contributions = contributions

        # Beste Variante
        best_f1 = 0
        best_name = "full_system"
        for v in results.variants:
            if v.f1_score > best_f1:
                best_f1 = v.f1_score
                best_name = v.name
        results.best_variant = best_name

        return results


def print_ablation_table(results: AblationResults):
    """Gibt eine formatierte Ablation-Tabelle aus."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY ERGEBNISSE")
    print("=" * 100)

    print(f"\nDataset: {results.dataset}")
    print(f"Sample Size: {results.sample_size}")
    print(f"Timestamp: {results.timestamp}")

    # Tabelle
    print(f"\n{'Variante':<30} {'Accept%':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(ms)':>10}")
    print("-" * 100)

    for v in results.variants:
        print(f"{v.name:<30} {v.acceptance_rate:>9.1%} {v.precision:>10.3f} "
              f"{v.recall:>10.3f} {v.f1_score:>10.3f} {v.avg_time_ms:>9.1f}")

    print("-" * 100)

    # Komponenten-Beiträge
    print("\n" + "-" * 50)
    print("KOMPONENTEN-BEITRÄGE (F1-Differenz zu Baseline)")
    print("-" * 50)

    sorted_contributions = sorted(
        results.component_contributions.items(),
        key=lambda x: -x[1]
    )

    for component, contribution in sorted_contributions:
        sign = "+" if contribution >= 0 else ""
        print(f"  {component:<25}: {sign}{contribution:.4f}")

    print(f"\nBeste Variante: {results.best_variant}")
    print(f"Baseline F1: {results.baseline_f1:.4f}")


def export_ablation_results(results: AblationResults, output_path: str):
    """Exportiert Ablation-Ergebnisse als JSON."""
    output = {
        "timestamp": results.timestamp,
        "dataset": results.dataset,
        "sample_size": results.sample_size,
        "best_variant": results.best_variant,
        "baseline_f1": results.baseline_f1,
        "component_contributions": results.component_contributions,
        "variants": [asdict(v) for v in results.variants],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Ablation-Ergebnisse exportiert: {output_path}")


def run_ablation_study(
    triples: List[Triple],
    ground_truth: List[bool] = None,
    embedding_model: Any = None,
    llm_client: Any = None,
    custom_variants: List[AblationVariant] = None
) -> AblationResults:
    """
    Convenience-Funktion für Ablation Study.

    Args:
        triples: Zu validierende Triples
        ground_truth: Wahre Labels
        embedding_model: Embedding-Modell
        llm_client: LLM-Client
        custom_variants: Eigene Varianten (optional)

    Returns:
        AblationResults
    """
    study = AblationStudy(
        embedding_model=embedding_model,
        llm_client=llm_client,
        variants=custom_variants
    )
    return study.run(triples, ground_truth)


# ===========================================================================
# CLI Interface
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Führt Ablation Study durch")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--output", type=str, default="results/ablation_study.json")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--all-variants", action="store_true",
                       help="Alle Standard-Varianten durchführen")

    args = parser.parse_args()

    # Embedding-Modell
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if args.gpu else "cpu"
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
            logger.info("LLM-Client initialisiert")
        except Exception as e:
            logger.warning(f"Kein LLM-Client: {e}")

    # Daten laden
    from src.evaluation.benchmark_loader import BenchmarkLoader
    loader = BenchmarkLoader()
    examples = loader.load_hotpotqa(split="validation", sample_size=args.sample_size)

    if not examples:
        logger.error("Keine Daten!")
        sys.exit(1)

    # Triples extrahieren
    from evaluation.hotpotqa_realistic_evaluation import RealisticTripleExtractor
    extractor = RealisticTripleExtractor()
    triples = []
    for example in examples:
        triples.extend(extractor.extract_baseline_triples(example))

    logger.info(f"{len(triples)} Triples extrahiert")

    # Ablation Study
    study = AblationStudy(
        embedding_model=embedding_model,
        llm_client=llm_client
    )

    results = study.run(triples)

    # Ausgabe
    print_ablation_table(results)
    export_ablation_results(results, args.output)

    print(f"\n Ablation Study abgeschlossen: {args.output}")


if __name__ == "__main__":
    main()
