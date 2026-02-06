#!/usr/bin/env python3
"""
Comprehensive Evaluation — Orchestrierung aller 4 Strategien.

Strategien:
  1. Intrinsische Evaluation (synthetische Fehlererkennung)
  2. Manuelle Annotation (Export/Import + Cohen's Kappa)
  3. Graph-Qualitätsmetriken (Vorher/Nachher)
  4. Ablation + Information Quality (Signal/Noise-Trennung)

Verwendung:
  # Schnelltest
  python scripts/run_comprehensive_evaluation.py --strategies 1 --sample-size 5

  # Alle Strategien
  python scripts/run_comprehensive_evaluation.py --strategies 1,2,3,4 --sample-size 50

  # Vollständig
  python scripts/run_comprehensive_evaluation.py --strategies 1,2,3,4 --sample-size 200

  # Annotation Re-Import
  python scripts/run_comprehensive_evaluation.py --strategies 2 \\
    --annotation-file results/comprehensive/annotation_sample.json
"""

import sys
from pathlib import Path

# Projektroot zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
import importlib.util
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample
from src.evaluation.comprehensive import (
    TaggedTriple,
    EnhancedTripleExtractor,
    GraphQualityMetrics,
    GraphQualityAnalyzer,
    AnnotationSample,
    AnnotationManager,
    InformationQualityResult,
    InformationQualityEvaluator,
    LaTeXTableGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Import from sibling script
# =============================================================================

def _import_intrinsic_module():
    """Import TestDataGenerator & IntrinsicEvaluator from run_intrinsic_evaluation.py."""
    script_path = Path(__file__).parent / "run_intrinsic_evaluation.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"run_intrinsic_evaluation.py not found at {script_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "run_intrinsic_evaluation", str(script_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Evaluation Config (shared across strategies)
# =============================================================================

def get_evaluation_config() -> ConsistencyConfig:
    """Get ConsistencyConfig suitable for HotpotQA evaluation."""
    return ConsistencyConfig(
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        similarity_threshold=0.85,
        valid_relation_types=[
            # Standard-Typen
            "WOHNT_IN", "ARBEITET_BEI", "KENNT", "BETEILIGT_AN",
            "BEFINDET_SICH_IN", "HAT_BEZIEHUNG_ZU", "TEIL_VON",
            # Zusätzliche Typen für QA-Benchmarks
            "GEBOREN_IN", "GESTORBEN_IN", "GRUENDETE", "MITGLIED_VON",
            "REGIE_BEI", "VERHEIRATET_MIT", "VERWANDT_MIT", "SPIELT_FUER",
            "ASSOZIIERT_MIT", "VERBUNDEN_MIT", "TEILNAHME_AN", "RELATED_TO",
        ],
    )


# =============================================================================
# ComprehensiveEvaluator
# =============================================================================

class ComprehensiveEvaluator:
    """
    Orchestriert alle 4 Evaluationsstrategien.

    Lädt Daten einmal und teilt sie zwischen den Strategien.
    Jede Strategie bekommt frische Graph/Orchestrator-Instanzen.
    """

    def __init__(
        self,
        strategies: List[int],
        sample_size: int = 200,
        benchmark: str = "hotpotqa",
        with_llm: bool = False,
        annotation_file: Optional[str] = None,
        output_dir: str = "results/comprehensive",
    ):
        self.strategies = strategies
        self.sample_size = sample_size
        self.benchmark = benchmark
        self.with_llm = with_llm
        self.annotation_file = annotation_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = get_evaluation_config()
        self.examples: Optional[List[QAExample]] = None
        self.results: Dict[str, Any] = {}

        logger.info("ComprehensiveEvaluator initialisiert")
        logger.info(f"  Strategien: {strategies}")
        logger.info(f"  Sample-Size: {sample_size}")
        logger.info(f"  Benchmark: {benchmark}")
        logger.info(f"  With-LLM: {with_llm}")
        logger.info(f"  Output: {output_dir}")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _ensure_examples_loaded(self) -> List[QAExample]:
        """Load benchmark examples once and cache."""
        if self.examples is not None:
            return self.examples

        logger.info(f"Lade {self.benchmark} Beispiele (n={self.sample_size})...")
        loader = BenchmarkLoader()

        if self.benchmark.lower() == "hotpotqa":
            self.examples = loader.load_hotpotqa(
                split="validation", sample_size=self.sample_size
            )
        elif self.benchmark.lower() == "musique":
            self.examples = loader.load_musique(
                split="validation", sample_size=self.sample_size
            )
        else:
            raise ValueError(f"Unbekannter Benchmark: {self.benchmark}")

        if not self.examples:
            raise RuntimeError("Keine Beispiele geladen!")

        logger.info(f"Geladen: {len(self.examples)} Beispiele")
        return self.examples

    def _create_variant_orchestrator(
        self,
        variant_name: str,
        graph_repo: InMemoryGraphRepository,
    ) -> Optional[ConsistencyOrchestrator]:
        """Create a ConsistencyOrchestrator configured for a specific variant."""
        if variant_name == "BASELINE":
            return None

        if variant_name == "STAGE1_ONLY":
            return ConsistencyOrchestrator(
                config=self.config,
                graph_repo=graph_repo,
                embedding_model=None,
                llm_client=None,
                enable_metrics=True,
                always_check_duplicates=False,
            )

        if variant_name == "STAGE1_2":
            return ConsistencyOrchestrator(
                config=self.config,
                graph_repo=graph_repo,
                embedding_model=None,
                llm_client=None,
                enable_metrics=True,
                always_check_duplicates=True,
            )

        if variant_name == "FULL":
            llm_client = self._get_llm_client() if self.with_llm else None
            return ConsistencyOrchestrator(
                config=self.config,
                graph_repo=graph_repo,
                embedding_model=None,
                llm_client=llm_client,
                enable_metrics=True,
                always_check_duplicates=True,
            )

        raise ValueError(f"Unbekannte Variante: {variant_name}")

    def _get_llm_client(self) -> Optional[Any]:
        """Try to create an Ollama-based LLM client."""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            # Quick health check
            client.models.list()
            logger.info("Ollama LLM-Client verbunden")
            return client
        except Exception as e:
            logger.warning(f"LLM-Client nicht verfügbar: {e}")
            return None

    # -----------------------------------------------------------------
    # Strategy 1: Intrinsic Evaluation
    # -----------------------------------------------------------------

    def _run_strategy_1(self) -> Dict[str, Any]:
        """Run intrinsic evaluation using synthetic error injection."""
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGIE 1: Intrinsische Evaluation")
        logger.info("=" * 60)

        intrinsic = _import_intrinsic_module()

        # Use config from the intrinsic evaluation (stricter relation types)
        intrinsic_config = ConsistencyConfig(
            valid_entity_types=[
                "Person", "Organisation", "Ort", "Ereignis", "Konzept",
            ],
            valid_relation_types=[
                "GEBOREN_IN", "WOHNT_IN", "ARBEITET_BEI", "STUDIERT_AN",
                "BEFINDET_SICH_IN", "ERHIELT", "KENNT",
            ],
            cardinality_rules={
                "GEBOREN_IN": {"max": 1},
            },
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.7,
            similarity_threshold=0.75,
        )

        # Create orchestrator (without embeddings for speed)
        graph_repo = InMemoryGraphRepository()
        orchestrator = ConsistencyOrchestrator(
            config=intrinsic_config,
            graph_repo=graph_repo,
            embedding_model=None,
            llm_client=None,
            enable_metrics=True,
            always_check_duplicates=True,
        )

        # Generate test cases and evaluate
        generator = intrinsic.TestDataGenerator(intrinsic_config)
        test_cases = generator.generate_all_test_cases()

        evaluator = intrinsic.IntrinsicEvaluator(orchestrator, graph_repo)
        metrics = evaluator.evaluate(test_cases)
        evaluator.print_report()

        result = {
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "accuracy": metrics.accuracy,
                "confusion_matrix": {
                    "tp": metrics.true_positives,
                    "fp": metrics.false_positives,
                    "tn": metrics.true_negatives,
                    "fn": metrics.false_negatives,
                },
            },
            "per_type": metrics.per_type_results,
            "detailed_results": evaluator.detailed_results,
        }

        # Save
        out_path = self.output_dir / "intrinsic_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Gespeichert: {out_path}")

        return result

    # -----------------------------------------------------------------
    # Strategy 2: Manual Annotation
    # -----------------------------------------------------------------

    def _run_strategy_2(self) -> Dict[str, Any]:
        """Run annotation export or import+agreement."""
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGIE 2: Manuelle Annotation")
        logger.info("=" * 60)

        manager = AnnotationManager(sample_size=100)

        # --- RE-IMPORT MODE ---
        if self.annotation_file:
            logger.info(f"Importiere Annotationen aus {self.annotation_file}...")
            samples = manager.import_annotations(self.annotation_file)
            agreement = manager.compute_agreement_metrics(samples)

            logger.info(f"Agreement Metriken:")
            for k, v in agreement.items():
                logger.info(f"  {k}: {v}")

            out_path = self.output_dir / "annotation_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(agreement, f, indent=2, ensure_ascii=False)
            logger.info(f"Gespeichert: {out_path}")

            return agreement

        # --- EXPORT MODE ---
        examples = self._ensure_examples_loaded()

        logger.info("Extrahiere und validiere Triples für Annotation...")
        graph_repo = InMemoryGraphRepository()
        orchestrator = ConsistencyOrchestrator(
            config=self.config,
            graph_repo=graph_repo,
            embedding_model=None,
            llm_client=None,
            enable_metrics=True,
            always_check_duplicates=True,
        )

        extractor = EnhancedTripleExtractor()
        validated_triples: List[tuple] = []

        for i, example in enumerate(examples):
            if (i + 1) % 50 == 0:
                logger.info(f"  Fortschritt: {i + 1}/{len(examples)}")

            tagged = extractor.extract_all_triples(example)

            for t in tagged:
                # Fresh triple
                fresh = Triple(
                    subject=Entity(
                        name=t.triple.subject.name,
                        entity_type=t.triple.subject.entity_type,
                        source_document=t.triple.subject.source_document,
                    ),
                    predicate=t.triple.predicate,
                    object=Entity(
                        name=t.triple.object.name,
                        entity_type=t.triple.object.entity_type,
                        source_document=t.triple.object.source_document,
                    ),
                    source_text=t.triple.source_text,
                    source_document_id=t.triple.source_document_id,
                    extraction_confidence=t.triple.extraction_confidence,
                )
                validated = orchestrator.process(fresh)
                validated_triples.append(
                    (validated, validated.validation_status.value)
                )

                if validated.validation_status == ValidationStatus.ACCEPTED:
                    try:
                        graph_repo.save_triple(validated)
                    except Exception:
                        pass

        logger.info(f"Validiert: {len(validated_triples)} Triples")

        # Create stratified sample
        samples = manager.create_annotation_sample(validated_triples)

        # Export
        out_path = self.output_dir / "annotation_sample.json"
        manager.export_for_annotation(samples, str(out_path))

        return {
            "mode": "export",
            "total_triples_validated": len(validated_triples),
            "sample_size": len(samples),
            "export_path": str(out_path),
            "instruction": (
                "Bitte annotieren Sie die Datei und führen Sie dann erneut aus mit: "
                f"--strategies 2 --annotation-file {out_path}"
            ),
        }

    # -----------------------------------------------------------------
    # Strategy 3: Graph Quality (Before/After)
    # -----------------------------------------------------------------

    def _run_strategy_3(self) -> Dict[str, Any]:
        """Compare graph quality before vs. after consistency filtering."""
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGIE 3: Graph-Qualitätsmetriken (Vorher/Nachher)")
        logger.info("=" * 60)

        examples = self._ensure_examples_loaded()
        extractor = EnhancedTripleExtractor()
        analyzer = GraphQualityAnalyzer(self.config)

        # --- Graph A: All triples, no filtering ---
        logger.info("Erstelle Graph A (ungefiltert)...")
        graph_a = InMemoryGraphRepository()

        all_tagged: List[TaggedTriple] = []
        for example in examples:
            all_tagged.extend(extractor.extract_all_triples(example))

        for tagged in all_tagged:
            try:
                # Create fresh triple for Graph A
                fresh = Triple(
                    subject=Entity(
                        name=tagged.triple.subject.name,
                        entity_type=tagged.triple.subject.entity_type,
                        source_document=tagged.triple.subject.source_document,
                    ),
                    predicate=tagged.triple.predicate,
                    object=Entity(
                        name=tagged.triple.object.name,
                        entity_type=tagged.triple.object.entity_type,
                        source_document=tagged.triple.object.source_document,
                    ),
                    source_text=tagged.triple.source_text,
                    source_document_id=tagged.triple.source_document_id,
                    extraction_confidence=tagged.triple.extraction_confidence,
                )
                graph_a.save_triple(fresh)
            except Exception:
                pass

        metrics_a = analyzer.analyze(graph_a)
        logger.info(f"Graph A: {metrics_a.total_entities} Entitäten, "
                    f"{metrics_a.total_relations} Relationen")

        # --- Graph B: Only accepted triples ---
        logger.info("Erstelle Graph B (gefiltert)...")
        graph_b = InMemoryGraphRepository()
        orchestrator = ConsistencyOrchestrator(
            config=self.config,
            graph_repo=graph_b,
            embedding_model=None,
            llm_client=None,
            enable_metrics=True,
            always_check_duplicates=True,
        )

        accepted_count = 0
        rejected_count = 0
        for tagged in all_tagged:
            try:
                fresh = Triple(
                    subject=Entity(
                        name=tagged.triple.subject.name,
                        entity_type=tagged.triple.subject.entity_type,
                        source_document=tagged.triple.subject.source_document,
                    ),
                    predicate=tagged.triple.predicate,
                    object=Entity(
                        name=tagged.triple.object.name,
                        entity_type=tagged.triple.object.entity_type,
                        source_document=tagged.triple.object.source_document,
                    ),
                    source_text=tagged.triple.source_text,
                    source_document_id=tagged.triple.source_document_id,
                    extraction_confidence=tagged.triple.extraction_confidence,
                )
                validated = orchestrator.process(fresh)

                if validated.validation_status == ValidationStatus.ACCEPTED:
                    graph_b.save_triple(validated)
                    accepted_count += 1
                else:
                    rejected_count += 1
            except Exception as e:
                logger.debug(f"Fehler bei Triple-Verarbeitung: {e}")

        metrics_b = analyzer.analyze(graph_b)
        logger.info(
            f"Graph B: {metrics_b.total_entities} Entitäten, "
            f"{metrics_b.total_relations} Relationen "
            f"(accepted={accepted_count}, rejected={rejected_count})"
        )

        # Compare
        comparison = analyzer.compare(metrics_a, metrics_b)

        # Print summary
        print("\n" + "-" * 50)
        print("GRAPH-QUALITÄT: VORHER → NACHHER")
        print("-" * 50)
        for key in ["schema_compliance_rate", "self_loop_count",
                     "entity_duplication_rate", "domain_constraint_violations",
                     "avg_degree", "density", "isolated_entities"]:
            b_val = comparison["before"].get(key, 0)
            a_val = comparison["after"].get(key, 0)
            d_val = comparison["delta"].get(key, 0)
            print(f"  {key}: {b_val} → {a_val} (Δ {d_val:+})")
        print("-" * 50)

        # Save
        out_path = self.output_dir / "graph_quality_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        logger.info(f"Gespeichert: {out_path}")

        return comparison

    # -----------------------------------------------------------------
    # Strategy 4: Ablation + Information Quality
    # -----------------------------------------------------------------

    def _run_strategy_4(self) -> Dict[str, Any]:
        """Run ablation study with information quality metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGIE 4: Ablation + Information Quality")
        logger.info("=" * 60)

        examples = self._ensure_examples_loaded()
        evaluator = InformationQualityEvaluator(self.config)

        variant_names = ["BASELINE", "STAGE1_ONLY", "STAGE1_2", "FULL"]
        all_results: List[Dict[str, Any]] = []

        for variant_name in variant_names:
            logger.info(f"\n--- Variante: {variant_name} ---")

            # Fresh graph per variant
            graph_repo = InMemoryGraphRepository()
            orchestrator = self._create_variant_orchestrator(
                variant_name, graph_repo
            )

            result = evaluator.evaluate_variant(
                examples=examples,
                variant_name=variant_name,
                orchestrator=orchestrator,
                graph_repo=graph_repo,
            )

            result_dict = result.to_dict()
            all_results.append(result_dict)

            # Print
            logger.info(
                f"  Preservation: {result.supporting_preservation_rate:.1%}, "
                f"Removal: {result.distractor_removal_rate:.1%}, "
                f"Precision: {result.information_precision:.1%}, "
                f"Recall: {result.information_recall:.1%}, "
                f"F1: {result.information_f1:.1%}"
            )

            # Reset orchestrator stats for clean next variant
            if orchestrator is not None:
                orchestrator.reset_statistics()

        # Print comparison table
        print("\n" + "-" * 70)
        print(f"{'Variante':<15} {'Preserv':>8} {'Removal':>8} "
              f"{'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['variant_name']:<15} "
                f"{r['supporting_preservation_rate']:>7.1%} "
                f"{r['distractor_removal_rate']:>7.1%} "
                f"{r['information_precision']:>7.1%} "
                f"{r['information_recall']:>7.1%} "
                f"{r['information_f1']:>7.1%}"
            )
        print("-" * 70)

        # Save
        out_path = self.output_dir / "ablation_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Gespeichert: {out_path}")

        return {"variants": all_results}

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run all requested strategies and save combined output."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPREHENSIVE EVALUATION")
        logger.info(f"Strategien: {self.strategies}")
        logger.info(f"Zeitstempel: {datetime.now().isoformat()}")
        logger.info("=" * 70)

        start = datetime.now()

        if 1 in self.strategies:
            self.results["strategy_1"] = self._run_strategy_1()

        if 2 in self.strategies:
            self.results["strategy_2"] = self._run_strategy_2()

        if 3 in self.strategies:
            self.results["strategy_3"] = self._run_strategy_3()

        if 4 in self.strategies:
            self.results["strategy_4"] = self._run_strategy_4()

        duration = (datetime.now() - start).total_seconds()
        self.results["meta"] = {
            "strategies": self.strategies,
            "sample_size": self.sample_size,
            "benchmark": self.benchmark,
            "with_llm": self.with_llm,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }

        self._generate_latex()
        self._generate_summary()

        logger.info(f"\nEvaluation abgeschlossen in {duration:.1f}s")
        logger.info(f"Ergebnisse in: {self.output_dir}")

        return self.results

    # -----------------------------------------------------------------
    # Output generation
    # -----------------------------------------------------------------

    def _generate_latex(self) -> None:
        """Generate combined LaTeX tables for all completed strategies."""
        latex_parts = []
        latex_parts.append("% Comprehensive Evaluation — LaTeX-Tabellen")
        latex_parts.append(f"% Generiert: {datetime.now().isoformat()}\n")

        gen = LaTeXTableGenerator

        if "strategy_1" in self.results:
            latex_parts.append("% --- Strategie 1: Intrinsische Evaluation ---")
            latex_parts.append(gen.intrinsic_table(self.results["strategy_1"]))

        if "strategy_2" in self.results:
            s2 = self.results["strategy_2"]
            if "confusion_matrix" in s2:
                latex_parts.append(
                    "% --- Strategie 2: Manuelle Annotation ---"
                )
                latex_parts.append(gen.annotation_table(s2))

        if "strategy_3" in self.results:
            latex_parts.append("% --- Strategie 3: Graph-Qualität ---")
            latex_parts.append(
                gen.graph_quality_table(self.results["strategy_3"])
            )

        if "strategy_4" in self.results:
            latex_parts.append("% --- Strategie 4: Ablation Study ---")
            variants = self.results["strategy_4"].get("variants", [])
            latex_parts.append(gen.ablation_table(variants))

        out_path = self.output_dir / "latex_tables.tex"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(latex_parts))
        logger.info(f"LaTeX-Tabellen gespeichert: {out_path}")

    def _generate_summary(self) -> None:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("COMPREHENSIVE EVALUATION — ZUSAMMENFASSUNG")
        lines.append("=" * 70)
        lines.append(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Benchmark:   {self.benchmark}")
        lines.append(f"Sample-Size: {self.sample_size}")
        lines.append(f"Strategien:  {self.strategies}")
        lines.append("")

        # Strategy 1
        if "strategy_1" in self.results:
            m = self.results["strategy_1"].get("metrics", {})
            lines.append("STRATEGIE 1: Intrinsische Evaluation")
            lines.append("-" * 40)
            lines.append(f"  Precision: {m.get('precision', 0):.1%}")
            lines.append(f"  Recall:    {m.get('recall', 0):.1%}")
            lines.append(f"  F1-Score:  {m.get('f1_score', 0):.1%}")
            lines.append(f"  Accuracy:  {m.get('accuracy', 0):.1%}")
            lines.append("")

        # Strategy 2
        if "strategy_2" in self.results:
            s2 = self.results["strategy_2"]
            lines.append("STRATEGIE 2: Manuelle Annotation")
            lines.append("-" * 40)
            if s2.get("mode") == "export":
                lines.append(
                    f"  Export: {s2.get('sample_size', 0)} Samples exportiert"
                )
                lines.append(f"  Datei: {s2.get('export_path', '')}")
                lines.append("  → Bitte annotieren und erneut ausführen")
            else:
                lines.append(f"  Precision:      {s2.get('precision', 0):.1%}")
                lines.append(f"  Recall:         {s2.get('recall', 0):.1%}")
                lines.append(f"  F1:             {s2.get('f1', 0):.1%}")
                lines.append(
                    f"  Cohen's Kappa:  {s2.get('cohens_kappa', 0):.3f}"
                )
            lines.append("")

        # Strategy 3
        if "strategy_3" in self.results:
            s3 = self.results["strategy_3"]
            b = s3.get("before", {})
            a = s3.get("after", {})
            lines.append("STRATEGIE 3: Graph-Qualität (Vorher → Nachher)")
            lines.append("-" * 40)
            lines.append(
                f"  Entitäten:        {b.get('total_entities', 0)} → "
                f"{a.get('total_entities', 0)}"
            )
            lines.append(
                f"  Relationen:       {b.get('total_relations', 0)} → "
                f"{a.get('total_relations', 0)}"
            )
            lines.append(
                f"  Schema-Compliance: "
                f"{b.get('schema_compliance_rate', 0):.1%} → "
                f"{a.get('schema_compliance_rate', 0):.1%}"
            )
            lines.append(
                f"  Self-Loops:       {b.get('self_loop_count', 0)} → "
                f"{a.get('self_loop_count', 0)}"
            )
            lines.append(
                f"  Duplikationsrate: "
                f"{b.get('entity_duplication_rate', 0):.1%} → "
                f"{a.get('entity_duplication_rate', 0):.1%}"
            )
            lines.append("")

        # Strategy 4
        if "strategy_4" in self.results:
            variants = self.results["strategy_4"].get("variants", [])
            lines.append("STRATEGIE 4: Ablation + Information Quality")
            lines.append("-" * 40)
            lines.append(
                f"  {'Variante':<15} {'Preserv':>8} {'Removal':>8} "
                f"{'Prec':>8} {'Recall':>8} {'F1':>8}"
            )
            for v in variants:
                lines.append(
                    f"  {v['variant_name']:<15} "
                    f"{v['supporting_preservation_rate']:>7.1%} "
                    f"{v['distractor_removal_rate']:>7.1%} "
                    f"{v['information_precision']:>7.1%} "
                    f"{v['information_recall']:>7.1%} "
                    f"{v['information_f1']:>7.1%}"
                )
            lines.append("")

        # Duration
        meta = self.results.get("meta", {})
        lines.append(
            f"Gesamtdauer: {meta.get('duration_seconds', 0):.1f}s"
        )
        lines.append("=" * 70)

        summary_text = "\n".join(lines)

        out_path = self.output_dir / "evaluation_summary.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print("\n" + summary_text)
        logger.info(f"Zusammenfassung gespeichert: {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation des Konsistenzmoduls (4 Strategien)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Schnelltest (Strategie 1 only)
  python scripts/run_comprehensive_evaluation.py --strategies 1 --sample-size 5

  # Alle Strategien
  python scripts/run_comprehensive_evaluation.py --strategies 1,2,3,4 --sample-size 50

  # Vollständig
  python scripts/run_comprehensive_evaluation.py --strategies 1,2,3,4 --sample-size 200

  # Annotation Re-Import
  python scripts/run_comprehensive_evaluation.py --strategies 2 \\
    --annotation-file results/comprehensive/annotation_sample.json
        """,
    )

    parser.add_argument(
        "--strategies",
        type=str,
        default="1,2,3,4",
        help="Komma-separierte Liste der Strategien (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Anzahl QA-Beispiele für Strategien 2-4 (default: 200)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["hotpotqa", "musique"],
        default="hotpotqa",
        help="Benchmark-Dataset (default: hotpotqa)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="LLM-Arbitration aktivieren (benötigt Ollama)",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="Pfad zu annotierter JSON-Datei (für Strategie 2 Re-Import)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comprehensive",
        help="Ausgabeverzeichnis (default: results/comprehensive)",
    )

    args = parser.parse_args()

    # Parse strategies
    try:
        strategies = [int(s.strip()) for s in args.strategies.split(",")]
        for s in strategies:
            if s not in (1, 2, 3, 4):
                raise ValueError(f"Ungültige Strategie: {s}")
    except ValueError as e:
        parser.error(f"Ungültiges --strategies Argument: {e}")

    # Run
    evaluator = ComprehensiveEvaluator(
        strategies=strategies,
        sample_size=args.sample_size,
        benchmark=args.benchmark,
        with_llm=args.with_llm,
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
    )

    results = evaluator.run()

    # Final output
    print(f"\nGenerierte Dateien in {args.output_dir}/:")
    for f in sorted(Path(args.output_dir).glob("*")):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
