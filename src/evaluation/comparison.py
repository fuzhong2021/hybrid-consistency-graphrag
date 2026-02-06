# src/evaluation/comparison.py
"""
Ablation Study Framework für Konsistenzmodul.

Ermöglicht systematischen Vergleich:
- Baseline: Nur GraphRAG
- +Stufe 1: GraphRAG + Regelbasiert
- +Stufe 2: GraphRAG + Regelbasiert + Embedding
- +Stufe 3: GraphRAG + Alle Stufen

Generiert LaTeX-Tabellen und Visualisierungen für die Masterarbeit.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time

from src.evaluation.benchmark_loader import QAExample, BenchmarkLoader, BenchmarkType
from src.evaluation.qa_evaluator import QAEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class AblationVariant(Enum):
    """Varianten für Ablation Study."""
    BASELINE = "baseline"              # Nur GraphRAG, keine Konsistenzprüfung
    STAGE1_ONLY = "stage1_only"        # Nur regelbasiert
    STAGE1_2 = "stage1_2"              # Regelbasiert + Embedding
    FULL = "full"                      # Alle drei Stufen


@dataclass
class AblationConfig:
    """Konfiguration für eine Ablation Study."""
    benchmark: BenchmarkType = BenchmarkType.HOTPOTQA
    sample_size: int = 100
    split: str = "validation"

    # Welche Varianten testen
    variants: List[AblationVariant] = field(default_factory=lambda: [
        AblationVariant.BASELINE,
        AblationVariant.STAGE1_ONLY,
        AblationVariant.STAGE1_2,
        AblationVariant.FULL,
    ])

    # Wiederholungen für statistische Signifikanz
    num_runs: int = 1

    # Export-Optionen
    export_path: Optional[str] = None
    export_format: str = "json"  # "json", "csv", "latex"


@dataclass
class VariantResult:
    """Ergebnis einer einzelnen Variante."""
    variant: AblationVariant
    evaluation: EvaluationResult

    # Vergleich zur Baseline
    em_delta: float = 0.0
    f1_delta: float = 0.0

    # Overhead-Analyse
    avg_overhead_ms: float = 0.0
    relative_overhead: float = 0.0  # Prozentual

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "variant": self.variant.value,
            "exact_match": self.evaluation.exact_match,
            "f1_score": self.evaluation.f1_score,
            "answer_recall": self.evaluation.answer_recall,
            "em_delta": self.em_delta,
            "f1_delta": self.f1_delta,
            "avg_latency_ms": self.evaluation.avg_latency_ms,
            "avg_overhead_ms": self.avg_overhead_ms,
            "relative_overhead": self.relative_overhead,
            "triples_accepted": self.evaluation.triples_accepted,
            "triples_rejected": self.evaluation.triples_rejected,
        }


@dataclass
class AblationResult:
    """Gesamtergebnis einer Ablation Study."""
    config: AblationConfig
    variant_results: List[VariantResult] = field(default_factory=list)

    # Zusammenfassung
    best_variant: Optional[AblationVariant] = None
    max_improvement_em: float = 0.0
    max_improvement_f1: float = 0.0

    # Zeitstempel
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "config": {
                "benchmark": self.config.benchmark.value,
                "sample_size": self.config.sample_size,
                "split": self.config.split,
                "num_runs": self.config.num_runs,
            },
            "variant_results": [v.to_dict() for v in self.variant_results],
            "best_variant": self.best_variant.value if self.best_variant else None,
            "max_improvement_em": self.max_improvement_em,
            "max_improvement_f1": self.max_improvement_f1,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
        }

    def get_comparison_table(self) -> str:
        """Erstellt eine formatierte Vergleichstabelle."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("ABLATION STUDY RESULTS")
        lines.append("=" * 80)
        lines.append(f"Benchmark: {self.config.benchmark.value}")
        lines.append(f"Sample Size: {self.config.sample_size}")
        lines.append("-" * 80)
        lines.append(f"{'Variante':<20} {'EM':>10} {'F1':>10} {'Delta EM':>12} {'Delta F1':>12} {'Latenz':>12}")
        lines.append("-" * 80)

        for vr in self.variant_results:
            lines.append(
                f"{vr.variant.value:<20} "
                f"{vr.evaluation.exact_match:>9.1%} "
                f"{vr.evaluation.f1_score:>9.1%} "
                f"{vr.em_delta:>+11.1%} "
                f"{vr.f1_delta:>+11.1%} "
                f"{vr.evaluation.avg_latency_ms:>10.1f}ms"
            )

        lines.append("-" * 80)
        if self.best_variant:
            lines.append(f"Beste Variante: {self.best_variant.value}")
            lines.append(f"Max. Verbesserung: EM +{self.max_improvement_em:.1%}, F1 +{self.max_improvement_f1:.1%}")
        lines.append("=" * 80 + "\n")

        return "\n".join(lines)


class ConsistencyAblationStudy:
    """
    Führt A/B-Vergleich durch: Mit vs. ohne Konsistenzmodul.

    Testet systematisch verschiedene Konfigurationen des
    Konsistenzmoduls auf QA-Benchmarks.
    """

    def __init__(
        self,
        graphrag_pipeline: Any = None,
        consistency_orchestrator_factory: Optional[Callable] = None,
        answer_generator: Optional[Callable] = None
    ):
        """
        Args:
            graphrag_pipeline: GraphRAG-Pipeline
            consistency_orchestrator_factory: Factory-Funktion für Orchestrator
                mit Signatur: (stages: List[str]) -> Orchestrator
            answer_generator: Optionale Antwort-Generator-Funktion
        """
        self.pipeline = graphrag_pipeline
        self.orchestrator_factory = consistency_orchestrator_factory
        self.answer_generator = answer_generator
        self.benchmark_loader = BenchmarkLoader()

    def run_ablation(
        self,
        config: AblationConfig,
        progress_callback: Optional[Callable] = None
    ) -> AblationResult:
        """
        Führt vollständige Ablation Study durch.

        Args:
            config: Ablation-Konfiguration
            progress_callback: Callback für Fortschritt

        Returns:
            AblationResult mit allen Varianten
        """
        from datetime import datetime

        started_at = datetime.utcnow()
        logger.info(f"Starte Ablation Study: {config.benchmark.value} ({config.sample_size} Beispiele)")

        # Daten laden
        examples = self.benchmark_loader.load(
            config.benchmark,
            config.split,
            config.sample_size
        )

        if not examples:
            logger.error("Keine Beispiele geladen")
            return AblationResult(config=config)

        # Varianten evaluieren
        variant_results: List[VariantResult] = []
        baseline_result: Optional[EvaluationResult] = None

        for i, variant in enumerate(config.variants):
            logger.info(f"Evaluiere Variante {i+1}/{len(config.variants)}: {variant.value}")

            if progress_callback:
                progress_callback(variant.value, i, len(config.variants))

            # Evaluator für Variante konfigurieren
            evaluator = self._create_evaluator_for_variant(variant)

            # Mehrere Runs für statistische Signifikanz
            run_results = []
            for run in range(config.num_runs):
                eval_result = evaluator.evaluate(
                    examples,
                    use_consistency=(variant != AblationVariant.BASELINE)
                )
                run_results.append(eval_result)

            # Durchschnitt über Runs
            avg_result = self._average_results(run_results, config.benchmark.value)

            # Baseline speichern
            if variant == AblationVariant.BASELINE:
                baseline_result = avg_result

            # Deltas berechnen
            em_delta = 0.0
            f1_delta = 0.0
            overhead = 0.0

            if baseline_result:
                em_delta = avg_result.exact_match - baseline_result.exact_match
                f1_delta = avg_result.f1_score - baseline_result.f1_score
                if baseline_result.avg_latency_ms > 0:
                    overhead = (avg_result.avg_latency_ms - baseline_result.avg_latency_ms)
                    relative_overhead = overhead / baseline_result.avg_latency_ms
                else:
                    relative_overhead = 0.0

            variant_result = VariantResult(
                variant=variant,
                evaluation=avg_result,
                em_delta=em_delta,
                f1_delta=f1_delta,
                avg_overhead_ms=overhead,
                relative_overhead=relative_overhead if baseline_result else 0.0
            )
            variant_results.append(variant_result)

        # Beste Variante finden
        best_variant = None
        max_em = 0.0
        max_f1 = 0.0

        for vr in variant_results:
            if vr.em_delta > max_em:
                max_em = vr.em_delta
                best_variant = vr.variant
            if vr.f1_delta > max_f1:
                max_f1 = vr.f1_delta

        finished_at = datetime.utcnow()
        duration = (finished_at - started_at).total_seconds()

        result = AblationResult(
            config=config,
            variant_results=variant_results,
            best_variant=best_variant,
            max_improvement_em=max_em,
            max_improvement_f1=max_f1,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_seconds=duration
        )

        # Optional: Exportieren
        if config.export_path:
            self._export_result(result, config.export_path, config.export_format)

        logger.info(f"Ablation Study abgeschlossen in {duration:.1f}s")
        return result

    def _create_evaluator_for_variant(self, variant: AblationVariant) -> QAEvaluator:
        """Erstellt einen QAEvaluator für eine spezifische Variante."""
        consistency = None

        if self.orchestrator_factory and variant != AblationVariant.BASELINE:
            # Stufen basierend auf Variante auswählen
            if variant == AblationVariant.STAGE1_ONLY:
                stages = ["rule_based"]
            elif variant == AblationVariant.STAGE1_2:
                stages = ["rule_based", "embedding_based"]
            else:  # FULL
                stages = ["rule_based", "embedding_based", "llm_arbitration"]

            consistency = self.orchestrator_factory(stages)

        return QAEvaluator(
            graphrag_pipeline=self.pipeline,
            consistency_module=consistency,
            answer_generator=self.answer_generator
        )

    def _average_results(
        self,
        results: List[EvaluationResult],
        benchmark_name: str
    ) -> EvaluationResult:
        """Mittelt mehrere Evaluation-Ergebnisse."""
        if len(results) == 1:
            return results[0]

        avg_em = sum(r.exact_match for r in results) / len(results)
        avg_f1 = sum(r.f1_score for r in results) / len(results)
        avg_recall = sum(r.answer_recall for r in results) / len(results)
        avg_latency = sum(r.avg_latency_ms for r in results) / len(results)

        return EvaluationResult(
            benchmark_name=benchmark_name,
            num_examples=results[0].num_examples,
            exact_match=avg_em,
            f1_score=avg_f1,
            answer_recall=avg_recall,
            avg_latency_ms=avg_latency,
            total_time_seconds=sum(r.total_time_seconds for r in results),
        )

    def _export_result(self, result: AblationResult, path: str, format: str):
        """Exportiert das Ergebnis in verschiedenen Formaten."""
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Ergebnis exportiert nach: {path}")

        elif format == "latex":
            latex = self.generate_latex_table(result)
            with open(path, "w", encoding="utf-8") as f:
                f.write(latex)
            logger.info(f"LaTeX-Tabelle exportiert nach: {path}")

        elif format == "csv":
            self._export_csv(result, path)

    def _export_csv(self, result: AblationResult, path: str):
        """Exportiert als CSV."""
        try:
            import pandas as pd

            rows = []
            for vr in result.variant_results:
                rows.append({
                    "variant": vr.variant.value,
                    "exact_match": vr.evaluation.exact_match,
                    "f1_score": vr.evaluation.f1_score,
                    "answer_recall": vr.evaluation.answer_recall,
                    "em_delta": vr.em_delta,
                    "f1_delta": vr.f1_delta,
                    "avg_latency_ms": vr.evaluation.avg_latency_ms,
                    "overhead_ms": vr.avg_overhead_ms,
                })

            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
            logger.info(f"CSV exportiert nach: {path}")

        except ImportError:
            logger.error("pandas nicht verfügbar für CSV-Export")

    def generate_latex_table(self, result: AblationResult) -> str:
        """
        Generiert LaTeX-Tabelle für Masterarbeit.

        Args:
            result: AblationResult

        Returns:
            LaTeX-formatierter String
        """
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study: Einfluss der Konsistenzstufen auf QA-Performance}
\label{tab:ablation-study}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Konfiguration} & \textbf{EM} & \textbf{F1} & \textbf{$\Delta$ EM} & \textbf{$\Delta$ F1} & \textbf{Latenz (ms)} \\
\midrule
"""

        for vr in result.variant_results:
            variant_name = {
                AblationVariant.BASELINE: "Baseline (GraphRAG)",
                AblationVariant.STAGE1_ONLY: "+ Regelbasiert",
                AblationVariant.STAGE1_2: "+ Embedding",
                AblationVariant.FULL: "+ LLM (Vollständig)",
            }.get(vr.variant, vr.variant.value)

            # Delta-Formatierung
            em_delta = f"+{vr.em_delta:.1%}" if vr.em_delta > 0 else f"{vr.em_delta:.1%}" if vr.em_delta < 0 else "--"
            f1_delta = f"+{vr.f1_delta:.1%}" if vr.f1_delta > 0 else f"{vr.f1_delta:.1%}" if vr.f1_delta < 0 else "--"

            latex += f"{variant_name} & {vr.evaluation.exact_match:.1%} & {vr.evaluation.f1_score:.1%} & {em_delta} & {f1_delta} & {vr.evaluation.avg_latency_ms:.0f} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize{Benchmark: """ + result.config.benchmark.value + r""", n=""" + str(result.config.sample_size) + r"""}
\end{table}
"""
        return latex

    def run_quick_comparison(
        self,
        benchmark: BenchmarkType = BenchmarkType.HOTPOTQA,
        sample_size: int = 50
    ) -> AblationResult:
        """
        Schneller Vergleich für Tests.

        Args:
            benchmark: Zu verwendender Benchmark
            sample_size: Anzahl Beispiele

        Returns:
            AblationResult
        """
        config = AblationConfig(
            benchmark=benchmark,
            sample_size=sample_size,
            variants=[
                AblationVariant.BASELINE,
                AblationVariant.FULL,
            ],
            num_runs=1
        )

        return self.run_ablation(config)


def run_ablation_study(
    benchmark: str = "hotpotqa",
    sample_size: int = 100,
    export_path: Optional[str] = None
) -> AblationResult:
    """
    Convenience-Funktion für Ablation Study.

    Args:
        benchmark: "hotpotqa" oder "musique"
        sample_size: Anzahl Beispiele
        export_path: Optionaler Export-Pfad

    Returns:
        AblationResult
    """
    benchmark_type = BenchmarkType.HOTPOTQA if benchmark.lower() == "hotpotqa" else BenchmarkType.MUSIQUE

    config = AblationConfig(
        benchmark=benchmark_type,
        sample_size=sample_size,
        export_path=export_path,
        export_format="json" if export_path and export_path.endswith(".json") else "latex"
    )

    study = ConsistencyAblationStudy()
    return study.run_ablation(config)


if __name__ == "__main__":
    # Demo-Ausführung
    logging.basicConfig(level=logging.INFO)

    print("\n=== Ablation Study Demo ===\n")

    # Kleine Demo mit Mock-Daten
    study = ConsistencyAblationStudy()

    config = AblationConfig(
        benchmark=BenchmarkType.HOTPOTQA,
        sample_size=5,  # Nur 5 für Demo
        variants=[
            AblationVariant.BASELINE,
            AblationVariant.FULL,
        ]
    )

    print("Starte Ablation Study (Demo)...")
    print("Hinweis: Für echte Ergebnisse muss GraphRAG-Pipeline konfiguriert sein.")

    # Würde normalerweise result = study.run_ablation(config) aufrufen
    # Hier nur Struktur-Demo

    print("\nBeispiel LaTeX-Output:")
    example_result = AblationResult(
        config=config,
        variant_results=[
            VariantResult(
                variant=AblationVariant.BASELINE,
                evaluation=EvaluationResult(
                    benchmark_name="hotpotqa",
                    num_examples=100,
                    exact_match=0.42,
                    f1_score=0.55,
                    avg_latency_ms=150
                )
            ),
            VariantResult(
                variant=AblationVariant.FULL,
                evaluation=EvaluationResult(
                    benchmark_name="hotpotqa",
                    num_examples=100,
                    exact_match=0.48,
                    f1_score=0.62,
                    avg_latency_ms=280
                ),
                em_delta=0.06,
                f1_delta=0.07
            )
        ]
    )

    print(study.generate_latex_table(example_result))
    print(example_result.get_comparison_table())

    print("\n=== Demo abgeschlossen ===")
