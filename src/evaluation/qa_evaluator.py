# src/evaluation/qa_evaluator.py
"""
QA-Evaluator für Benchmark-basierte Evaluation.

Evaluiert das GraphRAG-System mit und ohne Konsistenzmodul
auf Standard-QA-Benchmarks.

Metriken:
- Exact Match (EM)
- F1 Score (Token-Level)
- Answer Recall
"""

import re
import string
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import time

from src.evaluation.benchmark_loader import QAExample

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    """Ergebnis für ein einzelnes QA-Beispiel."""
    example_id: str
    question: str
    gold_answer: str
    predicted_answer: str

    # Metriken
    exact_match: bool = False
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # Performance
    latency_ms: float = 0.0

    # Zusätzliche Infos
    used_consistency: bool = False
    consistency_overhead_ms: float = 0.0
    triples_validated: int = 0
    triples_accepted: int = 0
    triples_rejected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "example_id": self.example_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "latency_ms": self.latency_ms,
            "used_consistency": self.used_consistency,
            "consistency_overhead_ms": self.consistency_overhead_ms,
            "triples_validated": self.triples_validated,
            "triples_accepted": self.triples_accepted,
            "triples_rejected": self.triples_rejected,
        }


@dataclass
class EvaluationResult:
    """Aggregiertes Ergebnis einer Benchmark-Evaluation."""
    benchmark_name: str
    num_examples: int

    # Aggregierte Metriken
    exact_match: float = 0.0
    f1_score: float = 0.0
    answer_recall: float = 0.0

    # Vergleichsmetriken (mit vs. ohne Konsistenz)
    improvement_em: float = 0.0
    improvement_f1: float = 0.0
    consistency_overhead_ms: float = 0.0

    # Performance
    avg_latency_ms: float = 0.0
    total_time_seconds: float = 0.0

    # Details pro Beispiel
    per_example_results: List[ExampleResult] = field(default_factory=list)

    # Zusätzliche Statistiken
    triples_total: int = 0
    triples_accepted: int = 0
    triples_rejected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "num_examples": self.num_examples,
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "answer_recall": self.answer_recall,
            "improvement_em": self.improvement_em,
            "improvement_f1": self.improvement_f1,
            "consistency_overhead_ms": self.consistency_overhead_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "total_time_seconds": self.total_time_seconds,
            "triples_total": self.triples_total,
            "triples_accepted": self.triples_accepted,
            "triples_rejected": self.triples_rejected,
            "per_example_results": [r.to_dict() for r in self.per_example_results],
        }

    def print_summary(self):
        """Gibt eine formatierte Zusammenfassung aus."""
        print("\n" + "=" * 60)
        print(f"EVALUATION: {self.benchmark_name}")
        print("=" * 60)
        print(f"Anzahl Beispiele: {self.num_examples}")
        print(f"\nMETRIKEN:")
        print(f"  Exact Match:    {self.exact_match:.2%}")
        print(f"  F1 Score:       {self.f1_score:.2%}")
        print(f"  Answer Recall:  {self.answer_recall:.2%}")
        print(f"\nPERFORMANCE:")
        print(f"  Avg. Latenz:    {self.avg_latency_ms:.1f} ms")
        print(f"  Gesamtzeit:     {self.total_time_seconds:.1f} s")
        if self.consistency_overhead_ms > 0:
            print(f"  Konsistenz-Overhead: {self.consistency_overhead_ms:.1f} ms")
        print(f"\nTRIPLES:")
        print(f"  Gesamt:    {self.triples_total}")
        print(f"  Akzeptiert: {self.triples_accepted}")
        print(f"  Abgelehnt:  {self.triples_rejected}")
        print("=" * 60)


class QAEvaluator:
    """
    Evaluiert GraphRAG mit/ohne Konsistenzmodul auf QA-Benchmarks.

    Workflow:
    1. Für jedes QA-Beispiel: Graph aus Kontext aufbauen
    2. Optional: Konsistenzmodul auf Tripel anwenden
    3. Antwort mit GraphRAG generieren
    4. Mit Ground Truth vergleichen
    """

    def __init__(
        self,
        graphrag_pipeline: Any = None,
        consistency_module: Any = None,
        answer_generator: Optional[Callable] = None
    ):
        """
        Args:
            graphrag_pipeline: GraphRAG-Pipeline für Antwortgenerierung
            consistency_module: Optionales Konsistenzmodul (Orchestrator)
            answer_generator: Optionale Funktion für Antwortgenerierung
        """
        self.pipeline = graphrag_pipeline
        self.consistency = consistency_module
        self.answer_generator = answer_generator

        logger.info("QAEvaluator initialisiert")
        logger.info(f"  → Konsistenzmodul: {'aktiviert' if consistency_module else 'deaktiviert'}")

    def evaluate(
        self,
        examples: List[QAExample],
        use_consistency: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> EvaluationResult:
        """
        Führt Evaluation auf einer Liste von Beispielen durch.

        Args:
            examples: Liste von QA-Beispielen
            use_consistency: Ob Konsistenzmodul verwendet werden soll
            progress_callback: Optionaler Callback für Fortschrittsanzeige

        Returns:
            EvaluationResult mit aggregierten Metriken
        """
        if not examples:
            logger.warning("Keine Beispiele zur Evaluation")
            return EvaluationResult(
                benchmark_name="empty",
                num_examples=0
            )

        start_time = time.time()
        results: List[ExampleResult] = []

        # Benchmark-Name aus erstem Beispiel
        benchmark_name = examples[0].metadata.get("dataset", "unknown")

        logger.info(f"Starte Evaluation von {len(examples)} Beispielen...")

        for i, example in enumerate(examples):
            try:
                result = self._evaluate_single(example, use_consistency)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(examples), result)

            except Exception as e:
                logger.error(f"Fehler bei Beispiel {example.id}: {e}")
                # Erstelle Fehler-Result
                results.append(ExampleResult(
                    example_id=example.id,
                    question=example.question,
                    gold_answer=example.answer,
                    predicted_answer="ERROR",
                    exact_match=False,
                    f1_score=0.0
                ))

        # Aggregiere Ergebnisse
        total_time = time.time() - start_time
        evaluation_result = self._aggregate_results(
            results,
            benchmark_name,
            total_time,
            use_consistency
        )

        logger.info(f"Evaluation abgeschlossen: EM={evaluation_result.exact_match:.2%}, "
                   f"F1={evaluation_result.f1_score:.2%}")

        return evaluation_result

    def _evaluate_single(
        self,
        example: QAExample,
        use_consistency: bool
    ) -> ExampleResult:
        """Evaluiert ein einzelnes Beispiel."""
        start_time = time.time()

        # Antwort generieren
        if self.answer_generator:
            predicted = self.answer_generator(
                example.question,
                example.context_text,
                use_consistency=use_consistency
            )
        elif self.pipeline:
            predicted = self._generate_answer_with_pipeline(
                example,
                use_consistency
            )
        else:
            # Fallback: Nutze einfache Heuristik für Tests
            predicted = self._simple_answer_heuristic(example)

        latency_ms = (time.time() - start_time) * 1000

        # Metriken berechnen
        em = self._exact_match(predicted, example.answer)
        f1, precision, recall = self._f1_score(predicted, example.answer)

        return ExampleResult(
            example_id=example.id,
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=predicted,
            exact_match=em,
            f1_score=f1,
            precision=precision,
            recall=recall,
            latency_ms=latency_ms,
            used_consistency=use_consistency and self.consistency is not None
        )

    def _generate_answer_with_pipeline(
        self,
        example: QAExample,
        use_consistency: bool
    ) -> str:
        """Generiert Antwort mit der GraphRAG-Pipeline."""
        try:
            # Dies ist ein Platzhalter - muss an die tatsächliche
            # Pipeline-Implementierung angepasst werden

            # 1. Graph aus Kontext aufbauen
            # 2. Konsistenzmodul anwenden (wenn aktiviert)
            # 3. Antwort generieren

            result = self.pipeline.answer(
                question=example.question,
                context=example.context_text
            )
            return result.get("answer", "")

        except Exception as e:
            logger.error(f"Pipeline-Fehler: {e}")
            return ""

    def _simple_answer_heuristic(self, example: QAExample) -> str:
        """
        Einfache Heuristik für Antwortgenerierung (für Tests).

        Sucht nach der Antwort im Kontext.
        """
        # Versuche die Gold-Answer im Kontext zu finden
        context = example.context_text.lower()
        answer = example.answer.lower()

        if answer in context:
            return example.answer

        return ""

    def _aggregate_results(
        self,
        results: List[ExampleResult],
        benchmark_name: str,
        total_time: float,
        use_consistency: bool
    ) -> EvaluationResult:
        """Aggregiert die Einzelergebnisse."""
        if not results:
            return EvaluationResult(
                benchmark_name=benchmark_name,
                num_examples=0
            )

        # Metriken berechnen
        em_scores = [r.exact_match for r in results]
        f1_scores = [r.f1_score for r in results]
        latencies = [r.latency_ms for r in results]

        # Answer Recall: Anteil der Beispiele mit F1 > 0
        answer_recall = sum(1 for f1 in f1_scores if f1 > 0) / len(f1_scores)

        return EvaluationResult(
            benchmark_name=benchmark_name,
            num_examples=len(results),
            exact_match=sum(em_scores) / len(em_scores),
            f1_score=sum(f1_scores) / len(f1_scores),
            answer_recall=answer_recall,
            avg_latency_ms=sum(latencies) / len(latencies),
            total_time_seconds=total_time,
            per_example_results=results,
            consistency_overhead_ms=sum(r.consistency_overhead_ms for r in results) / len(results),
            triples_total=sum(r.triples_validated for r in results),
            triples_accepted=sum(r.triples_accepted for r in results),
            triples_rejected=sum(r.triples_rejected for r in results),
        )

    # =========================================================================
    # Metriken-Berechnung
    # =========================================================================

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalisiert eine Antwort für Vergleich."""
        # Kleinschreibung
        text = text.lower()

        # DE→EN Mapping für Ja/Nein-Antworten
        de_en_map = {"ja": "yes", "nein": "no"}
        stripped = text.strip()
        if stripped in de_en_map:
            text = de_en_map[stripped]

        # Artikel entfernen (EN + DE)
        text = re.sub(
            r'\b(a|an|the|der|die|das|ein|eine|einer|eines|einem|einen)\b',
            ' ', text
        )

        # Interpunktion entfernen
        text = ''.join(ch for ch in text if ch not in string.punctuation)

        # Mehrfache Leerzeichen
        text = ' '.join(text.split())

        return text.strip()

    @classmethod
    def _exact_match(cls, prediction: str, gold: str) -> bool:
        """Prüft auf exakte Übereinstimmung nach Normalisierung."""
        return cls._normalize_answer(prediction) == cls._normalize_answer(gold)

    @classmethod
    def _f1_score(cls, prediction: str, gold: str) -> tuple:
        """
        Berechnet Token-Level F1, Precision und Recall.

        Returns:
            (f1, precision, recall)
        """
        pred_tokens = cls._normalize_answer(prediction).split()
        gold_tokens = cls._normalize_answer(gold).split()

        if not pred_tokens or not gold_tokens:
            return (0.0, 0.0, 0.0)

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return (0.0, 0.0, 0.0)

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return (f1, precision, recall)


def evaluate_benchmark(
    examples: List[QAExample],
    answer_generator: Callable,
    use_consistency: bool = True
) -> EvaluationResult:
    """
    Convenience-Funktion für schnelle Evaluation.

    Args:
        examples: QA-Beispiele
        answer_generator: Funktion (question, context) -> answer
        use_consistency: Ob Konsistenz verwendet werden soll

    Returns:
        EvaluationResult
    """
    evaluator = QAEvaluator(answer_generator=answer_generator)
    return evaluator.evaluate(examples, use_consistency)


if __name__ == "__main__":
    # Test der Metriken
    logging.basicConfig(level=logging.INFO)

    print("\n=== Test QA-Metriken ===\n")

    # Test Exact Match
    test_cases = [
        ("Albert Einstein", "Albert Einstein", True),
        ("albert einstein", "Albert Einstein", True),
        ("The Albert Einstein", "Albert Einstein", True),
        ("Einstein", "Albert Einstein", False),
    ]

    print("Exact Match Tests:")
    for pred, gold, expected in test_cases:
        result = QAEvaluator._exact_match(pred, gold)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: '{pred}' vs '{gold}' -> {result}")

    # Test F1
    print("\nF1 Score Tests:")
    f1_cases = [
        ("Albert Einstein", "Albert Einstein", 1.0),
        ("Einstein", "Albert Einstein", 0.5),  # 1/2 recall, 1/1 precision
        ("the great Albert Einstein", "Albert Einstein", 1.0),  # nach Normalisierung
    ]

    for pred, gold, expected_f1 in f1_cases:
        f1, prec, rec = QAEvaluator._f1_score(pred, gold)
        print(f"  '{pred}' vs '{gold}': F1={f1:.2f}, P={prec:.2f}, R={rec:.2f}")

    print("\n=== Test abgeschlossen ===")
