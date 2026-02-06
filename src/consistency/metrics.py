# src/consistency/metrics.py
"""
Zentrales Metriken-Modul für wissenschaftliche Evaluation.

Erfasst alle relevanten Metriken für die Masterarbeit:
- Precision/Recall pro Validierungsstufe
- F1 für Entity Resolution
- Token-Verbrauch (LLM-Kosten)
- Latenz-Messungen
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metriken für eine einzelne Validierungsstufe."""
    stage_name: str
    total_processed: int = 0
    passed: int = 0
    failed: int = 0
    escalated: int = 0
    total_time_ms: float = 0.0

    # Für Stufe 3 (LLM):
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    # Konfidenz-Tracking
    confidence_values: List[float] = field(default_factory=list)

    @property
    def avg_time_ms(self) -> float:
        """Durchschnittliche Verarbeitungszeit."""
        if self.total_processed == 0:
            return 0.0
        return self.total_time_ms / self.total_processed

    @property
    def pass_rate(self) -> float:
        """Anteil der bestandenen Validierungen."""
        if self.total_processed == 0:
            return 0.0
        return self.passed / self.total_processed

    @property
    def fail_rate(self) -> float:
        """Anteil der fehlgeschlagenen Validierungen."""
        if self.total_processed == 0:
            return 0.0
        return self.failed / self.total_processed

    @property
    def escalation_rate(self) -> float:
        """Anteil der eskalierten Validierungen."""
        if self.total_processed == 0:
            return 0.0
        return self.escalated / self.total_processed

    @property
    def avg_confidence(self) -> float:
        """Durchschnittliche Konfidenz."""
        if not self.confidence_values:
            return 0.0
        return sum(self.confidence_values) / len(self.confidence_values)

    @property
    def avg_tokens_per_call(self) -> float:
        """Durchschnittliche Token-Anzahl pro LLM-Aufruf."""
        if self.llm_calls == 0:
            return 0.0
        return self.tokens_used / self.llm_calls

    def record(self, passed: bool, escalated: bool, time_ms: float, confidence: float):
        """Zeichnet ein Validierungsergebnis auf."""
        self.total_processed += 1
        self.total_time_ms += time_ms
        self.confidence_values.append(confidence)

        if passed:
            self.passed += 1
        else:
            self.failed += 1

        if escalated:
            self.escalated += 1

    def record_llm_usage(self, input_tokens: int, output_tokens: int):
        """Zeichnet LLM-Token-Verbrauch auf."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.tokens_used += input_tokens + output_tokens
        self.llm_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Export."""
        return {
            "stage_name": self.stage_name,
            "total_processed": self.total_processed,
            "passed": self.passed,
            "failed": self.failed,
            "escalated": self.escalated,
            "pass_rate": round(self.pass_rate, 4),
            "fail_rate": round(self.fail_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_confidence": round(self.avg_confidence, 4),
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.llm_calls,
            "avg_tokens_per_call": round(self.avg_tokens_per_call, 1),
        }


@dataclass
class EntityResolutionMetrics:
    """Metriken für Entity Resolution (Duplikaterkennung)."""
    total_comparisons: int = 0
    duplicates_found: int = 0
    merges_performed: int = 0

    # Für Precision/Recall Berechnung (wenn Ground Truth verfügbar)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Similarity-Scores
    similarity_scores: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def avg_similarity(self) -> float:
        """Durchschnittlicher Similarity-Score bei Duplikaten."""
        if not self.similarity_scores:
            return 0.0
        return sum(self.similarity_scores) / len(self.similarity_scores)

    def record_comparison(self, is_duplicate: bool, similarity: float):
        """Zeichnet einen Vergleich auf."""
        self.total_comparisons += 1
        if is_duplicate:
            self.duplicates_found += 1
            self.similarity_scores.append(similarity)

    def record_merge(self):
        """Zeichnet einen Merge auf."""
        self.merges_performed += 1

    def record_ground_truth(self, predicted_duplicate: bool, actual_duplicate: bool):
        """Zeichnet Ergebnis mit Ground Truth auf (für Evaluation)."""
        if predicted_duplicate and actual_duplicate:
            self.true_positives += 1
        elif predicted_duplicate and not actual_duplicate:
            self.false_positives += 1
        elif not predicted_duplicate and actual_duplicate:
            self.false_negatives += 1
        # True Negatives werden nicht explizit gezählt

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Export."""
        return {
            "total_comparisons": self.total_comparisons,
            "duplicates_found": self.duplicates_found,
            "merges_performed": self.merges_performed,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "avg_similarity": round(self.avg_similarity, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class LLMUsageStats:
    """Token-Verbrauch für Kostenanalyse."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class ConsistencyMetrics:
    """Zentrale Metriken-Sammlung für wissenschaftliche Evaluation."""

    # Metriken pro Stufe
    stage1_metrics: StageMetrics = field(default_factory=lambda: StageMetrics("rule_based"))
    stage2_metrics: StageMetrics = field(default_factory=lambda: StageMetrics("embedding_based"))
    stage3_metrics: StageMetrics = field(default_factory=lambda: StageMetrics("llm_arbitration"))

    # Entity Resolution Metriken
    entity_resolution: EntityResolutionMetrics = field(default_factory=EntityResolutionMetrics)

    # Konfidenz-Verteilung (für Histogramme)
    confidence_distribution: Dict[str, List[float]] = field(default_factory=lambda: {
        "stage1": [],
        "stage2": [],
        "stage3": [],
        "final": []
    })

    # Gesamtstatistiken
    total_triples_processed: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_needs_review: int = 0
    total_processing_time_ms: float = 0.0

    # Zeitstempel
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # LLM-Nutzung aggregiert
    llm_usage_history: List[LLMUsageStats] = field(default_factory=list)

    def start(self):
        """Startet die Metriken-Erfassung."""
        self.started_at = datetime.utcnow()
        logger.info("Metriken-Erfassung gestartet")

    def finish(self):
        """Beendet die Metriken-Erfassung."""
        self.finished_at = datetime.utcnow()
        logger.info("Metriken-Erfassung beendet")

    @property
    def duration_seconds(self) -> float:
        """Gesamtdauer in Sekunden."""
        if not self.started_at or not self.finished_at:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def acceptance_rate(self) -> float:
        """Gesamte Akzeptanzrate."""
        if self.total_triples_processed == 0:
            return 0.0
        return self.total_accepted / self.total_triples_processed

    @property
    def rejection_rate(self) -> float:
        """Gesamte Ablehnungsrate."""
        if self.total_triples_processed == 0:
            return 0.0
        return self.total_rejected / self.total_triples_processed

    @property
    def review_rate(self) -> float:
        """Anteil, der manuell geprüft werden muss."""
        if self.total_triples_processed == 0:
            return 0.0
        return self.total_needs_review / self.total_triples_processed

    @property
    def total_llm_tokens(self) -> int:
        """Gesamte LLM-Token-Nutzung."""
        return sum(usage.total_tokens for usage in self.llm_usage_history)

    @property
    def total_llm_cost_usd(self) -> float:
        """
        Geschätzte LLM-Kosten (basierend auf GPT-4 Preisen).

        Preise (Stand 2024):
        - GPT-4 Turbo: $0.01/1K input, $0.03/1K output
        """
        input_tokens = sum(usage.input_tokens for usage in self.llm_usage_history)
        output_tokens = sum(usage.output_tokens for usage in self.llm_usage_history)

        # GPT-4 Turbo Preise
        input_cost = (input_tokens / 1000) * 0.01
        output_cost = (output_tokens / 1000) * 0.03

        return input_cost + output_cost

    def record_stage_result(
        self,
        stage: str,
        passed: bool,
        escalated: bool,
        time_ms: float,
        confidence: float
    ):
        """Zeichnet das Ergebnis einer Validierungsstufe auf."""
        if stage == "rule_based":
            metrics = self.stage1_metrics
            self.confidence_distribution["stage1"].append(confidence)
        elif stage == "embedding_based":
            metrics = self.stage2_metrics
            self.confidence_distribution["stage2"].append(confidence)
        elif stage == "llm_arbitration":
            metrics = self.stage3_metrics
            self.confidence_distribution["stage3"].append(confidence)
        else:
            logger.warning(f"Unbekannte Stufe: {stage}")
            return

        metrics.record(passed, escalated, time_ms, confidence)

    def record_llm_usage(self, usage: LLMUsageStats):
        """Zeichnet LLM-Nutzung auf."""
        self.llm_usage_history.append(usage)
        self.stage3_metrics.record_llm_usage(usage.input_tokens, usage.output_tokens)

    def record_final_result(
        self,
        accepted: bool,
        rejected: bool,
        needs_review: bool,
        final_confidence: float,
        total_time_ms: float
    ):
        """Zeichnet das finale Ergebnis auf."""
        self.total_triples_processed += 1
        self.total_processing_time_ms += total_time_ms
        self.confidence_distribution["final"].append(final_confidence)

        if accepted:
            self.total_accepted += 1
        elif rejected:
            self.total_rejected += 1
        elif needs_review:
            self.total_needs_review += 1

    def compute_precision_recall(self) -> Dict[str, Dict[str, float]]:
        """
        Berechnet Precision und Recall für jede Stufe.

        Hinweis: Erfordert Ground Truth für aussagekräftige Werte.
        Ohne Ground Truth werden Schätzungen basierend auf
        Konfidenz-Schwellenwerten verwendet.
        """
        return {
            "stage1": {
                "pass_rate": self.stage1_metrics.pass_rate,
                "fail_rate": self.stage1_metrics.fail_rate,
                "avg_confidence": self.stage1_metrics.avg_confidence,
            },
            "stage2": {
                "pass_rate": self.stage2_metrics.pass_rate,
                "fail_rate": self.stage2_metrics.fail_rate,
                "avg_confidence": self.stage2_metrics.avg_confidence,
            },
            "stage3": {
                "pass_rate": self.stage3_metrics.pass_rate,
                "fail_rate": self.stage3_metrics.fail_rate,
                "avg_confidence": self.stage3_metrics.avg_confidence,
            },
            "entity_resolution": {
                "precision": self.entity_resolution.precision,
                "recall": self.entity_resolution.recall,
                "f1_score": self.entity_resolution.f1_score,
            },
            "overall": {
                "acceptance_rate": self.acceptance_rate,
                "rejection_rate": self.rejection_rate,
                "review_rate": self.review_rate,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert alle Metriken zu einem Dictionary."""
        return {
            "metadata": {
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "summary": {
                "total_triples_processed": self.total_triples_processed,
                "total_accepted": self.total_accepted,
                "total_rejected": self.total_rejected,
                "total_needs_review": self.total_needs_review,
                "acceptance_rate": round(self.acceptance_rate, 4),
                "rejection_rate": round(self.rejection_rate, 4),
                "review_rate": round(self.review_rate, 4),
                "total_processing_time_ms": round(self.total_processing_time_ms, 2),
                "avg_processing_time_ms": round(
                    self.total_processing_time_ms / max(self.total_triples_processed, 1), 2
                ),
            },
            "stages": {
                "stage1_rule_based": self.stage1_metrics.to_dict(),
                "stage2_embedding": self.stage2_metrics.to_dict(),
                "stage3_llm": self.stage3_metrics.to_dict(),
            },
            "entity_resolution": self.entity_resolution.to_dict(),
            "llm_usage": {
                "total_tokens": self.total_llm_tokens,
                "total_input_tokens": sum(u.input_tokens for u in self.llm_usage_history),
                "total_output_tokens": sum(u.output_tokens for u in self.llm_usage_history),
                "total_calls": len(self.llm_usage_history),
                "estimated_cost_usd": round(self.total_llm_cost_usd, 4),
            },
            "precision_recall": self.compute_precision_recall(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Exportiert als JSON-String."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_dataframe(self):
        """
        Exportiert als pandas DataFrame für wissenschaftliche Analyse.

        Returns:
            pd.DataFrame mit allen Metriken als Zeilen
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas nicht installiert - DataFrame-Export nicht verfügbar")
            return None

        data = self.to_dict()

        # Flatten nested dict für DataFrame
        rows = []

        # Summary-Metriken
        for key, value in data["summary"].items():
            rows.append({"category": "summary", "metric": key, "value": value})

        # Stage-Metriken
        for stage_name, stage_data in data["stages"].items():
            for key, value in stage_data.items():
                rows.append({"category": stage_name, "metric": key, "value": value})

        # Entity Resolution
        for key, value in data["entity_resolution"].items():
            rows.append({"category": "entity_resolution", "metric": key, "value": value})

        # LLM Usage
        for key, value in data["llm_usage"].items():
            rows.append({"category": "llm_usage", "metric": key, "value": value})

        return pd.DataFrame(rows)

    def export(self, path: str, format: str = "json"):
        """
        Exportiert Metriken in eine Datei.

        Args:
            path: Dateipfad
            format: "json" oder "csv"
        """
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
            logger.info(f"Metriken exportiert nach: {path}")

        elif format == "csv":
            df = self.to_dataframe()
            if df is not None:
                df.to_csv(path, index=False)
                logger.info(f"Metriken exportiert nach: {path}")
            else:
                logger.error("CSV-Export fehlgeschlagen - pandas nicht verfügbar")

        else:
            raise ValueError(f"Unbekanntes Format: {format}")

    def get_confidence_histogram(self, bins: int = 10) -> Dict[str, List[int]]:
        """
        Erstellt Histogramm-Daten für Konfidenz-Verteilung.

        Args:
            bins: Anzahl der Bins

        Returns:
            Dict mit Bin-Counts pro Stufe
        """
        try:
            import numpy as np
        except ImportError:
            logger.error("numpy nicht installiert - Histogramm nicht verfügbar")
            return {}

        result = {}

        for stage, values in self.confidence_distribution.items():
            if values:
                counts, _ = np.histogram(values, bins=bins, range=(0, 1))
                result[stage] = counts.tolist()
            else:
                result[stage] = [0] * bins

        return result

    def reset(self):
        """Setzt alle Metriken zurück."""
        self.stage1_metrics = StageMetrics("rule_based")
        self.stage2_metrics = StageMetrics("embedding_based")
        self.stage3_metrics = StageMetrics("llm_arbitration")
        self.entity_resolution = EntityResolutionMetrics()
        self.confidence_distribution = {
            "stage1": [],
            "stage2": [],
            "stage3": [],
            "final": []
        }
        self.total_triples_processed = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_needs_review = 0
        self.total_processing_time_ms = 0.0
        self.started_at = None
        self.finished_at = None
        self.llm_usage_history = []
        logger.info("Metriken zurückgesetzt")


def create_metrics() -> ConsistencyMetrics:
    """Factory-Funktion für ConsistencyMetrics."""
    metrics = ConsistencyMetrics()
    metrics.start()
    return metrics
