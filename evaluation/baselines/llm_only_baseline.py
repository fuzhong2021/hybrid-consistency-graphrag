#!/usr/bin/env python3
# evaluation/baselines/llm_only_baseline.py
"""
LLM-Only Baseline für Konsistenzprüfung.

Verwendet direkt das LLM für jedes Triple ohne Vorfilterung.
Zeigt den Trade-off zwischen Genauigkeit und Kosten/Latenz.

Wissenschaftliche Grundlage:
- OpenAI GPT-4 für Fact Verification (OpenAI, 2023)
- Thorne et al. (2018): FEVER Benchmark
- Chen et al. (2023): LLMs as Knowledge Base Validators

Wichtig:
- Teuer (jedes Triple benötigt LLM-Aufruf)
- Langsam (hohe Latenz)
- Aber möglicherweise höchste Genauigkeit
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.models.entities import Triple, ValidationStatus
from src.consistency.base import ConsistencyConfig, ValidationOutcome
from src.consistency.llm_arbitrator import LLMArbitrator
from src.consistency.metrics import LLMUsageStats

logger = logging.getLogger(__name__)


@dataclass
class LLMOnlyConfig:
    """Konfiguration für LLM-Only Baseline."""
    llm_model: str = "llama3.1:8b"  # Oder "gpt-4-turbo"
    acceptance_threshold: float = 0.6  # Minimum LLM-Konfidenz für Akzeptanz
    max_retries: int = 2
    track_costs: bool = True


@dataclass
class BaselineResult:
    """Ergebnis einer Baseline-Validierung."""
    accepted: bool
    confidence: float
    processing_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class LLMOnlyBaseline:
    """
    LLM-Only Baseline: Direkte LLM-Validierung für jedes Triple.

    Implementiert:
    - Direkte LLM-Abfrage für Triple-Validierung
    - Keine Vorfilterung durch Regeln oder Embeddings
    - Token-Tracking für Kostenanalyse

    Trade-offs:
    + Möglicherweise höchste Genauigkeit
    + Nutzt Weltwissen des LLM
    - Sehr teuer bei großen Datasets
    - Hohe Latenz
    - Weniger reproduzierbar
    """

    def __init__(
        self,
        config: LLMOnlyConfig = None,
        consistency_config: ConsistencyConfig = None,
        llm_client: Any = None
    ):
        self.config = config or LLMOnlyConfig()
        self.consistency_config = consistency_config or ConsistencyConfig(
            llm_model=self.config.llm_model,
            max_llm_retries=self.config.max_retries
        )

        if llm_client is None:
            logger.warning("Kein LLM-Client - LLMOnlyBaseline wird nicht funktionieren")

        # Initialisiere LLM Arbitrator
        self.llm_usage_stats: List[LLMUsageStats] = []

        def metrics_callback(usage: LLMUsageStats):
            self.llm_usage_stats.append(usage)

        self.validator = LLMArbitrator(
            self.consistency_config,
            llm_client,
            metrics_callback=metrics_callback if self.config.track_costs else None
        )
        self.llm_client = llm_client

        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "uncertain": 0,
            "total_time_ms": 0.0,
            "total_tokens": 0,
            "llm_calls": 0,
        }

        logger.info(f"LLMOnlyBaseline initialisiert (model={self.config.llm_model})")

    def validate(self, triple: Triple, graph_repo: Any = None) -> BaselineResult:
        """
        Validiert ein Triple direkt mit LLM.

        Args:
            triple: Das zu validierende Triple
            graph_repo: Graph-Repository (für Kontext, optional)

        Returns:
            BaselineResult mit LLM-basierter Entscheidung
        """
        start_time = time.time()

        if self.llm_client is None:
            # Fallback wenn kein LLM verfügbar
            return BaselineResult(
                accepted=False,
                confidence=0.0,
                processing_time_ms=0.0,
                details={"error": "No LLM client available"}
            )

        # LLM Validierung
        result = self.validator.validate(triple, graph_repo)

        # Entscheidung basierend auf LLM-Ergebnis
        if result.outcome == ValidationOutcome.PASS:
            accepted = result.confidence >= self.config.acceptance_threshold
        elif result.outcome == ValidationOutcome.FAIL:
            accepted = False
        else:  # UNCERTAIN
            # Bei Unsicherheit: Konservativ ablehnen
            accepted = False
            self.stats["uncertain"] += 1

        confidence = result.confidence
        processing_time_ms = (time.time() - start_time) * 1000

        # Statistiken
        self.stats["total"] += 1
        self.stats["llm_calls"] += 1
        if accepted:
            self.stats["accepted"] += 1
            triple.validation_status = ValidationStatus.ACCEPTED
        else:
            self.stats["rejected"] += 1
            triple.validation_status = ValidationStatus.REJECTED
        self.stats["total_time_ms"] += processing_time_ms

        # Token-Tracking
        if self.llm_usage_stats:
            latest = self.llm_usage_stats[-1]
            self.stats["total_tokens"] += latest.total_tokens

        return BaselineResult(
            accepted=accepted,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            details={
                "baseline": "llm_only",
                "llm_outcome": result.outcome.value,
                "llm_model": self.config.llm_model,
                "tokens_used": self.llm_usage_stats[-1].total_tokens if self.llm_usage_stats else 0,
                "llm_details": result.details,
            }
        )

    def validate_batch(self, triples: List[Triple], graph_repo: Any = None) -> List[BaselineResult]:
        """
        Validiert mehrere Triples.

        Hinweis: Jedes Triple benötigt einen separaten LLM-Aufruf.
        Bei großen Batches kann das sehr teuer werden!
        """
        results = []
        for i, triple in enumerate(triples):
            if i > 0 and i % 10 == 0:
                logger.info(f"LLMOnlyBaseline: {i}/{len(triples)} verarbeitet, "
                           f"{self.stats['total_tokens']} Tokens verwendet")
            results.append(self.validate(triple, graph_repo))
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        total = self.stats["total"]

        # Kostenberechnung (GPT-4 Turbo Preise)
        input_tokens = sum(u.input_tokens for u in self.llm_usage_stats)
        output_tokens = sum(u.output_tokens for u in self.llm_usage_stats)
        estimated_cost = (input_tokens / 1000) * 0.01 + (output_tokens / 1000) * 0.03

        return {
            "baseline_type": "llm_only",
            "config": {
                "llm_model": self.config.llm_model,
                "acceptance_threshold": self.config.acceptance_threshold,
            },
            "total_processed": total,
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "uncertain": self.stats["uncertain"],
            "acceptance_rate": self.stats["accepted"] / total if total > 0 else 0,
            "avg_time_ms": self.stats["total_time_ms"] / total if total > 0 else 0,
            "llm_calls": self.stats["llm_calls"],
            "total_tokens": self.stats["total_tokens"],
            "estimated_cost_usd": round(estimated_cost, 4),
            "avg_tokens_per_triple": self.stats["total_tokens"] / total if total > 0 else 0,
        }

    def reset(self):
        """Setzt Statistiken zurück."""
        self.stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "uncertain": 0,
            "total_time_ms": 0.0,
            "total_tokens": 0,
            "llm_calls": 0,
        }
        self.llm_usage_stats = []


def create_llm_only_baseline(
    llm_client: Any = None,
    llm_model: str = "llama3.1:8b",
    consistency_config: ConsistencyConfig = None
) -> LLMOnlyBaseline:
    """Factory-Funktion für LLMOnlyBaseline."""
    return LLMOnlyBaseline(
        config=LLMOnlyConfig(llm_model=llm_model),
        consistency_config=consistency_config,
        llm_client=llm_client
    )
