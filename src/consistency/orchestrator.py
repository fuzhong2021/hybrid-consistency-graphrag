# src/consistency/orchestrator.py
"""
Konsistenz-Orchestrator

Koordiniert die dreistufige Validierung und implementiert
konfidenz-basiertes Routing zwischen den Stufen.

Erweitert um:
- Zentrales Metriken-Tracking für wissenschaftliche Evaluation
- Export-Funktionen für Analyse
- Konfidenz-basierte Batch-Optimierung
- Provenance-Tracking für Source-Reliability-Lernen
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from src.models.entities import Triple, ValidationStatus
from src.consistency.base import (
    ConsistencyConfig, ValidationStage, StageResult, ValidationOutcome
)
from src.consistency.rules.rule_validator import RuleBasedValidator
from src.consistency.embedding_validator import EmbeddingValidator
from src.consistency.llm_arbitrator import LLMArbitrator
from src.consistency.metrics import ConsistencyMetrics, LLMUsageStats, create_metrics
from src.consistency.provenance import ProvenanceTracker, ProvenanceConfig
from src.consistency.semantic_trigger import SemanticTriggerAnalyzer, SemanticTriggerConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistiken über die Verarbeitung."""
    total_processed: int = 0
    stage1_passed: int = 0
    stage2_required: int = 0
    stage3_required: int = 0
    stage3_semantic_triggered: int = 0  # Durch semantischen Trigger ausgelöst
    accepted: int = 0
    rejected: int = 0
    needs_review: int = 0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.accepted / self.total_processed

    @property
    def escalation_rate(self) -> float:
        if self.stage1_passed == 0:
            return 0.0
        return (self.stage2_required + self.stage3_required) / self.stage1_passed


class ConsistencyOrchestrator:
    """
    Zentraler Orchestrator für die dreistufige Konsistenzprüfung.

    Routing-Logik:
    ─────────────
    1. Stufe 1 (Rules) → Bei PASS mit conf ≥ 0.9: Direkt ACCEPTED
    2. Stufe 2 (Embedding) → Bei PASS mit conf ≥ 0.7: ACCEPTED
    3. Stufe 3 (LLM) → Finale Entscheidung oder REVIEW

    Features:
    - Zentrales Metriken-Tracking für wissenschaftliche Evaluation
    - Export-Funktionen für JSON/CSV
    - LLM-Token-Tracking für Kostenanalyse
    """

    def __init__(
        self,
        config: ConsistencyConfig,
        graph_repo: Any = None,
        embedding_model: Any = None,
        llm_client: Any = None,
        enable_metrics: bool = True,
        always_check_duplicates: bool = True
    ):
        """
        Initialisiert den Orchestrator mit allen drei Stufen.

        Args:
            config: Konsistenz-Konfiguration
            graph_repo: Neo4j Repository
            embedding_model: Embedding-Modell für Stufe 2
            llm_client: LLM-Client für Stufe 3
            enable_metrics: Aktiviert detailliertes Metriken-Tracking
            always_check_duplicates: Wenn True, wird Stufe 2 immer ausgeführt
                                     (für Duplikaterkennung auch bei hoher Stufe-1-Konfidenz)
        """
        self.config = config
        self.graph_repo = graph_repo
        self.enable_metrics = enable_metrics
        self.always_check_duplicates = always_check_duplicates
        self._has_embedding_model = embedding_model is not None

        # Metriken-System initialisieren
        self.metrics = create_metrics() if enable_metrics else None

        # Provenance-Tracker initialisieren (geteilte Instanz für alle Stufen)
        self.provenance_tracker = ProvenanceTracker(config=ProvenanceConfig(
            enable_source_learning=True,
            enable_corroboration=True,
            # #12: Missing Source Penalty Settings von ConsistencyConfig übernehmen
            enable_missing_source_penalty=config.enable_missing_source_penalty,
            missing_source_penalty=config.missing_source_penalty,
        ))

        # Callback für LLM-Token-Tracking
        def llm_metrics_callback(usage: LLMUsageStats):
            if self.metrics:
                self.metrics.record_llm_usage(usage)

        # Stufen initialisieren
        self.stage1 = RuleBasedValidator(config, embedding_model)
        self.stage2 = EmbeddingValidator(config, embedding_model)
        self.stage3 = LLMArbitrator(
            config,
            llm_client,
            metrics_callback=llm_metrics_callback if enable_metrics else None
        )

        # Teile den Provenance-Tracker mit Stage 2
        self.stage2._provenance_tracker = self.provenance_tracker

        # Semantischer Trigger für selektive LLM-Aufrufe
        self.semantic_trigger = SemanticTriggerAnalyzer(
            config=SemanticTriggerConfig(
                enable_semantic_trigger=config.enable_semantic_trigger,
                low_confidence_threshold=config.semantic_trigger_low_confidence,
            )
        )

        # Legacy-Statistiken (für Abwärtskompatibilität)
        self.stats = ProcessingStats()

        logger.info("ConsistencyOrchestrator initialisiert")
        logger.info(f"  → High Confidence Threshold: {config.high_confidence_threshold}")
        logger.info(f"  → Medium Confidence Threshold: {config.medium_confidence_threshold}")
        logger.info(f"  → Metriken-Tracking: {'aktiviert' if enable_metrics else 'deaktiviert'}")
        logger.info(f"  → Provenance-Tracking: aktiviert")
        logger.info(f"  → Missing Source Penalty: {'aktiviert' if config.enable_missing_source_penalty else 'deaktiviert'}")
        logger.info(f"  → Semantischer Trigger: {'aktiviert' if config.enable_semantic_trigger else 'deaktiviert'}")
    
    def process(self, triple: Triple) -> Triple:
        """
        Verarbeitet ein Triple durch alle notwendigen Validierungsstufen.

        Args:
            triple: Das zu validierende Triple

        Returns:
            Das Triple mit aktualisiertem Validierungsstatus
        """
        start_time = time.time()
        self.stats.total_processed += 1

        logger.info(f"━━━ Verarbeite: {triple} ━━━")

        final_confidence = 1.0
        stages_passed = []

        # ══════════════════════════════════════════════════════════════════
        # STUFE 1: Regelbasierte Validierung
        # ══════════════════════════════════════════════════════════════════
        result1 = self.stage1.validate(triple, self.graph_repo)
        triple.add_validation_event("rule_based", result1.outcome == ValidationOutcome.PASS,
                                    result1.confidence, result1.details)

        # Metriken aufzeichnen
        if self.metrics:
            self.metrics.record_stage_result(
                stage="rule_based",
                passed=result1.outcome == ValidationOutcome.PASS,
                escalated=result1.should_escalate,
                time_ms=result1.processing_time_ms,
                confidence=result1.confidence
            )

        if result1.outcome == ValidationOutcome.FAIL:
            triple.validation_status = ValidationStatus.REJECTED
            triple.conflicts.extend(result1.conflicts)
            self.stats.rejected += 1
            # Provenance: Aufzeichnen dass Triple abgelehnt wurde
            self.provenance_tracker.record_triple(
                triple, accepted=False, caused_conflict=len(result1.conflicts) > 0
            )
            self._finalize_metrics(triple, start_time, result1.confidence)
            self._log_result(triple, "Stufe 1 FAIL", start_time)
            return triple

        self.stats.stage1_passed += 1
        stages_passed.append("stage1")
        final_confidence = result1.confidence

        # Fast-Path: Hohe Konfidenz → Direkt akzeptieren
        # ABER: Wenn always_check_duplicates und (embedding_model ODER graph_repo) vorhanden,
        # trotzdem Stufe 2 ausführen für:
        # - Duplikaterkennung (Name-basiert oder Embedding-basiert)
        # - Provenance-Boost (#7), Anomalie-Erkennung (#8), TransE (#10)
        skip_stage2 = not (
            self.always_check_duplicates and
            (self._has_embedding_model or self.graph_repo is not None)
        )

        logger.debug(f"  → skip_stage2={skip_stage2}, always_check={self.always_check_duplicates}, "
                    f"has_emb={self._has_embedding_model}, has_repo={self.graph_repo is not None}")

        if (result1.outcome == ValidationOutcome.PASS and
            result1.confidence >= self.config.high_confidence_threshold and
            skip_stage2):
            triple.validation_status = ValidationStatus.ACCEPTED
            self.stats.accepted += 1
            # Provenance: Aufzeichnen dass Triple akzeptiert wurde
            self.provenance_tracker.record_triple(triple, accepted=True, caused_conflict=False)
            self._finalize_metrics(triple, start_time, final_confidence)
            self._log_result(triple, "Stufe 1 HIGH CONF", start_time)
            return triple

        # Wenn wir hier sind, geht es weiter zu Stufe 2
        logger.debug(f"  → Weiter zu Stufe 2 (skip_stage2={skip_stage2}, conf={result1.confidence})")

        # ══════════════════════════════════════════════════════════════════
        # STUFE 2: Embedding-basierte Prüfung
        # ══════════════════════════════════════════════════════════════════
        self.stats.stage2_required += 1

        result2 = self.stage2.validate(triple, self.graph_repo)
        triple.add_validation_event("embedding_based", result2.outcome == ValidationOutcome.PASS,
                                    result2.confidence, result2.details)
        triple.conflicts.extend(result2.conflicts)

        # Metriken aufzeichnen
        if self.metrics:
            self.metrics.record_stage_result(
                stage="embedding_based",
                passed=result2.outcome == ValidationOutcome.PASS,
                escalated=result2.should_escalate,
                time_ms=result2.processing_time_ms,
                confidence=result2.confidence
            )

            # Entity Resolution Metriken
            if result2.details.get("subject_duplicates_found", 0) > 0:
                self.metrics.entity_resolution.record_comparison(
                    is_duplicate=True,
                    similarity=result2.details.get("similarity_threshold", 0.85)
                )
            if result2.details.get("object_duplicates_found", 0) > 0:
                self.metrics.entity_resolution.record_comparison(
                    is_duplicate=True,
                    similarity=result2.details.get("similarity_threshold", 0.85)
                )

        # Entity Resolution: Duplikate auflösen
        if self.graph_repo:
            try:
                sub_result = self.stage2.resolve_entity(triple.subject, self.graph_repo)
                if sub_result.is_duplicate and sub_result.canonical_entity:
                    logger.debug(f"  Entity Resolution: '{triple.subject.name}' → "
                                f"'{sub_result.canonical_entity.name}'")
                    triple.subject = sub_result.canonical_entity
                    if self.metrics:
                        self.metrics.entity_resolution.record_merge()
            except Exception as e:
                logger.debug(f"  Entity Resolution Subject fehlgeschlagen: {e}")

            try:
                obj_result = self.stage2.resolve_entity(triple.object, self.graph_repo)
                if obj_result.is_duplicate and obj_result.canonical_entity:
                    logger.debug(f"  Entity Resolution: '{triple.object.name}' → "
                                f"'{obj_result.canonical_entity.name}'")
                    triple.object = obj_result.canonical_entity
                    if self.metrics:
                        self.metrics.entity_resolution.record_merge()
            except Exception as e:
                logger.debug(f"  Entity Resolution Object fehlgeschlagen: {e}")

        # Kombinierte Konfidenz
        combined_confidence = result1.confidence * result2.confidence
        final_confidence = combined_confidence
        stages_passed.append("stage2")

        # ══════════════════════════════════════════════════════════════════
        # SEMANTISCHER TRIGGER: Prüfe ob LLM trotz hoher Konfidenz nötig ist
        # ══════════════════════════════════════════════════════════════════
        trigger_result = self.semantic_trigger.should_trigger_llm(
            triple, self.graph_repo, combined_confidence
        )

        # Entscheidung: Stage 3 aufrufen WENN:
        # 1. Semantischer Trigger aktiv (Rollen-Konflikt erkannt), ODER
        # 2. Stage 2 nicht bestanden, ODER
        # 3. Konfidenz zu niedrig
        force_stage3 = trigger_result.should_trigger
        normal_escalation = (
            result2.outcome != ValidationOutcome.PASS or
            combined_confidence < self.config.medium_confidence_threshold
        )

        if not force_stage3 and not normal_escalation:
            # Akzeptieren: Keine Konflikte, genug Konfidenz, kein Trigger
            triple.validation_status = ValidationStatus.ACCEPTED
            self.stats.accepted += 1
            # Provenance: Aufzeichnen dass Triple akzeptiert wurde
            self.provenance_tracker.record_triple(
                triple, accepted=True, caused_conflict=len(result2.conflicts) > 0
            )
            self._finalize_metrics(triple, start_time, final_confidence)
            self._log_result(triple, "Stufe 2 PASS", start_time)
            return triple

        # ══════════════════════════════════════════════════════════════════
        # STUFE 3: LLM-Arbitration
        # ══════════════════════════════════════════════════════════════════
        self.stats.stage3_required += 1

        # Tracken ob durch semantischen Trigger ausgelöst
        if force_stage3:
            self.stats.stage3_semantic_triggered += 1
            logger.info(f"  → Stage 3 durch semantischen Trigger: {trigger_result.reason}")

        result3 = self.stage3.validate(triple, self.graph_repo)
        triple.add_validation_event("llm_arbitration", result3.outcome == ValidationOutcome.PASS,
                                    result3.confidence, result3.details)

        # Metriken aufzeichnen
        if self.metrics:
            self.metrics.record_stage_result(
                stage="llm_arbitration",
                passed=result3.outcome == ValidationOutcome.PASS,
                escalated=result3.outcome == ValidationOutcome.UNCERTAIN,
                time_ms=result3.processing_time_ms,
                confidence=result3.confidence
            )

        final_confidence = combined_confidence * result3.confidence
        stages_passed.append("stage3")

        # Finale Entscheidung
        had_conflicts = len(triple.conflicts) > 0 or len(result3.conflicts) > 0
        if result3.outcome == ValidationOutcome.PASS:
            triple.validation_status = ValidationStatus.ACCEPTED
            self.stats.accepted += 1
            # Provenance: Aufzeichnen dass Triple akzeptiert wurde
            self.provenance_tracker.record_triple(
                triple, accepted=True, caused_conflict=had_conflicts
            )
        elif result3.outcome == ValidationOutcome.FAIL:
            triple.validation_status = ValidationStatus.REJECTED
            self.stats.rejected += 1
            # Provenance: Aufzeichnen dass Triple abgelehnt wurde
            self.provenance_tracker.record_triple(
                triple, accepted=False, caused_conflict=had_conflicts
            )
        else:  # UNCERTAIN
            triple.validation_status = ValidationStatus.NEEDS_REVIEW
            self.stats.needs_review += 1
            # Provenance: Als akzeptiert aufzeichnen (needs_review = manuell prüfen)
            self.provenance_tracker.record_triple(
                triple, accepted=True, caused_conflict=True
            )

        self._finalize_metrics(triple, start_time, final_confidence)
        self._log_result(triple, f"Stufe 3 {result3.outcome.value}", start_time)
        return triple

    def _finalize_metrics(self, triple: Triple, start_time: float, final_confidence: float):
        """Finalisiert die Metriken für ein verarbeitetes Triple."""
        if not self.metrics:
            return

        total_time = (time.time() - start_time) * 1000

        self.metrics.record_final_result(
            accepted=triple.validation_status == ValidationStatus.ACCEPTED,
            rejected=triple.validation_status == ValidationStatus.REJECTED,
            needs_review=triple.validation_status == ValidationStatus.NEEDS_REVIEW,
            final_confidence=final_confidence,
            total_time_ms=total_time
        )
    
    def process_batch(self, triples: List[Triple]) -> List[Triple]:
        """Verarbeitet mehrere Triples."""
        return [self.process(t) for t in triples]
    
    def _log_result(self, triple: Triple, stage_info: str, start_time: float):
        """Loggt das Ergebnis der Verarbeitung."""
        duration = (time.time() - start_time) * 1000
        self.stats.total_time_ms += duration
        
        status_emoji = {
            ValidationStatus.ACCEPTED: "✅",
            ValidationStatus.REJECTED: "❌",
            ValidationStatus.NEEDS_REVIEW: "⚠️",
        }.get(triple.validation_status, "❓")
        
        logger.info(f"{status_emoji} {triple.validation_status.value.upper()} "
                   f"[{stage_info}] in {duration:.1f}ms")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Verarbeitungsstatistiken zurück."""
        return {
            "total_processed": self.stats.total_processed,
            "accepted": self.stats.accepted,
            "rejected": self.stats.rejected,
            "needs_review": self.stats.needs_review,
            "acceptance_rate": f"{self.stats.acceptance_rate:.1%}",
            "stage1_passed": self.stats.stage1_passed,
            "stage2_required": self.stats.stage2_required,
            "stage3_required": self.stats.stage3_required,
            "stage3_semantic_triggered": self.stats.stage3_semantic_triggered,
            "escalation_rate": f"{self.stats.escalation_rate:.1%}",
            "avg_time_ms": self.stats.total_time_ms / max(self.stats.total_processed, 1),
            "semantic_triggers": self.semantic_trigger.get_statistics(),
        }
    
    def reset_statistics(self):
        """Setzt die Statistiken zurück."""
        self.stats = ProcessingStats()
        if self.metrics:
            self.metrics.reset()
        self.semantic_trigger.reset_statistics()

    # =========================================================================
    # Export und Evaluation Methoden
    # =========================================================================

    def export_metrics(self, path: str, format: str = "json"):
        """
        Exportiert Metriken für wissenschaftliche Auswertung.

        Args:
            path: Dateipfad
            format: "json" oder "csv"
        """
        if not self.metrics:
            logger.warning("Metriken-Tracking nicht aktiviert")
            return

        self.metrics.finish()
        self.metrics.export(path, format)

    def get_evaluation_report(self) -> Dict[str, Any]:
        """
        Generiert einen umfassenden Evaluationsbericht.

        Enthält:
        - Zusammenfassung pro Stufe
        - Entity Resolution Statistiken
        - LLM-Kosten
        - Konfidenz-Verteilung
        """
        if not self.metrics:
            return {"error": "Metriken-Tracking nicht aktiviert"}

        report = {
            "summary": {
                "total_processed": self.metrics.total_triples_processed,
                "accepted": self.metrics.total_accepted,
                "rejected": self.metrics.total_rejected,
                "needs_review": self.metrics.total_needs_review,
                "acceptance_rate": f"{self.metrics.acceptance_rate:.1%}",
                "rejection_rate": f"{self.metrics.rejection_rate:.1%}",
                "review_rate": f"{self.metrics.review_rate:.1%}",
            },
            "stage_breakdown": {
                "stage1_rule_based": {
                    "processed": self.metrics.stage1_metrics.total_processed,
                    "pass_rate": f"{self.metrics.stage1_metrics.pass_rate:.1%}",
                    "avg_time_ms": f"{self.metrics.stage1_metrics.avg_time_ms:.2f}",
                    "avg_confidence": f"{self.metrics.stage1_metrics.avg_confidence:.2%}",
                },
                "stage2_embedding": {
                    "processed": self.metrics.stage2_metrics.total_processed,
                    "pass_rate": f"{self.metrics.stage2_metrics.pass_rate:.1%}",
                    "avg_time_ms": f"{self.metrics.stage2_metrics.avg_time_ms:.2f}",
                    "avg_confidence": f"{self.metrics.stage2_metrics.avg_confidence:.2%}",
                },
                "stage3_llm": {
                    "processed": self.metrics.stage3_metrics.total_processed,
                    "pass_rate": f"{self.metrics.stage3_metrics.pass_rate:.1%}",
                    "avg_time_ms": f"{self.metrics.stage3_metrics.avg_time_ms:.2f}",
                    "avg_confidence": f"{self.metrics.stage3_metrics.avg_confidence:.2%}",
                    "total_llm_calls": self.metrics.stage3_metrics.llm_calls,
                    "total_tokens": self.metrics.stage3_metrics.tokens_used,
                },
            },
            "entity_resolution": {
                "total_comparisons": self.metrics.entity_resolution.total_comparisons,
                "duplicates_found": self.metrics.entity_resolution.duplicates_found,
                "merges_performed": self.metrics.entity_resolution.merges_performed,
                "precision": f"{self.metrics.entity_resolution.precision:.2%}",
                "recall": f"{self.metrics.entity_resolution.recall:.2%}",
                "f1_score": f"{self.metrics.entity_resolution.f1_score:.2%}",
            },
            "llm_costs": {
                "total_tokens": self.metrics.total_llm_tokens,
                "estimated_cost_usd": f"${self.metrics.total_llm_cost_usd:.4f}",
            },
            "performance": {
                "total_time_ms": f"{self.metrics.total_processing_time_ms:.2f}",
                "avg_time_per_triple_ms": f"{self.metrics.total_processing_time_ms / max(self.metrics.total_triples_processed, 1):.2f}",
            },
            "confidence_histograms": self.metrics.get_confidence_histogram(),
        }

        return report

    def get_latex_table(self) -> str:
        """
        Generiert eine LaTeX-Tabelle für die Masterarbeit.

        Returns:
            LaTeX-formatierte Tabelle
        """
        if not self.metrics:
            return "% Metriken-Tracking nicht aktiviert"

        m = self.metrics

        latex = r"""
\begin{table}[htbp]
\centering
\caption{Evaluation des dreistufigen Konsistenzmoduls}
\label{tab:consistency-evaluation}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Stufe} & \textbf{Verarbeitet} & \textbf{Pass-Rate} & \textbf{Avg. Zeit (ms)} & \textbf{Avg. Konfidenz} & \textbf{Tokens} \\
\midrule
"""
        # Stufe 1
        latex += f"Regelbasiert & {m.stage1_metrics.total_processed} & {m.stage1_metrics.pass_rate:.1%} & {m.stage1_metrics.avg_time_ms:.1f} & {m.stage1_metrics.avg_confidence:.2%} & -- \\\\\n"

        # Stufe 2
        latex += f"Embedding & {m.stage2_metrics.total_processed} & {m.stage2_metrics.pass_rate:.1%} & {m.stage2_metrics.avg_time_ms:.1f} & {m.stage2_metrics.avg_confidence:.2%} & -- \\\\\n"

        # Stufe 3
        latex += f"LLM & {m.stage3_metrics.total_processed} & {m.stage3_metrics.pass_rate:.1%} & {m.stage3_metrics.avg_time_ms:.1f} & {m.stage3_metrics.avg_confidence:.2%} & {m.stage3_metrics.tokens_used:,} \\\\\n"

        latex += r"""
\midrule
\textbf{Gesamt} & """ + f"{m.total_triples_processed}" + r""" & """ + f"{m.acceptance_rate:.1%}" + r""" & """ + f"{m.total_processing_time_ms / max(m.total_triples_processed, 1):.1f}" + r""" & -- & """ + f"{m.total_llm_tokens:,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def get_provenance_statistics(self) -> Dict[str, Any]:
        """
        Gibt Provenance-Statistiken zurück.

        Enthält:
        - Source-Zuverlässigkeits-Scores
        - Corroboration-Statistiken
        - Top/Bottom-Quellen
        """
        return self.provenance_tracker.get_source_statistics()

    def get_provenance_report(self, source_id: str) -> Dict[str, Any]:
        """
        Gibt einen detaillierten Report für eine einzelne Quelle zurück.

        Args:
            source_id: ID der Quelle

        Returns:
            Detaillierter Report mit Statistiken
        """
        return self.provenance_tracker.get_source_report(source_id)

    def export_provenance_profiles(self, path: str):
        """
        Exportiert alle Source-Profile für spätere Verwendung.

        Args:
            path: Dateipfad für Export (JSON)
        """
        profiles = self.provenance_tracker.export_profiles()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Provenance-Profile exportiert nach {path}")

    def import_provenance_profiles(self, path: str):
        """
        Importiert Source-Profile aus früherer Session.

        Args:
            path: Dateipfad für Import (JSON)
        """
        with open(path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        self.provenance_tracker.import_profiles(profiles)
        logger.info(f"Provenance-Profile importiert von {path}")

    def print_summary(self):
        """Gibt eine formatierte Zusammenfassung auf der Konsole aus."""
        report = self.get_evaluation_report()

        print("\n" + "=" * 60)
        print("KONSISTENZMODUL - EVALUATION SUMMARY")
        print("=" * 60)

        print("\n📊 GESAMTSTATISTIK:")
        for key, value in report["summary"].items():
            print(f"   {key}: {value}")

        print("\n📈 STUFEN-AUFSCHLÜSSELUNG:")
        for stage, data in report["stage_breakdown"].items():
            print(f"\n   {stage}:")
            for key, value in data.items():
                print(f"      {key}: {value}")

        print("\n🔗 ENTITY RESOLUTION:")
        for key, value in report["entity_resolution"].items():
            print(f"   {key}: {value}")

        print("\n💰 LLM-KOSTEN:")
        for key, value in report["llm_costs"].items():
            print(f"   {key}: {value}")

        # Provenance-Statistiken
        prov_stats = self.get_provenance_statistics()
        if prov_stats.get("total_sources", 0) > 0:
            print("\n📜 PROVENANCE-TRACKING:")
            print(f"   Quellen verarbeitet: {prov_stats.get('total_sources', 0)}")
            print(f"   Triples verarbeitet: {prov_stats.get('total_triples_processed', 0)}")
            print(f"   Durchschnittliche Zuverlässigkeit: {prov_stats.get('avg_reliability', 0):.2%}")
            print(f"   Fakten mit Mehrfachbestätigung: {prov_stats.get('unique_facts_corroborated', 0)}")
            # #12: Missing Source Penalty Statistiken
            missing_count = prov_stats.get('missing_sources_count', 0)
            penalties_applied = prov_stats.get('missing_source_penalties_applied', 0)
            if missing_count > 0:
                print(f"   ⚠️  Triples ohne Quelle: {missing_count} (Penalties angewendet: {penalties_applied})")

            top_sources = prov_stats.get("top_sources", [])
            if top_sources:
                print("\n   Top-Quellen (nach Zuverlässigkeit):")
                for source_id, score, count in top_sources[:5]:
                    print(f"      {source_id}: {score:.2%} ({count} Triples)")

        # Semantischer Trigger Statistiken
        trigger_stats = self.semantic_trigger.get_statistics()
        if trigger_stats.get("total_checked", 0) > 0:
            print("\n🎯 SEMANTISCHER TRIGGER:")
            print(f"   Geprüft: {trigger_stats['total_checked']}")
            print(f"   Ausgelöst: {trigger_stats['triggered']}")
            print(f"   Trigger-Rate: {trigger_stats['trigger_rate']:.1%}")
            if trigger_stats["triggered"] > 0:
                reasons = trigger_stats.get("reasons", {})
                print(f"   Gründe:")
                for reason, count in reasons.items():
                    if count > 0:
                        print(f"      {reason}: {count}")

        print("\n" + "=" * 60)
