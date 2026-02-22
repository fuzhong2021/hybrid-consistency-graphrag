# src/consistency/provenance.py
"""
Provenance-basierte Konfidenz-Gewichtung für Knowledge Graphs.

Prinzip:
- Triples aus zuverlässigen Quellen haben höhere Konfidenz
- Mehrfach bestätigte Fakten sind zuverlässiger
- Konsistente Quellen werden höher gewichtet

Features:
1. Source Quality Scoring - Bewertung der Quellzuverlässigkeit
2. Corroboration Tracking - Mehrfachbestätigung von Fakten
3. Temporal Decay - Ältere Informationen können weniger zuverlässig sein
4. Conflict-aware Weighting - Quellen die oft Konflikte produzieren werden abgewertet

Wissenschaftliche Grundlage:
- Dong et al. (2014): "Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion"
- Pasternack & Roth (2010): "Knowing What to Believe"
"""

import logging
import math
from typing import Optional, Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from src.models.entities import Triple

logger = logging.getLogger(__name__)


@dataclass
class SourceProfile:
    """Profil einer Informationsquelle."""

    source_id: str

    # Qualitätsmetriken
    reliability_score: float = 0.5  # 0-1, initialer Default

    # Statistiken
    total_triples: int = 0
    accepted_triples: int = 0
    rejected_triples: int = 0
    conflicts_caused: int = 0

    # Zeitliche Informationen
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    # Kategorien/Tags
    categories: Set[str] = field(default_factory=set)

    @property
    def acceptance_rate(self) -> float:
        """Anteil akzeptierter Triples."""
        if self.total_triples == 0:
            return 0.5
        return self.accepted_triples / self.total_triples

    @property
    def conflict_rate(self) -> float:
        """Anteil an Triples die Konflikte verursachten."""
        if self.total_triples == 0:
            return 0.0
        return self.conflicts_caused / self.total_triples


@dataclass
class ProvenanceConfig:
    """Konfiguration für Provenance-Tracking."""

    # Bekannte vertrauenswürdige Quellen
    trusted_sources: Dict[str, float] = field(default_factory=lambda: {
        "wikipedia": 0.9,
        "wiki": 0.9,
        "wikidata": 0.95,
        "dbpedia": 0.85,
        "freebase": 0.85,
        "academic": 0.8,
        "pubmed": 0.85,
        "arxiv": 0.8,
        "gov": 0.85,
        "edu": 0.8,
    })

    # Bekannte weniger zuverlässige Quellen
    untrusted_patterns: List[str] = field(default_factory=lambda: [
        "blog",
        "forum",
        "reddit",
        "twitter",
        "social",
        "user_generated",
    ])

    # Default Scores
    default_source_score: float = 0.5
    unknown_source_score: float = 0.4

    # Corroboration (Mehrfachbestätigung)
    enable_corroboration: bool = True
    corroboration_bonus: float = 0.1  # Pro zusätzlicher Quelle
    max_corroboration_bonus: float = 0.3

    # Temporal Decay
    enable_temporal_decay: bool = False
    decay_half_life_days: int = 365  # Nach 1 Jahr halbiert sich der Bonus

    # Learning
    enable_source_learning: bool = True
    learning_rate: float = 0.1  # Wie stark beeinflusst neues Feedback den Score

    # Minimum Konfidenz (wird nie unterschritten)
    min_confidence: float = 0.1
    max_confidence: float = 0.99

    # Missing Source Penalty (#12)
    enable_missing_source_penalty: bool = True
    missing_source_penalty: float = 0.7  # 30% Konfidenz-Abzug (Multiplikator)

    # Source Verification (#13) - Prüft ob Quelle den Claim tatsächlich belegt
    enable_source_verification: bool = True
    source_verification_high_threshold: float = 0.5    # Quelle unterstützt stark
    source_verification_medium_threshold: float = 0.3  # Quelle unterstützt teilweise
    source_verification_low_threshold: float = 0.15    # Quelle unterstützt kaum
    source_verification_no_support_penalty: float = 0.5   # 50% Abzug wenn keine Unterstützung
    source_verification_low_support_penalty: float = 0.7  # 30% Abzug bei schwacher Unterstützung


class ProvenanceTracker:
    """
    Verfolgt und bewertet die Provenance von Knowledge Graph Triples.
    """

    def __init__(self, config: ProvenanceConfig = None):
        self.config = config or ProvenanceConfig()

        # Source Profiles
        self.sources: Dict[str, SourceProfile] = {}

        # Fakt-Corroboration: (subject_id, predicate, object_id) -> Set[source_ids]
        self.fact_sources: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

        # Statistiken
        self.total_triples_processed = 0
        self._missing_sources_count = 0  # #12: Anzahl Triples ohne Quelle
        self._penalties_applied = 0  # #12: Anzahl angewendeter Penalties
        self._source_verification_count = 0  # #13: Anzahl Source Verifications
        self._source_unsupported_count = 0   # #13: Anzahl nicht unterstützter Claims

        # Source Verifier (#13)
        self._source_verifier = None
        if self.config.enable_source_verification:
            self._init_source_verifier()

        logger.info(
            f"ProvenanceTracker initialisiert: "
            f"{len(self.config.trusted_sources)} trusted sources, "
            f"source_verification={self.config.enable_source_verification}"
        )

    def _init_source_verifier(self):
        """Initialisiert den Source Verifier."""
        try:
            from src.consistency.source_verification import (
                SourceVerifier, SourceVerificationConfig
            )
            verifier_config = SourceVerificationConfig(
                enable_source_verification=True,
                high_support_threshold=self.config.source_verification_high_threshold,
                medium_support_threshold=self.config.source_verification_medium_threshold,
                low_support_threshold=self.config.source_verification_low_threshold,
                no_support_penalty=self.config.source_verification_no_support_penalty,
                low_support_penalty=self.config.source_verification_low_support_penalty,
            )
            self._source_verifier = SourceVerifier(verifier_config)
            logger.info("Source Verifier initialisiert")
        except ImportError as e:
            logger.warning(f"Source Verifier nicht verfügbar: {e}")

    def get_source_profile(self, source_id: str) -> SourceProfile:
        """
        Holt oder erstellt ein Source-Profil.
        """
        if source_id not in self.sources:
            profile = SourceProfile(source_id=source_id)
            profile.reliability_score = self._initial_source_score(source_id)
            profile.first_seen = datetime.now()
            self.sources[source_id] = profile

        profile = self.sources[source_id]
        profile.last_seen = datetime.now()
        return profile

    def _initial_source_score(self, source_id: str) -> float:
        """
        Berechnet initialen Reliability-Score für eine neue Quelle.
        """
        source_lower = source_id.lower()

        # Prüfe trusted sources
        for pattern, score in self.config.trusted_sources.items():
            if pattern in source_lower:
                return score

        # Prüfe untrusted patterns
        for pattern in self.config.untrusted_patterns:
            if pattern in source_lower:
                return max(0.3, self.config.default_source_score - 0.2)

        return self.config.default_source_score

    def calculate_confidence(
        self,
        triple: Triple,
        base_confidence: float = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Berechnet Provenance-adjustierte Konfidenz für ein Triple.

        Args:
            triple: Das Triple
            base_confidence: Basis-Konfidenz (z.B. aus Extraktion)

        Returns:
            Tuple von (adjusted_confidence, details)
        """
        base = base_confidence if base_confidence is not None else triple.extraction_confidence
        details = {"base_confidence": base}

        # 1. Source Quality
        source_id = triple.source_document_id or "unknown"
        source_profile = self.get_source_profile(source_id)
        source_factor = source_profile.reliability_score
        details["source_score"] = source_factor
        details["source_id"] = source_id

        # 1b. Missing Source Penalty (#12)
        if not triple.source_document_id and self.config.enable_missing_source_penalty:
            penalty = self.config.missing_source_penalty
            source_factor *= penalty
            details["missing_source_penalty"] = penalty
            details["missing_source_warning"] = True
            self._missing_sources_count += 1
            self._penalties_applied += 1
            logger.debug(f"  Missing Source Penalty: ×{penalty}")

        # 1c. Source Verification (#13) - Prüft ob Quelle den Claim tatsächlich belegt
        if (self.config.enable_source_verification and
            self._source_verifier is not None and
            triple.source_text):
            verification_result = self._source_verifier.verify(triple)
            self._source_verification_count += 1

            details["source_verification"] = {
                "similarity": verification_result.similarity_score,
                "support_level": verification_result.support_level,
                "confidence_factor": verification_result.confidence_factor,
                "claim": verification_result.claim_text,
            }

            if not verification_result.is_verified:
                # Quelle unterstützt den Claim nicht
                source_factor *= verification_result.confidence_factor
                details["source_verification_warning"] = True
                self._source_unsupported_count += 1
                logger.debug(
                    f"  Source Verification: {verification_result.support_level} "
                    f"(sim={verification_result.similarity_score:.3f}) "
                    f"→ ×{verification_result.confidence_factor}"
                )
            else:
                # Quelle unterstützt den Claim - optionaler Bonus
                if verification_result.confidence_factor > 1.0:
                    source_factor *= verification_result.confidence_factor
                    details["source_verification_bonus"] = verification_result.confidence_factor
                    logger.debug(
                        f"  Source Verification: {verification_result.support_level} "
                        f"(sim={verification_result.similarity_score:.3f}) "
                        f"→ ×{verification_result.confidence_factor} BONUS"
                    )

        # 2. Corroboration Bonus
        corroboration_bonus = 0.0
        if self.config.enable_corroboration:
            fact_key = (triple.subject.id, triple.predicate.upper(), triple.object.id)
            corroborating_sources = self.fact_sources.get(fact_key, set())
            num_sources = len(corroborating_sources)

            if num_sources > 0:
                corroboration_bonus = min(
                    num_sources * self.config.corroboration_bonus,
                    self.config.max_corroboration_bonus
                )
                details["corroborating_sources"] = num_sources
                details["corroboration_bonus"] = corroboration_bonus

        # 3. Temporal Decay (optional)
        temporal_factor = 1.0
        if self.config.enable_temporal_decay and hasattr(triple, 'extraction_timestamp'):
            age_days = (datetime.now() - triple.extraction_timestamp).days
            half_life = self.config.decay_half_life_days
            temporal_factor = 0.5 ** (age_days / half_life)
            details["temporal_factor"] = temporal_factor

        # Kombinierte Konfidenz berechnen
        # Formel: adjusted = base * source_factor * temporal_factor + corroboration_bonus
        adjusted = base * source_factor * temporal_factor + corroboration_bonus

        # Clamp to valid range
        adjusted = max(self.config.min_confidence, min(self.config.max_confidence, adjusted))

        details["adjusted_confidence"] = adjusted
        details["adjustment"] = adjusted - base

        return adjusted, details

    def record_triple(
        self,
        triple: Triple,
        accepted: bool,
        caused_conflict: bool = False
    ):
        """
        Zeichnet ein verarbeitetes Triple auf.

        Aktualisiert Source-Statistiken und Fakt-Corroboration.
        """
        source_id = triple.source_document_id or "unknown"
        profile = self.get_source_profile(source_id)

        # Statistiken aktualisieren
        profile.total_triples += 1
        if accepted:
            profile.accepted_triples += 1
        else:
            profile.rejected_triples += 1
        if caused_conflict:
            profile.conflicts_caused += 1

        # Fakt-Corroboration aktualisieren
        if accepted and self.config.enable_corroboration:
            fact_key = (triple.subject.id, triple.predicate.upper(), triple.object.id)
            self.fact_sources[fact_key].add(source_id)

        # Source-Reliability lernen
        if self.config.enable_source_learning and profile.total_triples >= 5:
            self._update_source_reliability(profile)

        self.total_triples_processed += 1

    def _update_source_reliability(self, profile: SourceProfile):
        """
        Aktualisiert den Reliability-Score einer Quelle basierend auf Performance.
        """
        # Kombiniere verschiedene Faktoren
        acceptance_rate = profile.acceptance_rate
        conflict_rate = profile.conflict_rate

        # Ziel-Score basierend auf Performance
        target_score = acceptance_rate * (1 - conflict_rate * 0.5)

        # Exponential Moving Average
        alpha = self.config.learning_rate
        profile.reliability_score = (
            (1 - alpha) * profile.reliability_score +
            alpha * target_score
        )

        # Clamp
        profile.reliability_score = max(0.1, min(0.99, profile.reliability_score))

    def get_corroboration_count(self, triple: Triple) -> int:
        """
        Gibt die Anzahl der Quellen zurück die diesen Fakt bestätigen.
        """
        fact_key = (triple.subject.id, triple.predicate.upper(), triple.object.id)
        return len(self.fact_sources.get(fact_key, set()))

    def get_source_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über alle Quellen zurück.
        """
        if not self.sources:
            return {"total_sources": 0}

        reliability_scores = [p.reliability_score for p in self.sources.values()]

        return {
            "total_sources": len(self.sources),
            "total_triples_processed": self.total_triples_processed,
            "avg_reliability": sum(reliability_scores) / len(reliability_scores),
            "min_reliability": min(reliability_scores),
            "max_reliability": max(reliability_scores),
            "top_sources": sorted(
                [(s.source_id, s.reliability_score, s.total_triples)
                 for s in self.sources.values()],
                key=lambda x: -x[1]
            )[:10],
            "unique_facts_corroborated": len(self.fact_sources),
            # #12: Missing Source Penalty Statistics
            "missing_sources_count": self._missing_sources_count,
            "missing_source_penalties_applied": self._penalties_applied,
            # #13: Source Verification Statistics
            "source_verifications_performed": self._source_verification_count,
            "source_claims_unsupported": self._source_unsupported_count,
            "source_verification_rejection_rate": (
                self._source_unsupported_count / self._source_verification_count
                if self._source_verification_count > 0 else 0.0
            ),
        }

    def get_source_report(self, source_id: str) -> Dict[str, Any]:
        """
        Gibt einen detaillierten Report für eine Quelle zurück.
        """
        if source_id not in self.sources:
            return {"error": "Source not found"}

        profile = self.sources[source_id]

        return {
            "source_id": source_id,
            "reliability_score": profile.reliability_score,
            "total_triples": profile.total_triples,
            "accepted_triples": profile.accepted_triples,
            "rejected_triples": profile.rejected_triples,
            "acceptance_rate": profile.acceptance_rate,
            "conflicts_caused": profile.conflicts_caused,
            "conflict_rate": profile.conflict_rate,
            "first_seen": profile.first_seen.isoformat() if profile.first_seen else None,
            "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
            "categories": list(profile.categories),
        }

    def export_profiles(self) -> Dict[str, Dict]:
        """
        Exportiert alle Source-Profile.
        """
        return {
            source_id: self.get_source_report(source_id)
            for source_id in self.sources
        }

    def import_profiles(self, profiles: Dict[str, Dict]):
        """
        Importiert Source-Profile.
        """
        for source_id, data in profiles.items():
            profile = SourceProfile(
                source_id=source_id,
                reliability_score=data.get("reliability_score", 0.5),
                total_triples=data.get("total_triples", 0),
                accepted_triples=data.get("accepted_triples", 0),
                rejected_triples=data.get("rejected_triples", 0),
                conflicts_caused=data.get("conflicts_caused", 0),
            )
            if data.get("first_seen"):
                profile.first_seen = datetime.fromisoformat(data["first_seen"])
            if data.get("last_seen"):
                profile.last_seen = datetime.fromisoformat(data["last_seen"])
            if data.get("categories"):
                profile.categories = set(data["categories"])

            self.sources[source_id] = profile

        logger.info(f"Imported {len(profiles)} source profiles")


# =============================================================================
# ERKLÄRBARKEIT
# =============================================================================

class ProvenanceExplainer:
    """
    Erklärt Provenance-basierte Konfidenz-Anpassungen.
    """

    def __init__(self, tracker: ProvenanceTracker):
        self.tracker = tracker

    def explain_confidence(
        self,
        triple: Triple,
        details: Dict[str, Any]
    ) -> str:
        """
        Erstellt eine menschenlesbare Erklärung für die Konfidenz-Anpassung.
        """
        parts = []

        # Basis
        base = details.get("base_confidence", 0.5)
        parts.append(f"Basis-Konfidenz: {base:.0%}")

        # Source
        source_id = details.get("source_id", "unknown")
        source_score = details.get("source_score", 0.5)
        parts.append(f"Quelle '{source_id}': {source_score:.0%} Zuverlässigkeit")

        # Missing Source Penalty (#12)
        if details.get("missing_source_warning"):
            penalty = details.get("missing_source_penalty", 0.7)
            parts.append(f"⚠️  Keine Quellenangabe: ×{penalty:.0%} Penalty")

        # Source Verification (#13)
        if "source_verification" in details:
            sv = details["source_verification"]
            support = sv.get("support_level", "unknown")
            sim = sv.get("similarity", 0)
            factor = sv.get("confidence_factor", 1.0)

            if details.get("source_verification_warning"):
                parts.append(
                    f"⚠️  Quelle unterstützt Claim nicht: "
                    f"support={support}, sim={sim:.2f} → ×{factor:.0%}"
                )
            elif details.get("source_verification_bonus"):
                parts.append(
                    f"✓  Quelle unterstützt Claim: "
                    f"support={support}, sim={sim:.2f} → ×{factor:.0%} Bonus"
                )

        # Corroboration
        if "corroborating_sources" in details:
            num = details["corroborating_sources"]
            bonus = details.get("corroboration_bonus", 0)
            parts.append(f"Bestätigt durch {num} weitere Quelle(n): +{bonus:.0%}")

        # Temporal
        if "temporal_factor" in details:
            factor = details["temporal_factor"]
            parts.append(f"Zeitlicher Faktor: {factor:.0%}")

        # Ergebnis
        adjusted = details.get("adjusted_confidence", base)
        adjustment = details.get("adjustment", 0)
        sign = "+" if adjustment >= 0 else ""
        parts.append(f"→ Angepasste Konfidenz: {adjusted:.0%} ({sign}{adjustment:.0%})")

        return "\n".join(parts)

    def explain_source(self, source_id: str) -> str:
        """
        Erstellt eine menschenlesbare Erklärung für eine Quelle.
        """
        if source_id not in self.tracker.sources:
            return f"Quelle '{source_id}' ist unbekannt."

        report = self.tracker.get_source_report(source_id)

        parts = [
            f"Quelle: {source_id}",
            f"Zuverlässigkeit: {report['reliability_score']:.0%}",
            f"Verarbeitet: {report['total_triples']} Triples",
            f"Akzeptiert: {report['accepted_triples']} ({report['acceptance_rate']:.0%})",
            f"Konflikte: {report['conflicts_caused']} ({report['conflict_rate']:.0%})",
        ]

        return "\n".join(parts)
