# src/consistency/semantic_trigger.py
"""
Selektive LLM-Trigger-Strategie für Logische Widersprüche.

Dieses Modul implementiert Heuristiken, die entscheiden, ob ein Triple
zusätzlich durch das LLM (Stage 3) geprüft werden soll.

Problemstellung:
- Stage 3 (LLM) wird nur bei combined_confidence < 0.7 aufgerufen
- Logische Widersprüche wie "Elsa Einstein MUTTER_VON Albert Einstein"
  (wobei Elsa bereits Ehefrau ist) werden NICHT erkannt
- Diese Widersprüche passieren alle regelbasierten Checks

Lösung:
- Trigger 1: "Gefährliche" Relation + bekannte Entität im Graph
- Trigger 2: Rollen-Konflikt-Muster erkannt (inkompatible Relationen)
- Trigger 3: Niedrige Konfidenz + Familienrelation

Erwartete LLM-Aufrufrate: 5-10% der Triples (vs. 100% bei "immer prüfen")
"""

import logging
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Dict, Any, Optional

from src.models.entities import Triple

logger = logging.getLogger(__name__)


@dataclass
class SemanticTriggerConfig:
    """Konfiguration für selektive LLM-Trigger."""

    # Feature-Flag: Semantischer Trigger aktivieren
    enable_semantic_trigger: bool = True

    # Gefährliche Relationen (Familienbeziehungen und Hierarchien)
    # Diese Relationen können zu logischen Widersprüchen führen
    dangerous_relations: Set[str] = field(default_factory=lambda: {
        # Familie
        "KIND_VON", "VATER_VON", "MUTTER_VON",
        "VERHEIRATET_MIT", "GESCHWISTER_VON",
        "GROSSELTERN_VON", "ENKEL_VON",
        # Hierarchie (optional)
        "LEITET", "ARBEITET_BEI", "STUDIERT_AN",
    })

    # Inkompatible Relationspaare (Familie + Hierarchie)
    # Wenn eine neue Relation REL1 hinzugefügt wird und bereits REL2 existiert,
    # und (REL1, REL2) inkompatibel sind, wird LLM aufgerufen.
    incompatible_pairs: List[Tuple[Set[str], Set[str]]] = field(default_factory=lambda: [
        # === FAMILIE ===
        # Verheiratet ≠ Elternteil/Kind
        ({"VERHEIRATET_MIT"}, {"MUTTER_VON", "VATER_VON", "KIND_VON"}),
        # Nicht gleichzeitig Vater und Mutter der gleichen Person
        ({"MUTTER_VON"}, {"VATER_VON"}),
        # Nicht Geschwister und Eltern-Kind
        ({"GESCHWISTER_VON"}, {"KIND_VON", "VATER_VON", "MUTTER_VON"}),
        # Enkel/Großeltern ≠ direkte Eltern-Kind
        ({"GROSSELTERN_VON", "ENKEL_VON"}, {"KIND_VON", "VATER_VON", "MUTTER_VON"}),

        # === HIERARCHIE ===
        # Nicht gleichzeitig Chef und Mitarbeiter der gleichen Organisation
        ({"LEITET"}, {"ARBEITET_BEI"}),
        # Nicht Student und Leiter der gleichen Institution
        ({"STUDIERT_AN"}, {"LEITET"}),
    ])

    # Konfidenz-Schwelle für zusätzliche Prüfung bei Familienrelationen
    low_confidence_threshold: float = 0.6


@dataclass
class TriggerResult:
    """Ergebnis der Trigger-Analyse."""
    should_trigger: bool
    reason: str
    trigger_type: str = "none"  # "dangerous_relation", "role_conflict", "low_confidence", "none"
    conflicting_relation: Optional[str] = None  # Die existierende Relation, die den Konflikt auslöst

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Metriken."""
        return {
            "should_trigger": self.should_trigger,
            "reason": self.reason,
            "trigger_type": self.trigger_type,
            "conflicting_relation": self.conflicting_relation,
        }


class SemanticTriggerAnalyzer:
    """
    Entscheidet ob LLM-Prüfung für ein Triple nötig ist.

    Die Analyse basiert auf drei Triggern:
    1. Gefährliche Relation + bekannte Entität
    2. Rollen-Konflikt-Muster (inkompatible existierende Relationen)
    3. Niedrige Konfidenz + Familienrelation
    """

    def __init__(self, config: SemanticTriggerConfig = None):
        """
        Initialisiert den Analyzer.

        Args:
            config: Konfiguration oder Standard-Konfiguration wenn None
        """
        self.config = config or SemanticTriggerConfig()

        # Statistiken
        self.stats = {
            "total_checked": 0,
            "triggered": 0,
            "reasons": {
                "role_conflict": 0,
                "low_confidence": 0,
            }
        }

        logger.debug(f"SemanticTriggerAnalyzer initialisiert")
        logger.debug(f"  → Gefährliche Relationen: {len(self.config.dangerous_relations)}")
        logger.debug(f"  → Inkompatible Paare: {len(self.config.incompatible_pairs)}")

    def should_trigger_llm(
        self,
        triple: Triple,
        graph_repo: Any,
        current_confidence: float
    ) -> TriggerResult:
        """
        Entscheidet ob LLM aufgerufen werden soll.

        Args:
            triple: Das zu prüfende Triple
            graph_repo: Repository für Graph-Zugriff
            current_confidence: Aktuelle kombinierte Konfidenz (Stage 1 * Stage 2)

        Returns:
            TriggerResult mit Entscheidung und Begründung
        """
        if not self.config.enable_semantic_trigger:
            return TriggerResult(
                should_trigger=False,
                reason="Semantic Trigger deaktiviert",
                trigger_type="disabled"
            )

        self.stats["total_checked"] += 1
        predicate = triple.predicate.upper().replace(" ", "_")

        # ========================================================================
        # Trigger 1 & 2: Gefährliche Relation + bekannte Entität + Rollen-Konflikt
        # ========================================================================
        if predicate in self.config.dangerous_relations:
            # Hole existierende Relationen für Subject und Object
            existing_subject_rels = self._get_entity_relations(
                triple.subject.id, graph_repo
            )
            existing_object_rels = self._get_entity_relations(
                triple.object.id, graph_repo
            )

            # Prüfe auf Rollen-Konflikt-Muster
            # Suche Relationen zwischen denselben beiden Entitäten
            subject_id = triple.subject.id
            object_id = triple.object.id

            for rel in existing_subject_rels:
                target_id = rel.get("target", {}).get("id") if isinstance(rel.get("target"), dict) else rel.get("target_id")
                source_id = rel.get("source", {}).get("id") if isinstance(rel.get("source"), dict) else rel.get("source_id")

                # Relation vom Subject zum Object?
                if target_id == object_id:
                    rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                    if self._is_incompatible(predicate, rel_type):
                        self.stats["triggered"] += 1
                        self.stats["reasons"]["role_conflict"] += 1
                        logger.info(f"  → Semantischer Trigger: Rollen-Konflikt "
                                   f"{predicate} vs {rel_type} für {triple.subject.name} → {triple.object.name}")
                        return TriggerResult(
                            should_trigger=True,
                            reason=f"Rollen-Konflikt: {predicate} vs existierendes {rel_type}",
                            trigger_type="role_conflict",
                            conflicting_relation=rel_type
                        )

            for rel in existing_object_rels:
                target_id = rel.get("target", {}).get("id") if isinstance(rel.get("target"), dict) else rel.get("target_id")
                source_id = rel.get("source", {}).get("id") if isinstance(rel.get("source"), dict) else rel.get("source_id")

                # Relation vom Object zum Subject? (inverse Richtung)
                if target_id == subject_id or source_id == subject_id:
                    rel_type = rel.get("rel_type", "").upper().replace(" ", "_")
                    if self._is_incompatible(predicate, rel_type):
                        self.stats["triggered"] += 1
                        self.stats["reasons"]["role_conflict"] += 1
                        logger.info(f"  → Semantischer Trigger: Rollen-Konflikt "
                                   f"{predicate} vs {rel_type} für {triple.subject.name} ↔ {triple.object.name}")
                        return TriggerResult(
                            should_trigger=True,
                            reason=f"Rollen-Konflikt: {predicate} vs existierendes {rel_type} (inverse)",
                            trigger_type="role_conflict",
                            conflicting_relation=rel_type
                        )

        # ========================================================================
        # Trigger 3: Niedrige Konfidenz + Familienrelation
        # ========================================================================
        if (current_confidence < self.config.low_confidence_threshold and
            predicate in self.config.dangerous_relations):
            self.stats["triggered"] += 1
            self.stats["reasons"]["low_confidence"] += 1
            logger.info(f"  → Semantischer Trigger: Niedrige Konfidenz "
                       f"({current_confidence:.2f}) + Familienrelation {predicate}")
            return TriggerResult(
                should_trigger=True,
                reason=f"Niedrige Konfidenz ({current_confidence:.2f}) + Familienrelation",
                trigger_type="low_confidence"
            )

        # Kein Trigger
        return TriggerResult(
            should_trigger=False,
            reason="Standard-Routing (kein Konflikt-Muster erkannt)",
            trigger_type="none"
        )

    def _get_entity_relations(self, entity_id: str, graph_repo: Any) -> List[Dict]:
        """
        Holt alle Relationen einer Entität (ein- und ausgehend).

        Args:
            entity_id: ID der Entität
            graph_repo: Repository für Graph-Zugriff

        Returns:
            Liste von Relation-Dictionaries
        """
        if graph_repo is None:
            return []

        relations = []
        try:
            # Ausgehende Relationen (Entity als Source)
            outgoing = graph_repo.find_relations(source_id=entity_id)
            if outgoing:
                relations.extend(outgoing)
        except Exception as e:
            logger.debug(f"Fehler beim Laden ausgehender Relationen: {e}")

        try:
            # Eingehende Relationen (Entity als Target)
            incoming = graph_repo.find_relations(target_id=entity_id)
            if incoming:
                relations.extend(incoming)
        except Exception as e:
            logger.debug(f"Fehler beim Laden eingehender Relationen: {e}")

        return relations

    def _is_incompatible(self, rel1: str, rel2: str) -> bool:
        """
        Prüft ob zwei Relationen inkompatibel sind.

        Args:
            rel1: Erste Relation (neue)
            rel2: Zweite Relation (existierende)

        Returns:
            True wenn inkompatibel
        """
        rel1 = rel1.upper().replace(" ", "_")
        rel2 = rel2.upper().replace(" ", "_")

        for set1, set2 in self.config.incompatible_pairs:
            # Prüfe beide Richtungen
            if (rel1 in set1 and rel2 in set2) or (rel1 in set2 and rel2 in set1):
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über Trigger-Aktivierungen zurück.

        Returns:
            Dictionary mit Statistiken
        """
        total = self.stats["total_checked"]
        triggered = self.stats["triggered"]

        return {
            "total_checked": total,
            "triggered": triggered,
            "trigger_rate": triggered / total if total > 0 else 0.0,
            "reasons": self.stats["reasons"].copy(),
        }

    def reset_statistics(self):
        """Setzt die Statistiken zurück."""
        self.stats = {
            "total_checked": 0,
            "triggered": 0,
            "reasons": {
                "role_conflict": 0,
                "low_confidence": 0,
            }
        }
