# src/consistency/llm_arbitrator.py
"""
Stufe 3: LLM-basierte Konfliktauflösung

Wird nur bei komplexen Konflikten aktiviert, die regelbasiert
oder embedding-basiert nicht auflösbar sind.

Wissenschaftliche Grundlage:
- Nentidis et al. (2025): Conflict Resolution in KGs
- LLM-as-Judge Pattern

Erweitert um:
- Token-Tracking für Kostenanalyse
- Faktenverifikation gegen Graph-Kontext
- Strukturierte Reasoning-Ausgabe (Chain-of-Thought)
"""

import time
import json
import logging
from typing import Any, Optional, Dict, List, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field

from src.models.entities import Triple, ConflictSet, ConflictType
from src.consistency.base import ValidationStage, StageResult, ValidationOutcome, ConsistencyConfig
from src.consistency.metrics import LLMUsageStats

logger = logging.getLogger(__name__)


class ResolutionAction(Enum):
    """Mögliche Auflösungsaktionen."""
    ACCEPT_NEW = "accept_new"        # Neues Triple akzeptieren
    KEEP_EXISTING = "keep_existing"  # Existierendes behalten
    MERGE = "merge"                  # Zusammenführen
    REJECT_BOTH = "reject_both"      # Beide ablehnen
    HUMAN_REVIEW = "human_review"    # Manuelle Prüfung


@dataclass
class LLMResolution:
    """Strukturierte Ausgabe der LLM-Konfliktauflösung mit Reasoning."""
    is_valid: bool
    confidence: float
    resolution: ResolutionAction
    reasoning: str
    suggested_merge: Optional[Dict[str, Any]] = None

    # Chain-of-Thought Details
    fact_check_result: Optional[str] = None
    evidence_found: List[str] = field(default_factory=list)
    contradictions_found: List[str] = field(default_factory=list)

    # Token-Usage für diesen Aufruf
    usage_stats: Optional[LLMUsageStats] = None

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "resolution": self.resolution.value if isinstance(self.resolution, ResolutionAction) else self.resolution,
            "reasoning": self.reasoning,
            "suggested_merge": self.suggested_merge,
            "fact_check_result": self.fact_check_result,
            "evidence_found": self.evidence_found,
            "contradictions_found": self.contradictions_found,
            "usage_stats": self.usage_stats.to_dict() if self.usage_stats else None,
        }


class LLMArbitrator(ValidationStage):
    """Stufe 3: LLM-basierte Konfliktauflösung."""
    
    name = "llm_arbitration"
    
    ARBITRATION_PROMPT = """Du bist ein Experte für Knowledge Graph Qualitätssicherung.

## Aufgabe
Analysiere den folgenden Konflikt und entscheide, wie er aufgelöst werden soll.

## Neues Triple (zu validieren)
- Subject: {subject_name} (Typ: {subject_type})
- Prädikat: {predicate}
- Object: {object_name} (Typ: {object_type})
- Quelle: {source_text}
- Extraktions-Konfidenz: {extraction_confidence:.0%}

## Konflikt
- Typ: {conflict_type}
- Beschreibung: {conflict_description}
- Schweregrad: {severity:.0%}

## Existierende Informationen im Graph
{existing_info}

## Deine Aufgabe
Analysiere den Konflikt und entscheide:

1. Ist das neue Triple faktisch korrekt?
2. Widerspricht es den existierenden Informationen?
3. Wie sollte der Konflikt aufgelöst werden?

## Antwort (JSON-Format)
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "resolution": "accept_new" | "keep_existing" | "merge" | "reject_both" | "human_review",
    "reasoning": "Deine Begründung...",
    "suggested_merge": null oder {{"name": "...", "properties": {{}}}}
}}
"""
    
    # Zusätzlicher Prompt für Faktenverifikation
    FACT_VERIFICATION_PROMPT = """Prüfe die faktische Korrektheit des folgenden Tripels anhand des gegebenen Graph-Kontexts.

## Triple zu prüfen
- Subject: {subject_name} (Typ: {subject_type})
- Prädikat: {predicate}
- Object: {object_name} (Typ: {object_type})
- Quelltext: {source_text}

## Graph-Kontext
{graph_context}

## Aufgabe
1. Prüfe ob das Triple faktisch korrekt ist
2. Suche nach unterstützenden oder widersprechenden Evidenzen im Kontext
3. Bewerte die Vertrauenswürdigkeit

## Antwort (JSON-Format)
{{
    "is_factually_correct": true/false/null,
    "confidence": 0.0-1.0,
    "supporting_evidence": ["Evidenz 1", "Evidenz 2", ...],
    "contradicting_evidence": ["Widerspruch 1", ...],
    "reasoning": "Deine Begründung mit Chain-of-Thought..."
}}
"""

    def __init__(
        self,
        config: ConsistencyConfig,
        llm_client: Any = None,
        metrics_callback: Optional[Callable[[LLMUsageStats], None]] = None
    ):
        """
        Args:
            config: Konsistenz-Konfiguration
            llm_client: OpenAI-Client oder kompatibles LLM
            metrics_callback: Callback-Funktion für Token-Tracking
        """
        self.config = config
        self.llm_client = llm_client
        self.max_retries = config.max_llm_retries
        self.metrics_callback = metrics_callback

        # Token-Tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.usage_history: List[LLMUsageStats] = []

        logger.info(f"LLMArbitrator initialisiert (Model: {config.llm_model})")
    
    def validate(self, triple: Triple, graph_repo: Any = None) -> StageResult:
        """
        Löst Konflikte mittels LLM-Analyse auf.

        Wenn keine Konflikte vorliegen, wird trotzdem eine Faktenverifikation
        durchgeführt um semantische Widersprüche zu erkennen.
        """
        start_time = time.time()

        if not self.llm_client:
            logger.warning("Kein LLM-Client verfügbar - markiere für Human Review")
            return StageResult(
                outcome=ValidationOutcome.UNCERTAIN,
                confidence=0.5,
                conflicts=triple.conflicts,
                processing_time_ms=(time.time() - start_time) * 1000,
                details={"skipped": True, "reason": "no_llm_client"}
            )

        # Wenn keine Konflikte vorliegen: Semantische Faktenverifikation durchführen
        # Dies erkennt temporale, logische und Zustandswidersprüche
        if not triple.conflicts:
            fact_check = self._verify_semantic_consistency(triple, graph_repo)
            processing_time = (time.time() - start_time) * 1000

            if fact_check:
                logger.info(f"Stufe 3 [{triple.subject.name}]: Faktencheck → "
                           f"{'PASS' if fact_check.is_valid else 'FAIL'} "
                           f"(Confidence: {fact_check.confidence:.0%})")

                if fact_check.is_valid:
                    return StageResult(
                        outcome=ValidationOutcome.PASS,
                        confidence=fact_check.confidence,
                        processing_time_ms=processing_time,
                        details={
                            "fact_verification": True,
                            "reasoning": fact_check.reasoning,
                            "llm_calls": 1
                        }
                    )
                else:
                    # Semantischer Widerspruch erkannt!
                    conflict = ConflictSet(
                        conflict_type=ConflictType.CONTRADICTORY_RELATION,
                        conflicting_items=[triple],
                        description=f"Semantischer Widerspruch: {fact_check.reasoning}",
                        severity=0.9
                    )
                    return StageResult(
                        outcome=ValidationOutcome.FAIL,
                        confidence=fact_check.confidence,
                        conflicts=[conflict],
                        processing_time_ms=processing_time,
                        details={
                            "fact_verification": True,
                            "reasoning": fact_check.reasoning,
                            "contradictions": fact_check.contradictions_found,
                            "llm_calls": 1
                        }
                    )
            else:
                # Faktencheck fehlgeschlagen - UNCERTAIN
                return StageResult(
                    outcome=ValidationOutcome.UNCERTAIN,
                    confidence=0.5,
                    processing_time_ms=processing_time,
                    details={"fact_verification_failed": True}
                )

        # Jeden Konflikt einzeln arbitrieren
        resolved_conflicts = []
        resolutions = []
        overall_confidence = 1.0
        
        for conflict in triple.conflicts:
            resolution = self._arbitrate_single_conflict(triple, conflict, graph_repo)

            if resolution:
                resolved_conflicts.append(conflict)
                resolutions.append(resolution)

                # Unterstütze sowohl Dicts als auch LLMResolution Objekte
                if isinstance(resolution, dict):
                    overall_confidence *= resolution.get("confidence", 0.5)
                    conflict.resolution = resolution.get("resolution")
                else:
                    # LLMResolution Objekt
                    overall_confidence *= getattr(resolution, "confidence", 0.5)
                    conflict.resolution = getattr(resolution, "resolution", None)
        
        # Outcome bestimmen basierend auf Resolutions
        outcome = self._determine_outcome(resolutions, overall_confidence)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Stufe 3 [{triple.subject.name}]: {outcome.value}, "
                   f"{len(resolved_conflicts)}/{len(triple.conflicts)} Konflikte aufgelöst")
        
        return StageResult(
            outcome=outcome,
            confidence=overall_confidence,
            conflicts=resolved_conflicts,
            processing_time_ms=processing_time,
            details={
                "resolutions": [r.to_dict() if isinstance(r, LLMResolution) else r for r in resolutions],
                "conflicts_resolved": len(resolved_conflicts),
                "conflicts_total": len(triple.conflicts),
                "llm_calls": len([r for r in resolutions if r]),
                "total_tokens_used": sum(
                    r.usage_stats.total_tokens if isinstance(r, LLMResolution) and r.usage_stats else 0
                    for r in resolutions
                )
            }
        )
    
    def _arbitrate_single_conflict(
        self,
        triple: Triple,
        conflict: ConflictSet,
        graph_repo: Any
    ) -> Optional[LLMResolution]:
        """Arbitriert einen einzelnen Konflikt mit Token-Tracking."""

        # Existierende Informationen sammeln
        existing_info = self._gather_context(triple, graph_repo)

        # Prompt erstellen
        prompt = self.ARBITRATION_PROMPT.format(
            subject_name=triple.subject.name,
            subject_type=triple.subject.entity_type.value,
            predicate=triple.predicate,
            object_name=triple.object.name,
            object_type=triple.object.entity_type.value,
            source_text=triple.source_text or "Nicht verfügbar",
            extraction_confidence=triple.extraction_confidence,
            conflict_type=conflict.conflict_type.value,
            conflict_description=conflict.description,
            severity=conflict.severity,
            existing_info=existing_info
        )

        # LLM aufrufen mit Token-Tracking
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": "Du bist ein Knowledge Graph Experte. Antworte nur in validem JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                latency_ms = (time.time() - start_time) * 1000

                # Token-Tracking
                usage_stats = self._extract_usage_stats(response, latency_ms)
                self._record_usage(usage_stats)

                result = json.loads(response.choices[0].message.content)

                # Parse zu LLMResolution
                resolution = self._parse_resolution(result, usage_stats)

                logger.debug(f"LLM Resolution: {resolution.resolution} "
                           f"(Confidence: {resolution.confidence:.0%}, "
                           f"Tokens: {usage_stats.total_tokens})")

                return resolution

            except json.JSONDecodeError as e:
                logger.warning(f"JSON Parse Fehler (Versuch {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"LLM Fehler (Versuch {attempt + 1}): {e}")

        return None

    def _extract_usage_stats(self, response: Any, latency_ms: float) -> LLMUsageStats:
        """Extrahiert Token-Usage aus der LLM-Response."""
        input_tokens = 0
        output_tokens = 0

        # OpenAI-Format
        if hasattr(response, 'usage'):
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)

        return LLMUsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.llm_model,
            latency_ms=latency_ms
        )

    def _record_usage(self, usage_stats: LLMUsageStats):
        """Zeichnet Token-Usage auf."""
        self.total_input_tokens += usage_stats.input_tokens
        self.total_output_tokens += usage_stats.output_tokens
        self.total_calls += 1
        self.usage_history.append(usage_stats)

        # Callback aufrufen wenn vorhanden
        if self.metrics_callback:
            self.metrics_callback(usage_stats)

    def _parse_resolution(
        self,
        result: Dict[str, Any],
        usage_stats: LLMUsageStats
    ) -> LLMResolution:
        """Parsed LLM-Response zu strukturiertem LLMResolution-Objekt."""
        resolution_str = result.get("resolution", "human_review")

        # Mappe String zu Enum
        resolution_map = {
            "accept_new": ResolutionAction.ACCEPT_NEW,
            "keep_existing": ResolutionAction.KEEP_EXISTING,
            "merge": ResolutionAction.MERGE,
            "reject_both": ResolutionAction.REJECT_BOTH,
            "human_review": ResolutionAction.HUMAN_REVIEW,
        }

        resolution = resolution_map.get(resolution_str.lower(), ResolutionAction.HUMAN_REVIEW)

        return LLMResolution(
            is_valid=result.get("is_valid", False),
            confidence=result.get("confidence", 0.5),
            resolution=resolution,
            reasoning=result.get("reasoning", ""),
            suggested_merge=result.get("suggested_merge"),
            usage_stats=usage_stats
        )

    def verify_fact(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[LLMResolution]:
        """
        Führt Faktenverifikation für ein Triple durch.

        Separater Aufruf für zusätzliche Validierung kritischer Fakten.

        Args:
            triple: Das zu verifizierende Triple
            graph_repo: Repository für Graph-Kontext

        Returns:
            LLMResolution mit Faktencheck-Ergebnis
        """
        if not self.llm_client:
            logger.warning("Kein LLM-Client verfügbar für Faktenverifikation")
            return None

        # Erweiterten Kontext sammeln
        graph_context = self._gather_extended_context(triple, graph_repo)

        prompt = self.FACT_VERIFICATION_PROMPT.format(
            subject_name=triple.subject.name,
            subject_type=triple.subject.entity_type.value,
            predicate=triple.predicate,
            object_name=triple.object.name,
            object_type=triple.object.entity_type.value,
            source_text=triple.source_text or "Nicht verfügbar",
            graph_context=graph_context
        )

        try:
            start_time = time.time()

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "Du bist ein Faktenprüfer für Knowledge Graphs. Antworte nur in validem JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            latency_ms = (time.time() - start_time) * 1000
            usage_stats = self._extract_usage_stats(response, latency_ms)
            self._record_usage(usage_stats)

            result = json.loads(response.choices[0].message.content)

            # Konvertiere zu LLMResolution
            is_correct = result.get("is_factually_correct")
            confidence = result.get("confidence", 0.5)

            if is_correct is True:
                resolution = ResolutionAction.ACCEPT_NEW
            elif is_correct is False:
                resolution = ResolutionAction.REJECT_BOTH
            else:
                resolution = ResolutionAction.HUMAN_REVIEW

            return LLMResolution(
                is_valid=is_correct if is_correct is not None else False,
                confidence=confidence,
                resolution=resolution,
                reasoning=result.get("reasoning", ""),
                fact_check_result="verified" if is_correct else "refuted" if is_correct is False else "uncertain",
                evidence_found=result.get("supporting_evidence", []),
                contradictions_found=result.get("contradicting_evidence", []),
                usage_stats=usage_stats
            )

        except Exception as e:
            logger.error(f"Faktenverifikation fehlgeschlagen: {e}")
            return None

    def _gather_extended_context(self, triple: Triple, graph_repo: Any) -> str:
        """
        Sammelt erweiterten Kontext für Faktenverifikation.

        Holt mehr Relationen und auch 2-Hop Nachbarn.
        """
        if not graph_repo:
            return "Kein Graph-Zugriff verfügbar."

        context_parts = []

        try:
            # 1-Hop Relationen des Subjects
            subject_relations = graph_repo.find_relations(source_id=triple.subject.id)
            if subject_relations:
                context_parts.append(f"Bekannte Fakten über '{triple.subject.name}':")
                for rel in subject_relations[:10]:  # Max 10
                    target = rel.get("target", {})
                    context_parts.append(f"  - {rel.get('rel_type', '?')} → {target.get('name', '?')}")

            # 1-Hop Relationen zum Object
            object_relations = graph_repo.find_relations(target_id=triple.object.id)
            if object_relations:
                context_parts.append(f"\nBekannte Fakten über '{triple.object.name}':")
                for rel in object_relations[:10]:
                    source = rel.get("source", {})
                    context_parts.append(f"  - {source.get('name', '?')} → {rel.get('rel_type', '?')}")

            # Ähnliche Tripel mit gleichem Prädikat
            similar_triples = graph_repo.find_relations_by_type(triple.predicate)
            if similar_triples:
                context_parts.append(f"\nÄhnliche '{triple.predicate}'-Relationen:")
                for rel in similar_triples[:5]:
                    source = rel.get("source", {}).get("name", "?")
                    target = rel.get("target", {}).get("name", "?")
                    context_parts.append(f"  - {source} → {target}")

        except Exception as e:
            logger.debug(f"Konnte erweiterten Kontext nicht laden: {e}")

        return "\n".join(context_parts) if context_parts else "Keine relevanten Fakten im Graph gefunden."

    # Spezialisierter Prompt für semantische Widerspruchserkennung
    SEMANTIC_CONSISTENCY_PROMPT = """Du bist ein Experte für logische Konsistenz in Knowledge Graphs.

## Neues Triple (zu prüfen)
- Subject: {subject_name}
- Prädikat: {predicate}
- Object: {object_name}
- Quelltext: {source_text}

## Bekannte Fakten im Graph
{graph_context}

## Deine Aufgabe
Prüfe ob das neue Triple mit den bekannten Fakten **widerspricht**.

WICHTIG:
- Wenn keine relevanten Fakten im Graph sind → is_consistent: TRUE
- Wenn das Triple neu und unabhängig ist → is_consistent: TRUE
- Nur bei KLAREM WIDERSPRUCH zu bekannten Fakten → is_consistent: FALSE

Mögliche Widerspruchstypen:

1. **Temporale Widersprüche**: Ereignisse nach dem Tod, unmögliche Zeitabfolgen
   - Beispiel: Person stirbt 1955, gewinnt aber Preis 1960 → WIDERSPRUCH

2. **Zustandswidersprüche**: Inkompatible Zustände
   - Beispiel: Person ist tot, heiratet aber später → WIDERSPRUCH

3. **Logische Widersprüche**: Zyklische oder unmögliche Beziehungen
   - Beispiel: Ehefrau ist gleichzeitig Mutter der gleichen Person → WIDERSPRUCH

4. **Faktische Widersprüche**: Direkte Konflikte mit bekannten Fakten
   - Beispiel: Geboren in Ulm, aber neues Triple sagt geboren in München → WIDERSPRUCH

## Antwort (JSON-Format)
{{
    "is_consistent": true/false,
    "confidence": 0.0-1.0,
    "contradiction_type": null | "temporal" | "state" | "logical" | "factual",
    "reasoning": "Deine Schritt-für-Schritt Begründung...",
    "contradicting_facts": ["Fakt 1", "Fakt 2", ...]
}}
"""

    def _verify_semantic_consistency(
        self,
        triple: Triple,
        graph_repo: Any
    ) -> Optional[LLMResolution]:
        """
        Prüft auf semantische Widersprüche (temporal, logisch, Zustand).

        Wird aufgerufen wenn keine expliziten Konflikte vorliegen,
        aber das LLM semantische Inkonsistenzen erkennen soll.
        """
        # Kontext sammeln
        graph_context = self._gather_extended_context(triple, graph_repo)

        prompt = self.SEMANTIC_CONSISTENCY_PROMPT.format(
            subject_name=triple.subject.name,
            predicate=triple.predicate,
            object_name=triple.object.name,
            source_text=triple.source_text or "Nicht verfügbar",
            graph_context=graph_context
        )

        try:
            start_time = time.time()

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Du prüfst Knowledge Graph Triples auf semantische Konsistenz. "
                                   "Sei kritisch und erkenne Widersprüche. Antworte nur in validem JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            latency_ms = (time.time() - start_time) * 1000
            usage_stats = self._extract_usage_stats(response, latency_ms)
            self._record_usage(usage_stats)

            result = json.loads(response.choices[0].message.content)

            is_consistent = result.get("is_consistent", True)
            confidence = result.get("confidence", 0.5)
            contradiction_type = result.get("contradiction_type")
            reasoning = result.get("reasoning", "")
            contradicting_facts = result.get("contradicting_facts", [])

            logger.debug(f"Semantische Prüfung: consistent={is_consistent}, "
                        f"type={contradiction_type}, confidence={confidence:.0%}")

            if is_consistent:
                resolution = ResolutionAction.ACCEPT_NEW
            else:
                resolution = ResolutionAction.REJECT_BOTH

            return LLMResolution(
                is_valid=is_consistent,
                confidence=confidence,
                resolution=resolution,
                reasoning=reasoning,
                fact_check_result="consistent" if is_consistent else f"contradiction_{contradiction_type}",
                contradictions_found=contradicting_facts,
                usage_stats=usage_stats
            )

        except Exception as e:
            logger.error(f"Semantische Konsistenzprüfung fehlgeschlagen: {e}")
            return None

    def get_usage_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung der LLM-Nutzung zurück."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "avg_tokens_per_call": (
                (self.total_input_tokens + self.total_output_tokens) / max(self.total_calls, 1)
            ),
            "model": self.config.llm_model,
        }

    def reset_usage_tracking(self):
        """Setzt das Token-Tracking zurück."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.usage_history = []
    
    def _gather_context(self, triple: Triple, graph_repo: Any) -> str:
        """Sammelt Kontext aus dem Graphen für die LLM-Analyse."""
        if not graph_repo:
            return "Kein Graph-Zugriff verfügbar."
        
        context_parts = []
        
        # Existierende Relationen des Subjects
        try:
            subject_relations = graph_repo.find_relations(source_id=triple.subject.id)
            if subject_relations:
                context_parts.append(f"Relationen von '{triple.subject.name}':")
                for rel in subject_relations[:5]:  # Max 5
                    target = rel.get("target", {})
                    context_parts.append(f"  → {rel.get('rel_type', '?')} → {target.get('name', '?')}")
        except Exception as e:
            logger.debug(f"Konnte Subject-Relationen nicht laden: {e}")
        
        # Existierende Relationen zum Object
        try:
            object_relations = graph_repo.find_relations(target_id=triple.object.id)
            if object_relations:
                context_parts.append(f"\nRelationen zu '{triple.object.name}':")
                for rel in object_relations[:5]:
                    source = rel.get("source", {})
                    context_parts.append(f"  {source.get('name', '?')} → {rel.get('rel_type', '?')} →")
        except Exception as e:
            logger.debug(f"Konnte Object-Relationen nicht laden: {e}")
        
        return "\n".join(context_parts) if context_parts else "Keine existierenden Relationen gefunden."
    
    def _get_resolution_action(self, r) -> str:
        """Hilfsmethode um resolution aus Dict oder LLMResolution zu lesen."""
        if isinstance(r, dict):
            return r.get("resolution", "")
        else:
            # LLMResolution Objekt - resolution ist ein ResolutionAction Enum
            res = getattr(r, "resolution", None)
            if res is None:
                return ""
            # ResolutionAction Enum zu String konvertieren
            return res.value if hasattr(res, "value") else str(res)

    def _determine_outcome(self, resolutions: List[Dict], confidence: float) -> ValidationOutcome:
        """Bestimmt das finale Outcome basierend auf allen Resolutions."""
        if not resolutions:
            return ValidationOutcome.UNCERTAIN

        # Zähle Resolution-Typen (unterstütze sowohl Dicts als auch LLMResolution Objekte)
        accept_count = sum(1 for r in resolutions if self._get_resolution_action(r) == "accept_new")
        reject_count = sum(1 for r in resolutions if self._get_resolution_action(r) in ["keep_existing", "reject_both"])
        review_count = sum(1 for r in resolutions if self._get_resolution_action(r) == "human_review")
        
        # Entscheidungslogik
        if review_count > 0:
            return ValidationOutcome.UNCERTAIN  # Human Review nötig
        
        if accept_count > reject_count and confidence >= 0.6:
            return ValidationOutcome.PASS
        elif reject_count > accept_count:
            return ValidationOutcome.FAIL
        else:
            return ValidationOutcome.UNCERTAIN
