#!/usr/bin/env python3
# evaluation/streaming/cross_doc_generator.py
"""
Cross-Document Conflict Generator für Streaming Evaluation.

Generiert semantische Widersprüche über Dokumente hinweg.

KRITISCHER UNTERSCHIED zu HotpotQA:
- KEINE Kardinalitätsregeln!
- Widersprüche müssen über NLI erkannt werden
- Das testet die semantische Analyse des Hybrid-Systems

Beispiel:
    Dokument A: "Einstein wurde in Ulm geboren"
    Dokument B: "Einstein wurde in München geboren"
    → NLI erkennt: CONTRADICTION (ohne Kardinalität!)

Wissenschaftliche Referenz:
- Thorne et al. (2018): FEVER - Evidence-based Fact Verification
- Williams et al. (2018): MultiNLI - Cross-Domain NLI
"""

import random
import logging
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from src.models.entities import Entity, EntityType, Triple
from .triple_generator import AnnotatedTriple, TripleCategory

logger = logging.getLogger(__name__)


@dataclass
class ConflictPattern:
    """Ein Muster für generierte Konflikte."""
    subject: str
    predicate: str
    original_object: str
    conflicting_object: str
    conflict_type: str  # "factual", "temporal", "quantitative", "categorical"


@dataclass
class CrossDocConflictGenerator:
    """
    Generiert Cross-Document Konflikte für NLI-basierte Widerspruchserkennung.

    WICHTIG: Keine Kardinalitätsregeln!
    Die Konflikte müssen durch semantische Analyse (NLI) erkannt werden.

    Attribute:
        conflict_templates: Templates für Konflikt-Generierung
        seed: Random seed für Reproduzierbarkeit
    """
    conflict_templates: Dict[str, List[Tuple[str, str]]] = field(default_factory=lambda: {
        # Faktische Konflikte (Geburtsort, Nationalität, etc.)
        "birth_location": [
            ("Ulm", "Munich"),
            ("Berlin", "Vienna"),
            ("London", "Paris"),
            ("New York", "Boston"),
            ("Tokyo", "Osaka"),
        ],
        # Zeitliche Konflikte (Jahreszahlen, Zeiträume)
        "dates": [
            ("1879", "1880"),
            ("1920", "1921"),
            ("19th century", "20th century"),
            ("1945", "1946"),
        ],
        # Quantitative Konflikte
        "quantities": [
            ("three", "four"),
            ("first", "second"),
            ("100", "200"),
            ("single", "multiple"),
        ],
        # Kategoriale Konflikte
        "categories": [
            ("physicist", "chemist"),
            ("American", "British"),
            ("actor", "director"),
            ("novel", "short story"),
        ],
    })
    seed: int = 42

    def __post_init__(self):
        random.seed(self.seed)
        self._entity_cache: Dict[str, Entity] = {}

    def _get_or_create_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.CONCEPT,
        source_doc: Optional[str] = None
    ) -> Entity:
        """Holt oder erstellt eine Entity."""
        cache_key = f"{name.lower().strip()}_{entity_type.value}"
        if cache_key not in self._entity_cache:
            entity = Entity(
                name=name.strip(),
                entity_type=entity_type,
                source_document=source_doc
            )
            self._entity_cache[cache_key] = entity
        return self._entity_cache[cache_key]

    def generate_conflicting_object(
        self,
        original_object: str,
        conflict_type: str = "factual"
    ) -> Optional[str]:
        """
        Generiert ein widersprüchliches Objekt.

        Args:
            original_object: Das ursprüngliche Objekt
            conflict_type: Art des Konflikts

        Returns:
            Ein widersprüchliches Objekt oder None
        """
        obj_lower = original_object.lower()

        # Versuche passenden Konflikt aus Templates
        for template_type, pairs in self.conflict_templates.items():
            for orig, conflict in pairs:
                if orig.lower() in obj_lower:
                    return original_object.replace(orig, conflict)
                if conflict.lower() in obj_lower:
                    return original_object.replace(conflict, orig)

        # Generischer Konflikt: Negation oder alternative Formulierung
        if any(word in obj_lower for word in ["true", "correct", "yes", "is"]):
            return original_object.replace("true", "false").replace("correct", "incorrect").replace("yes", "no").replace(" is ", " is not ")

        # Für Namen: Leichte Variation
        parts = original_object.split()
        if len(parts) >= 2:
            # Tausche Reihenfolge
            return " ".join(reversed(parts))

        return None

    def generate_cross_doc_conflicts(
        self,
        base_triples: List[AnnotatedTriple],
        conflict_ratio: float = 0.3
    ) -> List[AnnotatedTriple]:
        """
        Generiert Cross-Document Konflikte aus Basis-Triples.

        OHNE KARDINALITÄTSREGELN - nur über NLI erkennbar!

        Args:
            base_triples: Basis-Triples (typischerweise SUPPORTS)
            conflict_ratio: Anteil der Basis-Triples die Konflikte bekommen

        Returns:
            Liste von Konflikt-Triples mit ground_truth_accept=False
        """
        conflict_triples = []
        supports_triples = [t for t in base_triples if t.category == TripleCategory.SUPPORTS]

        # Wähle zufällige Triples für Konfliktgenerierung
        n_conflicts = max(1, int(len(supports_triples) * conflict_ratio))
        selected = random.sample(supports_triples, min(n_conflicts, len(supports_triples)))

        for base in selected:
            # Generiere konfliktierendes Objekt
            conflicting_obj = self.generate_conflicting_object(
                base.triple.object.name
            )

            if not conflicting_obj:
                continue

            # Erstelle konfliktierendes Triple
            # WICHTIG: Gleiche Entity, gleiche Relation, aber ANDERER Wert
            # Das ist ein semantischer Widerspruch, KEINE Kardinalitätsverletzung!
            conflict_entity = self._get_or_create_entity(
                conflicting_obj,
                base.triple.object.entity_type,
                f"cross_doc_{base.original_claim_id}"
            )

            # WICHTIG: Neue Relation verwenden um Kardinalitätsprüfung zu umgehen!
            # Wir nutzen STATES statt der Original-Relation
            conflict_triple = Triple(
                subject=base.triple.subject,
                predicate="STATES",  # Generische Relation ohne Kardinalitätsregel
                object=conflict_entity,
                source_text=f"[CROSS-DOC CONFLICT] {base.triple.source_text}",
                source_document_id=f"cross_doc_{base.original_claim_id}",
                extraction_confidence=0.85
            )

            annotated = AnnotatedTriple(
                triple=conflict_triple,
                category=TripleCategory.CROSS_DOC,
                ground_truth_accept=False,  # SOLLTE ABGELEHNT WERDEN
                ground_truth_merge=False,
                original_claim_id=base.original_claim_id,
                evidence_text=f"Conflict: {base.triple.object.name} vs {conflicting_obj}",
                nli_label="CONTRADICTION"
            )
            conflict_triples.append(annotated)

        logger.info(f"Generiert: {len(conflict_triples)} Cross-Doc-Konflikt-Triples")
        return conflict_triples

    def generate_semantic_contradictions(
        self,
        base_triples: List[AnnotatedTriple],
        num_contradictions: int = 20
    ) -> List[AnnotatedTriple]:
        """
        Generiert semantische Widersprüche die NUR über NLI erkennbar sind.

        Diese Methode erstellt Widersprüche die:
        - Dieselbe Entity referenzieren
        - Aber widersprüchliche Aussagen machen
        - KEINE Kardinalitätsregel verletzen

        Beispiel:
        Triple 1: "Einstein STATES born in Ulm"
        Triple 2: "Einstein STATES born in Munich"
        → Beide haben Relation "STATES" (keine Kardinalität=1)
        → NLI muss erkennen: "born in Ulm" vs "born in Munich" = CONTRADICTION

        Args:
            base_triples: Basis-Triples
            num_contradictions: Anzahl zu generierender Widersprüche

        Returns:
            Liste von Widerspruchs-Triples
        """
        contradiction_triples = []
        supports_triples = [t for t in base_triples if t.category == TripleCategory.SUPPORTS]

        if not supports_triples:
            return contradiction_triples

        # Wähle Triples für Widerspruchsgenerierung
        selected = random.sample(
            supports_triples,
            min(num_contradictions, len(supports_triples))
        )

        for base in selected:
            # Generiere semantisch widersprüchliche Aussage
            contradiction = self._generate_semantic_contradiction(base)
            if contradiction:
                contradiction_triples.append(contradiction)

        logger.info(f"Generiert: {len(contradiction_triples)} semantische Widersprüche")
        return contradiction_triples

    def _generate_semantic_contradiction(
        self,
        base: AnnotatedTriple
    ) -> Optional[AnnotatedTriple]:
        """
        Generiert einen semantischen Widerspruch zu einem Triple.

        Strategien:
        1. Objekt-Negation: "X is true" → "X is false"
        2. Alternativer Wert: "born in A" → "born in B"
        3. Eigenschafts-Widerspruch: "is tall" → "is short"
        """
        # Versuche alternatives Objekt zu generieren
        original_obj = base.triple.object.name
        source_text = base.triple.source_text

        # Strategie 1: Direkte Negation im source_text
        negated_text = self._negate_statement(source_text)

        # Strategie 2: Alternatives Objekt
        alt_object = self._generate_alternative_object(original_obj)

        if not alt_object:
            return None

        # Erstelle widersprüchliche Entity
        contradiction_entity = Entity(
            name=alt_object,
            entity_type=base.triple.object.entity_type,
            source_document=f"contradiction_{base.original_claim_id}"
        )

        # WICHTIG: Generische Relation ohne Kardinalität
        # Das zwingt das System zur NLI-basierten Erkennung
        contradiction_triple = Triple(
            subject=base.triple.subject,
            predicate="ASSERTS",  # Keine Kardinalitätsregel!
            object=contradiction_entity,
            source_text=negated_text,
            source_document_id=f"contradiction_{base.original_claim_id}",
            extraction_confidence=0.85
        )

        return AnnotatedTriple(
            triple=contradiction_triple,
            category=TripleCategory.CROSS_DOC,
            ground_truth_accept=False,
            ground_truth_merge=False,
            original_claim_id=base.original_claim_id,
            evidence_text=negated_text,
            nli_label="CONTRADICTION"
        )

    def _negate_statement(self, text: str) -> str:
        """Negiert eine Aussage."""
        # Einfache Negations-Regeln
        negations = [
            (" is ", " is not "),
            (" was ", " was not "),
            (" are ", " are not "),
            (" were ", " were not "),
            (" has ", " has not "),
            (" have ", " have not "),
            (" can ", " cannot "),
            (" will ", " will not "),
        ]

        for pos, neg in negations:
            if pos in text:
                return text.replace(pos, neg, 1)

        # Fallback: "not" einfügen
        words = text.split()
        if len(words) >= 3:
            words.insert(2, "not")
        return " ".join(words)

    def _generate_alternative_object(self, original: str) -> Optional[str]:
        """Generiert ein alternatives (widersprüchliches) Objekt."""
        obj_lower = original.lower()

        # Check templates
        for pairs in self.conflict_templates.values():
            for a, b in pairs:
                if a.lower() == obj_lower:
                    return b
                if b.lower() == obj_lower:
                    return a

        # Generische Alternativen
        alternatives = {
            "true": "false",
            "false": "true",
            "yes": "no",
            "no": "yes",
            "first": "last",
            "last": "first",
            "best": "worst",
            "worst": "best",
            "largest": "smallest",
            "smallest": "largest",
        }

        for orig, alt in alternatives.items():
            if orig in obj_lower:
                return original.lower().replace(orig, alt).title()

        # Fallback: Zufällige Orte/Daten
        fallback_objects = [
            "an unknown location",
            "a different date",
            "another person",
            "somewhere else",
            "a different time",
        ]
        return random.choice(fallback_objects)

    def clear_cache(self):
        """Leert den Entity-Cache."""
        self._entity_cache.clear()
