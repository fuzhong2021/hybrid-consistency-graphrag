#!/usr/bin/env python3
# evaluation/streaming/triple_generator.py
"""
FEVER Triple Generator für Streaming Evaluation.

Konvertiert FEVER Claims zu Triples für die Konsistenzprüfung.

WICHTIG: Keine Kardinalitätsregeln verwenden!
- FEVER Relationen nutzen generische Prädikate (CLAIMS, STATES, etc.)
- Widersprüche werden über NLI erkannt, nicht über Kardinalität
- Das demonstriert den Mehrwert des Hybrid-Systems

Wissenschaftliche Referenz:
- Thorne et al. (2018): FEVER Dataset
- Natural Language Processing Pipelines für Claim Extraction
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from src.models.entities import Entity, EntityType, Triple

logger = logging.getLogger(__name__)


class TripleCategory(Enum):
    """Kategorie eines generierten Triples."""
    SUPPORTS = "supports"      # FEVER SUPPORTS → Ground Truth: ACCEPT
    REFUTES = "refutes"        # FEVER REFUTES → Ground Truth: REJECT
    ENTITY_VARIANT = "entity_variant"  # Entitätsvariante → Ground Truth: MERGE
    CROSS_DOC = "cross_doc"    # Cross-Doc Konflikt → Ground Truth: REJECT


@dataclass
class AnnotatedTriple:
    """Triple mit Ground Truth Annotation."""
    triple: Triple
    category: TripleCategory
    ground_truth_accept: bool
    ground_truth_merge: bool = False  # Für Entity-Varianten
    original_claim_id: str = ""
    evidence_text: str = ""
    nli_label: str = ""  # ENTAILMENT, CONTRADICTION, NEUTRAL

    @property
    def should_accept(self) -> bool:
        """Ground Truth: Sollte das Triple akzeptiert werden?"""
        return self.ground_truth_accept

    @property
    def should_reject(self) -> bool:
        """Ground Truth: Sollte das Triple abgelehnt werden?"""
        return not self.ground_truth_accept and not self.ground_truth_merge

    @property
    def should_merge(self) -> bool:
        """Ground Truth: Sollte das Triple mit existierendem gemergt werden?"""
        return self.ground_truth_merge


@dataclass
class FEVERTripleGenerator:
    """
    Generiert Triples aus FEVER Claims für Streaming Evaluation.

    Schlüsselidee:
    - SUPPORTS Claims → sollten AKZEPTIERT werden
    - REFUTES Claims → sollten via NLI als Widerspruch erkannt werden
    - KEIN Kardinalitätsbasierter Ansatz!

    Attribute:
        min_claim_words: Minimum Wortzahl für valide Claims
        entity_extractor: Optional, externes NER Modell
    """
    min_claim_words: int = 4
    relation_types: List[str] = field(default_factory=lambda: [
        "CLAIMS", "STATES", "DESCRIBES", "MENTIONS", "ASSERTS"
    ])

    def __post_init__(self):
        self._entity_cache: Dict[str, Entity] = {}

    def extract_subject_object(
        self,
        claim: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extrahiert Subject, Prädikat und Object aus einem Claim.

        Verwendet regelbasierte Heuristiken:
        1. Erstes Nomen/Eigenname als Subject
        2. Verb als Prädikat
        3. Rest als Object

        Returns:
            Tuple von (subject, predicate, object) oder (None, None, None)
        """
        if not claim or len(claim.split()) < self.min_claim_words:
            return None, None, None

        # Einfache Heuristik: Erstes Wort oder Phrase als Subject
        words = claim.split()

        # Subject: Kapitalisierte Wörter am Anfang (bis zum ersten Kleinbuchstaben-Wort)
        subject_words = []
        i = 0
        while i < len(words):
            word = words[i]
            # Artikel überspringen
            if word.lower() in ["the", "a", "an"]:
                subject_words.append(word)
                i += 1
                continue
            # Großgeschriebene Wörter sind Teil des Subjects
            if word[0].isupper():
                subject_words.append(word)
                i += 1
            else:
                break

        if not subject_words:
            # Fallback: Erste 1-3 Wörter
            subject_words = words[:min(3, len(words))]
            i = len(subject_words)

        subject = " ".join(subject_words)

        # Object: Letzte 2-4 Wörter oder nach "is/was/are/were"
        remaining = words[i:]
        if not remaining:
            return None, None, None

        # Suche nach Kopulaverben
        copulas = ["is", "was", "are", "were", "has", "had", "have", "became", "becomes"]
        obj_start = 0
        predicate = "STATES"

        for j, word in enumerate(remaining):
            if word.lower() in copulas:
                predicate = word.upper()
                obj_start = j + 1
                break

        if obj_start >= len(remaining):
            obj_start = max(0, len(remaining) - 4)

        object_text = " ".join(remaining[obj_start:])

        # Bereinige Object
        object_text = re.sub(r'[.,!?;:]+$', '', object_text)

        if len(object_text) < 2:
            return None, None, None

        return subject, predicate, object_text

    def _get_or_create_entity(
        self,
        name: str,
        source_doc: Optional[str] = None
    ) -> Entity:
        """Holt oder erstellt eine Entity."""
        cache_key = name.lower().strip()
        if cache_key not in self._entity_cache:
            entity_type = self._infer_entity_type(name)
            entity = Entity(
                name=name.strip(),
                entity_type=entity_type,
                source_document=source_doc
            )
            self._entity_cache[cache_key] = entity
        return self._entity_cache[cache_key]

    def _infer_entity_type(self, name: str) -> EntityType:
        """Heuristik für Entity-Typ."""
        name_lower = name.lower()

        # Location Keywords
        if any(kw in name_lower for kw in [
            "city", "country", "state", "province", "river", "mountain",
            "lake", "ocean", "island", "continent", "valley"
        ]):
            return EntityType.LOCATION

        # Organization Keywords
        if any(kw in name_lower for kw in [
            "university", "company", "inc", "ltd", "corp", "school",
            "band", "team", "organization", "association", "institute",
            "foundation", "agency", "department"
        ]):
            return EntityType.ORGANIZATION

        # Event Keywords
        if any(kw in name_lower for kw in [
            "war", "battle", "championship", "election", "festival",
            "award", "ceremony", "tournament", "olympics", "concert"
        ]):
            return EntityType.EVENT

        # Concept Keywords
        if any(kw in name_lower for kw in [
            "film", "movie", "album", "song", "book", "novel", "series",
            "theory", "law", "principle", "concept"
        ]):
            return EntityType.CONCEPT

        # Default: PERSON (viele FEVER Claims handeln von Personen)
        return EntityType.PERSON

    def generate_from_claim(
        self,
        claim_id: str,
        claim_text: str,
        label: str,
        evidence_text: str = ""
    ) -> Optional[AnnotatedTriple]:
        """
        Generiert ein annotiertes Triple aus einem FEVER Claim.

        Args:
            claim_id: ID des Claims
            claim_text: Der Claim-Text
            label: FEVER Label (SUPPORTS, REFUTES, NOT ENOUGH INFO)
            evidence_text: Optional, der Evidence-Text

        Returns:
            AnnotatedTriple oder None wenn Extraktion fehlschlägt
        """
        # Parse Claim
        subject, predicate, object_text = self.extract_subject_object(claim_text)

        if not subject or not object_text:
            logger.debug(f"Konnte Claim nicht parsen: {claim_text[:50]}...")
            return None

        # Erstelle Entities
        subject_entity = self._get_or_create_entity(subject, claim_id)
        object_entity = self._get_or_create_entity(object_text, claim_id)

        # Wähle Relation ohne Kardinalität
        relation = predicate if predicate in self.relation_types else "CLAIMS"

        # Erstelle Triple
        triple = Triple(
            subject=subject_entity,
            predicate=relation,
            object=object_entity,
            source_text=evidence_text if evidence_text else claim_text,
            source_document_id=claim_id,
            extraction_confidence=0.85
        )

        # Ground Truth basierend auf FEVER Label
        if label == "SUPPORTS":
            category = TripleCategory.SUPPORTS
            ground_truth_accept = True
            nli_label = "ENTAILMENT"
        elif label == "REFUTES":
            category = TripleCategory.REFUTES
            ground_truth_accept = False
            nli_label = "CONTRADICTION"
        else:  # NOT ENOUGH INFO
            # NEI Claims werden nicht verwendet - zu unsicher
            return None

        return AnnotatedTriple(
            triple=triple,
            category=category,
            ground_truth_accept=ground_truth_accept,
            ground_truth_merge=False,
            original_claim_id=claim_id,
            evidence_text=evidence_text,
            nli_label=nli_label
        )

    def generate_batch(
        self,
        claims: List[Dict[str, Any]],
        include_nei: bool = False
    ) -> List[AnnotatedTriple]:
        """
        Generiert Triples aus einer Liste von Claims.

        Args:
            claims: Liste von Claim-Dictionaries mit 'id', 'claim', 'label', 'evidence'
            include_nei: Wenn True, werden NOT ENOUGH INFO Claims einbezogen

        Returns:
            Liste von AnnotatedTriples
        """
        results = []

        for claim_data in claims:
            label = claim_data.get("label", "")

            # Filtere NEI wenn nicht gewünscht
            if not include_nei and label == "NOT ENOUGH INFO":
                continue

            # Extrahiere Evidence-Text
            evidence_text = ""
            evidence = claim_data.get("evidence", [])
            if evidence:
                if isinstance(evidence, list) and len(evidence) > 0:
                    ev = evidence[0]
                    if isinstance(ev, list) and len(ev) >= 3:
                        evidence_text = ev[2]
                    elif isinstance(ev, dict):
                        evidence_text = ev.get("text", "")

            annotated = self.generate_from_claim(
                claim_id=str(claim_data.get("id", "")),
                claim_text=claim_data.get("claim", ""),
                label=label,
                evidence_text=evidence_text
            )

            if annotated:
                results.append(annotated)

        logger.info(f"Generiert: {len(results)} Triples aus {len(claims)} Claims")
        return results

    def get_statistics(
        self,
        triples: List[AnnotatedTriple]
    ) -> Dict[str, Any]:
        """Gibt Statistiken über generierte Triples."""
        if not triples:
            return {"total": 0}

        supports = sum(1 for t in triples if t.category == TripleCategory.SUPPORTS)
        refutes = sum(1 for t in triples if t.category == TripleCategory.REFUTES)
        entity_variants = sum(1 for t in triples if t.category == TripleCategory.ENTITY_VARIANT)
        cross_doc = sum(1 for t in triples if t.category == TripleCategory.CROSS_DOC)

        return {
            "total": len(triples),
            "supports": supports,
            "refutes": refutes,
            "entity_variants": entity_variants,
            "cross_doc": cross_doc,
            "should_accept": sum(1 for t in triples if t.should_accept),
            "should_reject": sum(1 for t in triples if t.should_reject),
            "should_merge": sum(1 for t in triples if t.should_merge),
        }

    def clear_cache(self):
        """Leert den Entity-Cache."""
        self._entity_cache.clear()
