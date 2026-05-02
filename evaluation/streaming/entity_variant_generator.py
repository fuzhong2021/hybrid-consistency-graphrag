#!/usr/bin/env python3
# evaluation/streaming/entity_variant_generator.py
"""
Entity Variant Generator für Streaming Evaluation.

Generiert realistische Entity-Varianten für Entity Resolution Testing.

Wissenschaftliche Referenz:
- Lairgi et al. (2024): iText2KG Entity Resolution
  → α=0.6 für Name Similarity, α=0.4 für Embedding Similarity
- Jaro-Winkler Distanz für Namen-Vergleich
- Sentence Embeddings für semantische Ähnlichkeit

Varianten-Typen:
1. Abkürzungen: "Albert Einstein" → "A. Einstein"
2. Titel: "Einstein" → "Prof. Einstein", "Dr. Einstein"
3. Alias: "Albert Einstein" → "Einstein, Albert"
4. Spitznamen: Falls bekannt
5. Akronyme: "United States" → "U.S.", "USA"
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from src.models.entities import Entity, EntityType, Triple
from .triple_generator import AnnotatedTriple, TripleCategory

logger = logging.getLogger(__name__)


@dataclass
class EntityVariant:
    """Eine Entity-Variante mit Metadaten."""
    original_name: str
    variant_name: str
    variant_type: str  # "abbreviation", "title", "alias", "acronym", "nickname"
    expected_similarity: float  # Erwartete Ähnlichkeit (0-1)


@dataclass
class EntityVariantGenerator:
    """
    Generiert Entity-Varianten für Entity Resolution Testing.

    Implementiert verschiedene Varianten-Typen basierend auf
    linguistischen Regeln.

    Attribute:
        titles: Liste von Titeln für Personen
        org_suffixes: Liste von Organisations-Suffixen
        location_abbreviations: Dict von Orts-Abkürzungen
    """
    titles: List[str] = field(default_factory=lambda: [
        "Dr.", "Prof.", "Mr.", "Mrs.", "Ms.", "Sir", "Lord", "Lady",
        "President", "Senator", "Governor", "King", "Queen", "Prince", "Princess"
    ])

    org_suffixes: List[str] = field(default_factory=lambda: [
        "Inc.", "Inc", "Corp.", "Corp", "Ltd.", "Ltd", "LLC", "GmbH",
        "Co.", "Company", "Corporation", "Foundation", "Institute"
    ])

    location_abbreviations: Dict[str, str] = field(default_factory=lambda: {
        "United States": "U.S.",
        "United Kingdom": "U.K.",
        "United States of America": "USA",
        "United Arab Emirates": "UAE",
        "European Union": "EU",
        "New York City": "NYC",
        "Los Angeles": "LA",
        "San Francisco": "SF",
    })

    def generate_variants(
        self,
        name: str,
        entity_type: EntityType = EntityType.PERSON,
        max_variants: int = 3
    ) -> List[EntityVariant]:
        """
        Generiert Varianten eines Entity-Namens.

        Args:
            name: Original-Name
            entity_type: Typ der Entity
            max_variants: Maximale Anzahl Varianten

        Returns:
            Liste von EntityVariant Objekten
        """
        variants = []

        # Typ-spezifische Generierung
        if entity_type == EntityType.PERSON:
            variants.extend(self._generate_person_variants(name))
        elif entity_type == EntityType.ORGANIZATION:
            variants.extend(self._generate_org_variants(name))
        elif entity_type == EntityType.LOCATION:
            variants.extend(self._generate_location_variants(name))
        else:
            # Generische Varianten
            variants.extend(self._generate_generic_variants(name))

        # Begrenze auf max_variants
        return variants[:max_variants]

    def _generate_person_variants(self, name: str) -> List[EntityVariant]:
        """Generiert Varianten für Personennamen."""
        variants = []
        parts = name.split()

        if len(parts) < 2:
            # Nur ein Wort - wenige Varianten möglich
            # Titel hinzufügen
            for title in ["Dr.", "Prof."]:
                variants.append(EntityVariant(
                    original_name=name,
                    variant_name=f"{title} {name}",
                    variant_type="title",
                    expected_similarity=0.75
                ))
            return variants

        first_name = parts[0]
        last_name = parts[-1]
        middle_parts = parts[1:-1] if len(parts) > 2 else []

        # 1. Abkürzung des Vornamens: "Albert Einstein" → "A. Einstein"
        initial = first_name[0] + "."
        variants.append(EntityVariant(
            original_name=name,
            variant_name=f"{initial} {last_name}",
            variant_type="abbreviation",
            expected_similarity=0.70
        ))

        # 2. Nur Nachname: "Einstein"
        variants.append(EntityVariant(
            original_name=name,
            variant_name=last_name,
            variant_type="abbreviation",
            expected_similarity=0.60
        ))

        # 3. Nachname, Vorname: "Einstein, Albert"
        variants.append(EntityVariant(
            original_name=name,
            variant_name=f"{last_name}, {first_name}",
            variant_type="alias",
            expected_similarity=0.85
        ))

        # 4. Mit Titel
        if not any(name.startswith(t) for t in self.titles):
            title = "Prof." if len(name) > 10 else "Dr."
            variants.append(EntityVariant(
                original_name=name,
                variant_name=f"{title} {name}",
                variant_type="title",
                expected_similarity=0.80
            ))

        # 5. Initialen für Mittelnamen
        if middle_parts:
            middle_initials = " ".join(p[0] + "." for p in middle_parts)
            variants.append(EntityVariant(
                original_name=name,
                variant_name=f"{first_name} {middle_initials} {last_name}",
                variant_type="abbreviation",
                expected_similarity=0.90
            ))

        return variants

    def _generate_org_variants(self, name: str) -> List[EntityVariant]:
        """Generiert Varianten für Organisationsnamen."""
        variants = []

        # 1. Suffix entfernen/hinzufügen
        for suffix in self.org_suffixes:
            if name.endswith(suffix):
                # Suffix entfernen
                base_name = name[:-len(suffix)].strip()
                variants.append(EntityVariant(
                    original_name=name,
                    variant_name=base_name,
                    variant_type="abbreviation",
                    expected_similarity=0.85
                ))
                break
        else:
            # Suffix hinzufügen
            variants.append(EntityVariant(
                original_name=name,
                variant_name=f"{name} Inc.",
                variant_type="alias",
                expected_similarity=0.85
            ))

        # 2. Akronym generieren
        words = [w for w in name.split() if w[0].isupper() and w.lower() not in ["the", "of", "and"]]
        if len(words) >= 2:
            acronym = "".join(w[0].upper() for w in words)
            if len(acronym) >= 2:
                variants.append(EntityVariant(
                    original_name=name,
                    variant_name=acronym,
                    variant_type="acronym",
                    expected_similarity=0.50
                ))

        # 3. "The" entfernen/hinzufügen
        if name.startswith("The "):
            variants.append(EntityVariant(
                original_name=name,
                variant_name=name[4:],
                variant_type="alias",
                expected_similarity=0.95
            ))
        else:
            variants.append(EntityVariant(
                original_name=name,
                variant_name=f"The {name}",
                variant_type="alias",
                expected_similarity=0.95
            ))

        return variants

    def _generate_location_variants(self, name: str) -> List[EntityVariant]:
        """Generiert Varianten für Ortsnamen."""
        variants = []

        # 1. Bekannte Abkürzungen
        if name in self.location_abbreviations:
            variants.append(EntityVariant(
                original_name=name,
                variant_name=self.location_abbreviations[name],
                variant_type="acronym",
                expected_similarity=0.75
            ))

        # 2. "City/State" entfernen
        for suffix in [" City", " State", " Province", " County"]:
            if name.endswith(suffix):
                variants.append(EntityVariant(
                    original_name=name,
                    variant_name=name[:-len(suffix)],
                    variant_type="abbreviation",
                    expected_similarity=0.85
                ))
                break

        # 3. Akronym für mehrteilige Namen
        parts = name.split()
        if len(parts) >= 2 and len(parts) <= 4:
            # Versuche Akronym
            acronym = "".join(p[0].upper() for p in parts if p[0].isupper())
            if len(acronym) >= 2:
                variants.append(EntityVariant(
                    original_name=name,
                    variant_name=acronym,
                    variant_type="acronym",
                    expected_similarity=0.55
                ))

        return variants

    def _generate_generic_variants(self, name: str) -> List[EntityVariant]:
        """Generiert generische Varianten."""
        variants = []
        parts = name.split()

        # 1. Erstes Wort
        if len(parts) > 1:
            variants.append(EntityVariant(
                original_name=name,
                variant_name=parts[0],
                variant_type="abbreviation",
                expected_similarity=0.55
            ))

        # 2. Ohne Artikel
        if parts[0].lower() in ["the", "a", "an"]:
            variants.append(EntityVariant(
                original_name=name,
                variant_name=" ".join(parts[1:]),
                variant_type="alias",
                expected_similarity=0.90
            ))

        return variants

    def generate_variant_triples(
        self,
        base_triples: List[AnnotatedTriple],
        variants_per_entity: int = 2
    ) -> List[AnnotatedTriple]:
        """
        Generiert Triples mit Entity-Varianten für Merge-Testing.

        Args:
            base_triples: Basis-Triples (typischerweise SUPPORTS)
            variants_per_entity: Anzahl Varianten pro Entity

        Returns:
            Liste von AnnotatedTriples mit ground_truth_merge=True
        """
        variant_triples = []
        processed_entities: Set[str] = set()

        for base in base_triples:
            # Nur SUPPORTS Triples als Basis verwenden
            if base.category != TripleCategory.SUPPORTS:
                continue

            # Subject-Varianten
            subj_name = base.triple.subject.name
            if subj_name.lower() not in processed_entities:
                processed_entities.add(subj_name.lower())

                variants = self.generate_variants(
                    subj_name,
                    base.triple.subject.entity_type,
                    max_variants=variants_per_entity
                )

                for variant in variants:
                    # Neue Entity mit Varianten-Namen
                    variant_entity = Entity(
                        name=variant.variant_name,
                        entity_type=base.triple.subject.entity_type,
                        source_document=f"variant_{base.original_claim_id}"
                    )

                    # Neues Triple mit Variante als Subject
                    variant_triple = Triple(
                        subject=variant_entity,
                        predicate=base.triple.predicate,
                        object=base.triple.object,
                        source_text=f"[ENTITY VARIANT: {variant.variant_type}] {base.triple.source_text}",
                        source_document_id=f"variant_{base.original_claim_id}",
                        extraction_confidence=0.85
                    )

                    annotated = AnnotatedTriple(
                        triple=variant_triple,
                        category=TripleCategory.ENTITY_VARIANT,
                        ground_truth_accept=True,  # Sollte akzeptiert werden nach Merge
                        ground_truth_merge=True,   # WICHTIG: Sollte gemergt werden!
                        original_claim_id=base.original_claim_id,
                        evidence_text=base.evidence_text,
                        nli_label="ENTAILMENT"
                    )
                    variant_triples.append(annotated)

            # Object-Varianten (optional, für ausgewogenen Test)
            obj_name = base.triple.object.name
            if obj_name.lower() not in processed_entities and len(obj_name.split()) >= 2:
                processed_entities.add(obj_name.lower())

                variants = self.generate_variants(
                    obj_name,
                    base.triple.object.entity_type,
                    max_variants=1  # Weniger Object-Varianten
                )

                for variant in variants:
                    variant_entity = Entity(
                        name=variant.variant_name,
                        entity_type=base.triple.object.entity_type,
                        source_document=f"variant_{base.original_claim_id}"
                    )

                    variant_triple = Triple(
                        subject=base.triple.subject,
                        predicate=base.triple.predicate,
                        object=variant_entity,
                        source_text=f"[ENTITY VARIANT: {variant.variant_type}] {base.triple.source_text}",
                        source_document_id=f"variant_{base.original_claim_id}",
                        extraction_confidence=0.85
                    )

                    annotated = AnnotatedTriple(
                        triple=variant_triple,
                        category=TripleCategory.ENTITY_VARIANT,
                        ground_truth_accept=True,
                        ground_truth_merge=True,
                        original_claim_id=base.original_claim_id,
                        evidence_text=base.evidence_text,
                        nli_label="ENTAILMENT"
                    )
                    variant_triples.append(annotated)

        logger.info(f"Generiert: {len(variant_triples)} Entity-Varianten-Triples")
        return variant_triples

    def calculate_expected_similarity(
        self,
        original: str,
        variant: str,
        name_weight: float = 0.6,
        embedding_weight: float = 0.4
    ) -> float:
        """
        Berechnet erwartete Ähnlichkeit nach iText2KG Formel.

        similarity = α * name_similarity + (1-α) * embedding_similarity

        Da wir kein Embedding haben, schätzen wir basierend auf
        der Varianten-Charakteristik.
        """
        # Jaro-Winkler Approximation
        name_sim = self._jaro_winkler_sim(original.lower(), variant.lower())

        # Embedding-Ähnlichkeit schätzen basierend auf Überlappung
        words_orig = set(original.lower().split())
        words_var = set(variant.lower().split())
        overlap = len(words_orig & words_var)
        union = len(words_orig | words_var)
        embedding_sim = overlap / union if union > 0 else 0.0

        return name_weight * name_sim + embedding_weight * embedding_sim

    def _jaro_winkler_sim(self, s1: str, s2: str) -> float:
        """Vereinfachte Jaro-Winkler Ähnlichkeit."""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Match window
        match_dist = max(len1, len2) // 2 - 1
        if match_dist < 0:
            match_dist = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_dist)
            end = min(i + match_dist + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

        # Winkler prefix bonus
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + 0.1 * prefix * (1 - jaro)
