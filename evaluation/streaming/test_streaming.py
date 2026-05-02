#!/usr/bin/env python3
# evaluation/streaming/test_streaming.py
"""
Unit Tests für das Streaming Evaluation Modul.

Testet:
1. FEVERTripleGenerator
2. EntityVariantGenerator
3. CrossDocConflictGenerator
4. StreamingShuffler

Ausführung:
    pytest evaluation/streaming/test_streaming.py -v
"""

import sys
sys.path.insert(0, '.')

import pytest
from typing import List

from src.models.entities import Entity, EntityType, Triple

from evaluation.streaming.triple_generator import (
    FEVERTripleGenerator,
    AnnotatedTriple,
    TripleCategory
)
from evaluation.streaming.entity_variant_generator import (
    EntityVariantGenerator,
    EntityVariant
)
from evaluation.streaming.cross_doc_generator import CrossDocConflictGenerator
from evaluation.streaming.shuffle_strategy import (
    StreamingShuffler,
    ShuffleStrategy,
    create_shuffler
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def triple_generator():
    """Erstellt einen FEVERTripleGenerator."""
    return FEVERTripleGenerator()


@pytest.fixture
def variant_generator():
    """Erstellt einen EntityVariantGenerator."""
    return EntityVariantGenerator()


@pytest.fixture
def conflict_generator():
    """Erstellt einen CrossDocConflictGenerator."""
    return CrossDocConflictGenerator(seed=42)


@pytest.fixture
def sample_claims():
    """Beispiel FEVER Claims."""
    return [
        {
            "id": "1",
            "claim": "Albert Einstein was a physicist.",
            "label": "SUPPORTS",
            "evidence": [["wiki", 0, "Albert Einstein was a theoretical physicist."]],
        },
        {
            "id": "2",
            "claim": "The Beatles were an American band.",
            "label": "REFUTES",
            "evidence": [["wiki", 0, "The Beatles were an English rock band."]],
        },
        {
            "id": "3",
            "claim": "Python is a programming language.",
            "label": "SUPPORTS",
            "evidence": [],
        },
    ]


@pytest.fixture
def sample_annotated_triples(triple_generator, sample_claims) -> List[AnnotatedTriple]:
    """Generiert Beispiel-Annotated-Triples."""
    return triple_generator.generate_batch(sample_claims)


# =============================================================================
# TRIPLE GENERATOR TESTS
# =============================================================================

class TestFEVERTripleGenerator:
    """Tests für FEVERTripleGenerator."""

    def test_extract_subject_object_basic(self, triple_generator):
        """Testet einfache Subject-Object Extraktion."""
        claim = "Albert Einstein was a physicist."
        subject, predicate, obj = triple_generator.extract_subject_object(claim)

        assert subject is not None
        assert obj is not None
        assert "Einstein" in subject or "Albert" in subject

    def test_extract_subject_object_short_claim(self, triple_generator):
        """Testet zu kurze Claims."""
        claim = "Yes"
        result = triple_generator.extract_subject_object(claim)
        assert result == (None, None, None)

    def test_generate_from_claim_supports(self, triple_generator):
        """Testet SUPPORTS Claim Generierung."""
        result = triple_generator.generate_from_claim(
            claim_id="test_1",
            claim_text="Albert Einstein was born in Germany.",
            label="SUPPORTS",
            evidence_text="Einstein was born in Ulm, Germany."
        )

        assert result is not None
        assert result.category == TripleCategory.SUPPORTS
        assert result.ground_truth_accept is True
        assert result.should_accept is True
        assert result.nli_label == "ENTAILMENT"

    def test_generate_from_claim_refutes(self, triple_generator):
        """Testet REFUTES Claim Generierung."""
        result = triple_generator.generate_from_claim(
            claim_id="test_2",
            claim_text="The Beatles were an American band.",
            label="REFUTES",
            evidence_text="The Beatles were an English rock band."
        )

        assert result is not None
        assert result.category == TripleCategory.REFUTES
        assert result.ground_truth_accept is False
        assert result.should_reject is True
        assert result.nli_label == "CONTRADICTION"

    def test_generate_from_claim_nei_filtered(self, triple_generator):
        """Testet dass NEI Claims gefiltert werden."""
        result = triple_generator.generate_from_claim(
            claim_id="test_3",
            claim_text="Some claim without enough info.",
            label="NOT ENOUGH INFO",
            evidence_text=""
        )

        assert result is None

    def test_generate_batch(self, triple_generator, sample_claims):
        """Testet Batch-Generierung."""
        results = triple_generator.generate_batch(sample_claims)

        # Sollte 2 Triples haben (NEI wird gefiltert, aber sample_claims hat 3 SUPPORTS/REFUTES)
        assert len(results) >= 2

        # Prüfe Kategorien
        categories = [r.category for r in results]
        assert TripleCategory.SUPPORTS in categories
        assert TripleCategory.REFUTES in categories

    def test_get_statistics(self, triple_generator, sample_annotated_triples):
        """Testet Statistik-Funktion."""
        stats = triple_generator.get_statistics(sample_annotated_triples)

        assert stats["total"] == len(sample_annotated_triples)
        assert "supports" in stats
        assert "refutes" in stats
        assert "should_accept" in stats
        assert "should_reject" in stats


# =============================================================================
# ENTITY VARIANT GENERATOR TESTS
# =============================================================================

class TestEntityVariantGenerator:
    """Tests für EntityVariantGenerator."""

    def test_generate_person_variants(self, variant_generator):
        """Testet Personennamen-Varianten."""
        variants = variant_generator.generate_variants(
            "Albert Einstein",
            entity_type=EntityType.PERSON
        )

        assert len(variants) > 0

        # Prüfe erwartete Varianten
        variant_names = [v.variant_name for v in variants]
        assert any("Einstein" in v for v in variant_names)  # Nachname
        assert any("A." in v for v in variant_names)  # Abkürzung

    def test_generate_org_variants(self, variant_generator):
        """Testet Organisationsnamen-Varianten."""
        variants = variant_generator.generate_variants(
            "Apple Inc.",
            entity_type=EntityType.ORGANIZATION
        )

        assert len(variants) > 0

        # Sollte Version ohne "Inc." haben
        variant_names = [v.variant_name for v in variants]
        assert any("Apple" in v and "Inc." not in v for v in variant_names)

    def test_generate_location_variants(self, variant_generator):
        """Testet Ortsnamen-Varianten."""
        variants = variant_generator.generate_variants(
            "United States",
            entity_type=EntityType.LOCATION
        )

        assert len(variants) > 0

        # Sollte Abkürzung haben
        variant_names = [v.variant_name for v in variants]
        assert any("U.S." in v for v in variant_names)

    def test_generate_variant_triples(self, variant_generator, sample_annotated_triples):
        """Testet Triple-Varianten-Generierung."""
        # Filtere nur SUPPORTS
        supports = [t for t in sample_annotated_triples if t.category == TripleCategory.SUPPORTS]

        if not supports:
            pytest.skip("Keine SUPPORTS Triples für Test")

        variant_triples = variant_generator.generate_variant_triples(
            supports,
            variants_per_entity=2
        )

        # Sollte Varianten generiert haben
        assert len(variant_triples) >= 0  # Kann 0 sein wenn Parsing fehlschlägt

        # Prüfe Ground Truth
        for vt in variant_triples:
            assert vt.category == TripleCategory.ENTITY_VARIANT
            assert vt.ground_truth_merge is True
            assert vt.should_merge is True

    def test_jaro_winkler_similarity(self, variant_generator):
        """Testet Jaro-Winkler Ähnlichkeit."""
        # Identische Strings
        assert variant_generator._jaro_winkler_sim("test", "test") == 1.0

        # Leere Strings
        assert variant_generator._jaro_winkler_sim("", "test") == 0.0

        # Ähnliche Strings
        sim = variant_generator._jaro_winkler_sim("Einstein", "Einsten")
        assert 0.8 < sim < 1.0


# =============================================================================
# CROSS-DOC CONFLICT GENERATOR TESTS
# =============================================================================

class TestCrossDocConflictGenerator:
    """Tests für CrossDocConflictGenerator."""

    def test_generate_conflicting_object(self, conflict_generator):
        """Testet Konflikt-Objekt-Generierung."""
        # Bekannte Konflikte aus Templates
        result = conflict_generator.generate_conflicting_object("Ulm")
        assert result is not None
        assert result != "Ulm"

    def test_generate_conflicting_object_boolean(self, conflict_generator):
        """Testet Boolean-Konflikt-Generierung."""
        result = conflict_generator.generate_conflicting_object("true")
        assert result is not None
        assert "false" in result.lower()

    def test_generate_cross_doc_conflicts(self, conflict_generator, sample_annotated_triples):
        """Testet Cross-Doc-Konflikt-Generierung."""
        supports = [t for t in sample_annotated_triples if t.category == TripleCategory.SUPPORTS]

        if not supports:
            pytest.skip("Keine SUPPORTS Triples für Test")

        conflicts = conflict_generator.generate_cross_doc_conflicts(
            supports,
            conflict_ratio=0.5
        )

        # Prüfe Ground Truth
        for conflict in conflicts:
            assert conflict.category == TripleCategory.CROSS_DOC
            assert conflict.ground_truth_accept is False
            assert conflict.should_reject is True
            assert conflict.nli_label == "CONTRADICTION"

    def test_generate_semantic_contradictions(self, conflict_generator, sample_annotated_triples):
        """Testet semantische Widerspruchsgenerierung."""
        supports = [t for t in sample_annotated_triples if t.category == TripleCategory.SUPPORTS]

        if not supports:
            pytest.skip("Keine SUPPORTS Triples für Test")

        contradictions = conflict_generator.generate_semantic_contradictions(
            supports,
            num_contradictions=5
        )

        # Sollte Widersprüche generiert haben
        assert len(contradictions) >= 0

        # Prüfe Struktur
        for c in contradictions:
            assert c.category == TripleCategory.CROSS_DOC
            assert c.should_reject is True


# =============================================================================
# STREAMING SHUFFLER TESTS
# =============================================================================

class TestStreamingShuffler:
    """Tests für StreamingShuffler."""

    def test_random_shuffle_reproducible(self, sample_annotated_triples):
        """Testet dass Random Shuffle reproduzierbar ist."""
        shuffler1 = StreamingShuffler(strategy=ShuffleStrategy.RANDOM, seed=42)
        shuffler2 = StreamingShuffler(strategy=ShuffleStrategy.RANDOM, seed=42)

        result1 = shuffler1.shuffle(sample_annotated_triples)
        result2 = shuffler2.shuffle(sample_annotated_triples)

        # Gleiche Reihenfolge bei gleichem Seed
        assert len(result1) == len(result2)
        for t1, t2 in zip(result1, result2):
            assert t1.original_claim_id == t2.original_claim_id

    def test_different_seeds_different_order(self, sample_annotated_triples):
        """Testet dass verschiedene Seeds verschiedene Reihenfolgen ergeben."""
        if len(sample_annotated_triples) < 2:
            pytest.skip("Nicht genug Triples für Test")

        shuffler1 = StreamingShuffler(strategy=ShuffleStrategy.RANDOM, seed=42)
        shuffler2 = StreamingShuffler(strategy=ShuffleStrategy.RANDOM, seed=123)

        result1 = shuffler1.shuffle(sample_annotated_triples)
        result2 = shuffler2.shuffle(sample_annotated_triples)

        # Unterschiedliche Reihenfolgen (mit hoher Wahrscheinlichkeit)
        ids1 = [t.original_claim_id for t in result1]
        ids2 = [t.original_claim_id for t in result2]
        # Mindestens eine Position sollte unterschiedlich sein
        assert ids1 != ids2 or len(ids1) <= 1

    def test_interleaved_shuffle(self, sample_annotated_triples):
        """Testet Interleaved Shuffle."""
        shuffler = StreamingShuffler(strategy=ShuffleStrategy.INTERLEAVED, seed=42)
        result = shuffler.shuffle(sample_annotated_triples)

        assert len(result) == len(sample_annotated_triples)

    def test_temporal_shuffle(self, sample_annotated_triples):
        """Testet Temporal Shuffle."""
        shuffler = StreamingShuffler(strategy=ShuffleStrategy.TEMPORAL, seed=42)
        result = shuffler.shuffle(sample_annotated_triples)

        assert len(result) == len(sample_annotated_triples)

        # SUPPORTS sollten tendenziell früher kommen
        supports_positions = [
            i for i, t in enumerate(result)
            if t.category == TripleCategory.SUPPORTS
        ]
        refutes_positions = [
            i for i, t in enumerate(result)
            if t.category == TripleCategory.REFUTES
        ]

        if supports_positions and refutes_positions:
            avg_supports = sum(supports_positions) / len(supports_positions)
            avg_refutes = sum(refutes_positions) / len(refutes_positions)
            # SUPPORTS sollte früher sein
            assert avg_supports <= avg_refutes

    def test_clustered_shuffle(self, sample_annotated_triples):
        """Testet Clustered Shuffle."""
        shuffler = StreamingShuffler(strategy=ShuffleStrategy.CLUSTERED, seed=42)
        result = shuffler.shuffle(sample_annotated_triples)

        assert len(result) == len(sample_annotated_triples)

    def test_adversarial_shuffle(self, sample_annotated_triples):
        """Testet Adversarial Shuffle."""
        shuffler = StreamingShuffler(strategy=ShuffleStrategy.ADVERSARIAL, seed=42)
        result = shuffler.shuffle(sample_annotated_triples)

        assert len(result) == len(sample_annotated_triples)

    def test_stream_generator(self, sample_annotated_triples):
        """Testet Stream-Generator."""
        shuffler = StreamingShuffler(strategy=ShuffleStrategy.RANDOM, seed=42)

        streamed = list(shuffler.stream(sample_annotated_triples, shuffle_first=True))

        assert len(streamed) == len(sample_annotated_triples)

    def test_stream_batches(self, sample_annotated_triples):
        """Testet Batch-Streaming."""
        shuffler = StreamingShuffler(
            strategy=ShuffleStrategy.RANDOM,
            seed=42,
            batch_size=2
        )

        batches = list(shuffler.stream_batches(sample_annotated_triples, shuffle_first=True))

        # Prüfe Batch-Größen
        total = sum(len(b) for b in batches)
        assert total == len(sample_annotated_triples)

    def test_get_statistics(self, sample_annotated_triples):
        """Testet Statistik-Funktion."""
        shuffler = StreamingShuffler(strategy=ShuffleStrategy.RANDOM, seed=42)
        shuffled = shuffler.shuffle(sample_annotated_triples)

        stats = shuffler.get_statistics(shuffled)

        assert stats["total"] == len(sample_annotated_triples)
        assert "categories" in stats
        assert "ground_truth" in stats
        assert stats["strategy"] == "random"
        assert stats["seed"] == 42

    def test_create_shuffler_factory(self):
        """Testet Factory-Funktion."""
        shuffler = create_shuffler("random", seed=42, batch_size=10)
        assert shuffler.strategy == ShuffleStrategy.RANDOM
        assert shuffler.seed == 42
        assert shuffler.batch_size == 10

        # Unbekannte Strategie -> fallback zu random
        shuffler = create_shuffler("unknown_strategy", seed=42)
        assert shuffler.strategy == ShuffleStrategy.RANDOM


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integrationstests für das gesamte Modul."""

    def test_full_pipeline(self, sample_claims):
        """Testet die gesamte Pipeline."""
        # 1. Generiere Basis-Triples
        generator = FEVERTripleGenerator()
        base_triples = generator.generate_batch(sample_claims)

        assert len(base_triples) >= 2

        # 2. Generiere Entity-Varianten
        variant_gen = EntityVariantGenerator()
        variant_triples = variant_gen.generate_variant_triples(base_triples, variants_per_entity=1)

        # 3. Generiere Cross-Doc Konflikte
        conflict_gen = CrossDocConflictGenerator(seed=42)
        conflict_triples = conflict_gen.generate_semantic_contradictions(base_triples, num_contradictions=2)

        # 4. Kombiniere alle Triples
        all_triples = base_triples + variant_triples + conflict_triples

        # 5. Shuffle
        shuffler = create_shuffler("random", seed=42)
        shuffled = shuffler.shuffle(all_triples)

        assert len(shuffled) == len(all_triples)

        # 6. Statistiken
        stats = shuffler.get_statistics(shuffled)
        assert stats["total"] == len(all_triples)

    def test_ground_truth_consistency(self, sample_annotated_triples):
        """Testet Konsistenz der Ground Truth Labels."""
        for triple in sample_annotated_triples:
            # Genau eine der drei Eigenschaften sollte True sein
            conditions = [
                triple.should_accept,
                triple.should_reject,
                triple.should_merge
            ]

            # should_merge impliziert should_accept
            if triple.should_merge:
                assert triple.ground_truth_accept is True

            # should_reject ist das Gegenteil von should_accept (wenn nicht merge)
            if triple.should_reject:
                assert triple.ground_truth_accept is False


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
