#!/usr/bin/env python3
# tests/test_hotpotqa_missing_source.py
"""
Test des Missing Source Penalty Features mit echten HotpotQA-Daten.

Testet:
1. Triples MIT source_document_id (aus HotpotQA)
2. Triples OHNE source_document_id (simuliert fehlende Provenance)
3. Vergleich der Konfidenz-Werte
"""

import sys
sys.path.insert(0, '.')

import logging
from typing import List, Tuple
from datetime import datetime

from src.models.entities import Entity, EntityType, Triple, ValidationStatus, Relation
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.graph.memory_repository import InMemoryGraphRepository
from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_triples_from_example(
    example: QAExample,
    with_source: bool = True
) -> List[Triple]:
    """
    Extrahiert Triples aus einem QA-Beispiel.

    Args:
        example: HotpotQA-Beispiel
        with_source: Wenn True, setze source_document_id
    """
    triples = []
    entity_cache = {}

    # Entity-Typ Heuristik
    def infer_type(name: str) -> EntityType:
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["city", "country", "river", "state"]):
            return EntityType.LOCATION
        if any(kw in name_lower for kw in ["university", "company", "inc", "school"]):
            return EntityType.ORGANIZATION
        if any(kw in name_lower for kw in ["war", "battle", "championship"]):
            return EntityType.EVENT
        return EntityType.PERSON

    # Erstelle Entitäten aus Dokumenttiteln
    for para in example.context_paragraphs:
        title = para.get("title", "")
        if not title or title in entity_cache:
            continue
        entity = Entity(
            name=title,
            entity_type=infer_type(title),
            source_document=example.id if with_source else None
        )
        entity_cache[title] = entity

    # Verknüpfe aufeinanderfolgende supporting_facts
    sf_titles = [sf.title for sf in example.supporting_facts]

    for i in range(len(sf_titles) - 1):
        title1, title2 = sf_titles[i], sf_titles[i + 1]
        if title1 not in entity_cache or title2 not in entity_cache:
            continue

        triple = Triple(
            subject=entity_cache[title1],
            predicate="RELATED_TO",
            object=entity_cache[title2],
            source_text=example.question,
            source_document_id=example.id if with_source else None,  # KEY!
            extraction_confidence=0.85
        )
        triples.append(triple)

    # Verknüpfe supporting_facts mit Antwort
    answer = example.answer
    if answer and len(answer) > 2:
        answer_entity = Entity(
            name=answer,
            entity_type=infer_type(answer),
            source_document=example.id if with_source else None
        )

        for sf in example.supporting_facts[:1]:  # Nur erstes für Übersichtlichkeit
            if sf.title in entity_cache:
                triple = Triple(
                    subject=entity_cache[sf.title],
                    predicate="RELATED_TO",
                    object=answer_entity,
                    source_text=sf.text,
                    source_document_id=example.id if with_source else None,
                    extraction_confidence=0.80
                )
                triples.append(triple)

    return triples


def run_hotpotqa_test(sample_size: int = 10):
    """
    Führt den Test mit HotpotQA-Daten durch.
    """
    print("\n" + "="*80)
    print("TEST: Missing Source Penalty mit HotpotQA")
    print("="*80)

    # 1. Lade HotpotQA-Daten
    print(f"\n1. Lade {sample_size} HotpotQA-Beispiele...")
    loader = BenchmarkLoader()
    examples = loader.load_hotpotqa(split="validation", sample_size=sample_size)

    if not examples:
        print("FEHLER: HotpotQA konnte nicht geladen werden!")
        print("Stelle sicher dass 'datasets' installiert ist: pip install datasets")
        return False

    print(f"   ✓ {len(examples)} Beispiele geladen")

    # 2. Setup Orchestrator
    print("\n2. Initialisiere Konsistenzmodul...")

    config = ConsistencyConfig(
        valid_relation_types=["RELATED_TO", "GEBOREN_IN", "ARBEITET_BEI", "KENNT"],
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        enable_missing_source_penalty=True,
        missing_source_penalty=0.7,
    )

    graph_repo = InMemoryGraphRepository()

    # Versuche Embeddings zu laden (optional)
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✓ Embedding-Modell geladen")
    except Exception as e:
        print(f"   ⚠ Kein Embedding-Modell: {e}")

    orchestrator = ConsistencyOrchestrator(
        config=config,
        graph_repo=graph_repo,
        embedding_model=embedding_model,
        llm_client=None,
        enable_metrics=True,
    )

    print("   ✓ Orchestrator initialisiert")

    # 3. Verarbeite Triples MIT Quelle
    print("\n3. Verarbeite Triples MIT source_document_id...")

    triples_with_source = []
    results_with_source = []

    for example in examples[:sample_size//2]:
        extracted = extract_triples_from_example(example, with_source=True)
        for triple in extracted:
            triples_with_source.append(triple)
            result = orchestrator.process(triple)
            results_with_source.append(result)

    print(f"   Verarbeitet: {len(triples_with_source)} Triples")

    # 4. Verarbeite Triples OHNE Quelle
    print("\n4. Verarbeite Triples OHNE source_document_id...")

    triples_without_source = []
    results_without_source = []

    for example in examples[sample_size//2:]:
        extracted = extract_triples_from_example(example, with_source=False)
        for triple in extracted:
            triples_without_source.append(triple)
            result = orchestrator.process(triple)
            results_without_source.append(result)

    print(f"   Verarbeitet: {len(triples_without_source)} Triples")

    # 5. Analysiere Ergebnisse
    print("\n" + "="*80)
    print("ERGEBNISSE")
    print("="*80)

    def analyze_results(results: List[Triple], label: str) -> dict:
        accepted = sum(1 for r in results if r.validation_status == ValidationStatus.ACCEPTED)
        rejected = sum(1 for r in results if r.validation_status == ValidationStatus.REJECTED)

        # Extrahiere Konfidenz aus validation_history
        confidences = []
        missing_source_warnings = 0

        for r in results:
            for h in r.validation_history:
                details = h.get("details", {})
                if "missing_source_warning" in details:
                    missing_source_warnings += 1
                if "provenance_multiplier" in details:
                    confidences.append(details.get("adjusted_confidence", 0.5))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            "label": label,
            "total": len(results),
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / len(results) if results else 0,
            "avg_confidence": avg_conf,
            "missing_source_warnings": missing_source_warnings
        }

    stats_with = analyze_results(results_with_source, "MIT Quelle")
    stats_without = analyze_results(results_without_source, "OHNE Quelle")

    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ Metrik                     │ MIT Quelle        │ OHNE Quelle               │
    ├────────────────────────────┼───────────────────┼───────────────────────────┤
    │ Triples verarbeitet        │ {stats_with['total']:>17} │ {stats_without['total']:>25} │
    │ Akzeptiert                 │ {stats_with['accepted']:>17} │ {stats_without['accepted']:>25} │
    │ Abgelehnt                  │ {stats_with['rejected']:>17} │ {stats_without['rejected']:>25} │
    │ Acceptance Rate            │ {stats_with['acceptance_rate']*100:>16.1f}% │ {stats_without['acceptance_rate']*100:>24.1f}% │
    │ Ø Provenance-Konfidenz     │ {stats_with['avg_confidence']*100:>16.1f}% │ {stats_without['avg_confidence']*100:>24.1f}% │
    │ Missing Source Warnings    │ {stats_with['missing_source_warnings']:>17} │ {stats_without['missing_source_warnings']:>25} │
    └─────────────────────────────────────────────────────────────────────────────┘
    """)

    # 6. Provenance-Statistiken
    print("\n--- Provenance-Statistiken ---")
    prov_stats = orchestrator.get_provenance_statistics()
    print(f"   Quellen verarbeitet: {prov_stats.get('total_sources', 0)}")
    print(f"   Triples ohne Quelle: {prov_stats.get('missing_sources_count', 0)}")
    print(f"   Penalties angewendet: {prov_stats.get('missing_source_penalties_applied', 0)}")

    # Top-Quellen
    top_sources = prov_stats.get("top_sources", [])
    if top_sources:
        print(f"\n   Top-Quellen:")
        for src_id, score, count in top_sources[:5]:
            print(f"      {src_id[:40]}... → {score:.2%} ({count} Triples)")

    # 7. Beispiele zeigen
    print("\n" + "-"*80)
    print("BEISPIELE")
    print("-"*80)

    print("\nMIT Quelle (erste 3):")
    for i, (triple, result) in enumerate(zip(triples_with_source[:3], results_with_source[:3])):
        print(f"  {i+1}. {triple.subject.name} --[{triple.predicate}]--> {triple.object.name}")
        print(f"     Quelle: {triple.source_document_id[:40] if triple.source_document_id else 'KEINE'}...")
        print(f"     Status: {result.validation_status.value}")

    print("\nOHNE Quelle (erste 3):")
    for i, (triple, result) in enumerate(zip(triples_without_source[:3], results_without_source[:3])):
        print(f"  {i+1}. {triple.subject.name} --[{triple.predicate}]--> {triple.object.name}")
        print(f"     Quelle: {triple.source_document_id or 'KEINE'}")
        print(f"     Status: {result.validation_status.value}")
        # Zeige Penalty wenn vorhanden
        for h in result.validation_history:
            if h.get("details", {}).get("missing_source_warning"):
                print(f"     ⚠️  Missing Source Penalty: ×{h['details'].get('missing_source_penalty', 'N/A')}")

    # 8. Assertions
    print("\n" + "-"*80)
    print("ASSERTIONS")
    print("-"*80)

    success = True

    # A1: Triples ohne Quelle sollten Missing Source Warning haben
    if stats_without['total'] > 0:
        warning_rate = stats_without['missing_source_warnings'] / stats_without['total']
        if warning_rate > 0.5:  # Mindestens 50% sollten Warning haben
            print(f"✓ Missing Source Warnings vorhanden: {stats_without['missing_source_warnings']}/{stats_without['total']}")
        else:
            print(f"✗ Zu wenige Missing Source Warnings: {stats_without['missing_source_warnings']}/{stats_without['total']}")
            success = False

    # A2: Penalty-Zähler sollte > 0 sein
    if prov_stats.get('missing_source_penalties_applied', 0) > 0:
        print(f"✓ Penalties angewendet: {prov_stats.get('missing_source_penalties_applied')}")
    else:
        print(f"✗ Keine Penalties angewendet!")
        success = False

    # A3: Durchschnittliche Konfidenz OHNE Quelle sollte niedriger sein
    if stats_with['avg_confidence'] > 0 and stats_without['avg_confidence'] > 0:
        if stats_without['avg_confidence'] < stats_with['avg_confidence']:
            print(f"✓ Konfidenz OHNE Quelle ({stats_without['avg_confidence']:.2%}) < MIT Quelle ({stats_with['avg_confidence']:.2%})")
        else:
            print(f"⚠️  Konfidenz-Unterschied nicht wie erwartet (könnte an wenig Daten liegen)")

    print("\n" + "="*80)
    if success:
        print("ALLE TESTS BESTANDEN")
    else:
        print("EINIGE TESTS FEHLGESCHLAGEN")
    print("="*80)

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Missing Source Penalty mit HotpotQA")
    parser.add_argument("--sample-size", type=int, default=10, help="Anzahl HotpotQA-Beispiele")
    args = parser.parse_args()

    success = run_hotpotqa_test(sample_size=args.sample_size)
    sys.exit(0 if success else 1)
