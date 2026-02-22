# tests/test_missing_source_penalty.py
"""
Test für #12: Missing Source Penalty

Umfassender Test aller Szenarien:
1. Richtiger Fakt MIT Quelle
2. Richtiger Fakt OHNE Quelle
3. Falscher Fakt OHNE Quelle
4. Falscher Fakt MIT Quelle

Testet die komplette Pipeline inkl. Graph-Integration.
"""

import sys
sys.path.insert(0, '.')

from src.models.entities import Entity, EntityType, Triple, ValidationStatus, Relation
from src.consistency.provenance import ProvenanceTracker, ProvenanceConfig, ProvenanceExplainer
from src.consistency.base import ConsistencyConfig
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.graph.memory_repository import InMemoryGraphRepository


def test_all_scenarios_full_pipeline():
    """
    Testet alle 4 Szenarien durch die komplette Pipeline.

    HotpotQA-Kontext:
    - Frage: "What government position was held by the woman who portrayed Yelena Belova?"
    - Korrekte Fakten aus Supporting Facts
    - Falsche Fakten (erfunden/widersprechend)
    """

    print("\n" + "="*80)
    print("TEST #12: Missing Source Penalty - ALLE SZENARIEN (Full Pipeline)")
    print("="*80)

    # =========================================================================
    # SETUP: Konfiguration und Graph
    # =========================================================================
    config = ConsistencyConfig(
        valid_entity_types=["Person", "Organisation", "Ort", "Ereignis", "Konzept"],
        valid_relation_types=[
            "WOHNT_IN", "ARBEITET_BEI", "KENNT", "GEBOREN_IN", "LEITET",
            "SCHRIEB", "SPIELT", "HAT_POSITION", "VERHEIRATET_MIT", "KIND_VON"
        ],
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        # Missing Source Penalty aktiviert
        enable_missing_source_penalty=True,
        missing_source_penalty=0.7,
        # Domain Constraints für Validierung
        domain_constraints={
            "GEBOREN_IN": {"subject_types": ["Person"], "object_types": ["Ort"]},
            "VERHEIRATET_MIT": {"subject_types": ["Person"], "object_types": ["Person"]},
            "KIND_VON": {"subject_types": ["Person"], "object_types": ["Person"]},
            "HAT_POSITION": {"subject_types": ["Person"], "object_types": ["Konzept", "Organisation"]},
        },
        # Kardinalitätsregeln
        cardinality_rules={
            "GEBOREN_IN": {"max": 1},
            "KIND_VON": {"max": 2},
        },
    )

    # In-Memory Graph für Tests
    graph_repo = InMemoryGraphRepository()

    # Orchestrator
    orchestrator = ConsistencyOrchestrator(
        config=config,
        graph_repo=graph_repo,
        embedding_model=None,
        llm_client=None,
        enable_metrics=True,
        always_check_duplicates=True,
    )

    # =========================================================================
    # BASELINE: Existierende Fakten im Graph (Ground Truth)
    # =========================================================================
    print("\n" + "-"*80)
    print("SETUP: Existierende Fakten im Graph (Ground Truth)")
    print("-"*80)

    # Florence Pugh - korrekte Daten
    florence = Entity(name="Florence Pugh", entity_type=EntityType.PERSON)
    oxford = Entity(name="Oxford", entity_type=EntityType.LOCATION)

    existing_triple = Triple(
        subject=florence,
        predicate="GEBOREN_IN",
        object=oxford,
        source_document_id="wikipedia_florence_pugh",
        extraction_confidence=0.95
    )

    # Füge zum Graph hinzu (als Ground Truth)
    graph_repo.create_entity(florence)
    graph_repo.create_entity(oxford)
    print(f"  ✓ Ground Truth: {existing_triple}")

    # =========================================================================
    # SZENARIO 1: Richtiger Fakt MIT Quelle
    # =========================================================================
    print("\n" + "-"*80)
    print("SZENARIO 1: RICHTIGER FAKT MIT QUELLE")
    print("-"*80)

    triple_1 = Triple(
        subject=Entity(name="Florence Pugh", entity_type=EntityType.PERSON),
        predicate="SPIELT",
        object=Entity(name="Yelena Belova", entity_type=EntityType.PERSON),
        source_document_id="hotpotqa_wiki_black_widow",
        source_text="Florence Pugh portrayed Yelena Belova in Black Widow (2021).",
        extraction_confidence=0.90
    )

    result_1 = orchestrator.process(triple_1)

    print(f"\n  Triple: {triple_1}")
    print(f"  Quelle: {triple_1.source_document_id}")
    print(f"  → Status: {result_1.validation_status.value}")
    print(f"  → Historie: {[h['stage'] for h in result_1.validation_history]}")

    # Provenance Details
    prov_details_1 = None
    for h in result_1.validation_history:
        if "source_score" in h.get("details", {}):
            prov_details_1 = h["details"]
            break

    if prov_details_1:
        print(f"  → Source Score: {prov_details_1.get('source_score', 'N/A')}")
        print(f"  → Missing Source Penalty: {prov_details_1.get('missing_source_penalty', 'KEINE')}")

    # =========================================================================
    # SZENARIO 2: Richtiger Fakt OHNE Quelle
    # =========================================================================
    print("\n" + "-"*80)
    print("SZENARIO 2: RICHTIGER FAKT OHNE QUELLE")
    print("-"*80)

    triple_2 = Triple(
        subject=Entity(name="Florence Pugh", entity_type=EntityType.PERSON),
        predicate="GEBOREN_IN",
        object=Entity(name="Oxford", entity_type=EntityType.LOCATION),
        source_document_id=None,  # KEINE QUELLE!
        source_text=None,
        extraction_confidence=0.90
    )

    result_2 = orchestrator.process(triple_2)

    print(f"\n  Triple: {triple_2}")
    print(f"  Quelle: {triple_2.source_document_id}")
    print(f"  → Status: {result_2.validation_status.value}")
    print(f"  → Historie: {[h['stage'] for h in result_2.validation_history]}")

    prov_details_2 = None
    for h in result_2.validation_history:
        if "missing_source_warning" in h.get("details", {}):
            prov_details_2 = h["details"]
            break

    if prov_details_2:
        print(f"  → Source Score: {prov_details_2.get('source_score', 'N/A')}")
        print(f"  → Missing Source Penalty: {prov_details_2.get('missing_source_penalty', 'KEINE')}")
        print(f"  → Missing Source Warning: {prov_details_2.get('missing_source_warning', False)}")

    # =========================================================================
    # SZENARIO 3: Falscher Fakt OHNE Quelle (Schema-Verletzung)
    # =========================================================================
    print("\n" + "-"*80)
    print("SZENARIO 3: FALSCHER FAKT OHNE QUELLE (Schema-Verletzung)")
    print("-"*80)

    triple_3 = Triple(
        subject=Entity(name="Oxford", entity_type=EntityType.LOCATION),  # FALSCH: Ort als Subject
        predicate="GEBOREN_IN",
        object=Entity(name="Florence Pugh", entity_type=EntityType.PERSON),  # FALSCH: Person als Object
        source_document_id=None,  # KEINE QUELLE!
        source_text=None,
        extraction_confidence=0.85
    )

    result_3 = orchestrator.process(triple_3)

    print(f"\n  Triple: {triple_3}")
    print(f"  Quelle: {triple_3.source_document_id}")
    print(f"  Problem: Subject ist Ort, sollte Person sein (Domain Constraint)")
    print(f"  → Status: {result_3.validation_status.value}")
    print(f"  → Historie: {[h['stage'] for h in result_3.validation_history]}")

    # Zeige Konflikte
    if result_3.conflicts:
        print(f"  → Konflikte: {len(result_3.conflicts)}")
        for c in result_3.conflicts:
            print(f"     - {c.conflict_type.value}: {c.description}")

    # =========================================================================
    # SZENARIO 4: Falscher Fakt MIT Quelle (Widerspruch zu Ground Truth)
    # =========================================================================
    print("\n" + "-"*80)
    print("SZENARIO 4: FALSCHER FAKT MIT QUELLE (Kardinalitäts-Konflikt)")
    print("-"*80)

    # Erst: Füge korrekten Geburtsort ein UND persistiere die Relation im Graph
    # (Die Kardinalitätsprüfung benötigt existierende Relationen im Graph)
    florence_entity = graph_repo.find_by_name("Florence Pugh")
    if not florence_entity:
        florence_entity = [Entity(name="Florence Pugh", entity_type=EntityType.PERSON)]
        graph_repo.create_entity(florence_entity[0])
    oxford_entity = graph_repo.find_by_name("Oxford")
    if not oxford_entity:
        oxford_entity = [Entity(name="Oxford", entity_type=EntityType.LOCATION)]
        graph_repo.create_entity(oxford_entity[0])

    # Erstelle die Relation im Graph (simuliert dass vorheriger Fakt akzeptiert wurde)
    existing_relation = Relation(
        source_id=florence_entity[0].id,
        target_id=oxford_entity[0].id,
        relation_type="GEBOREN_IN",
        source_document_id="wikipedia_florence_pugh",
        confidence=0.95
    )
    graph_repo.create_relation(existing_relation)
    print(f"  Setup: Florence Pugh --[GEBOREN_IN]--> Oxford bereits im Graph persistiert")

    # Jetzt: Widersprüchlicher Fakt (anderer Geburtsort)
    # Wichtig: Verwende die gleiche Entity-ID für Florence Pugh,
    # damit die Kardinalitätsprüfung die existierende Relation findet
    london = Entity(name="London", entity_type=EntityType.LOCATION)
    graph_repo.create_entity(london)

    triple_4 = Triple(
        subject=florence_entity[0],  # Gleiche Entity wie im Graph!
        predicate="GEBOREN_IN",
        object=london,  # Anderer Geburtsort - KONFLIKT!
        source_document_id="unreliable_blog_xyz",  # Unzuverlässige Quelle
        source_text="Florence Pugh was born in London.",
        extraction_confidence=0.80
    )

    result_4 = orchestrator.process(triple_4)

    print(f"\n  Triple: {triple_4}")
    print(f"  Quelle: {triple_4.source_document_id}")
    print(f"  Problem: GEBOREN_IN hat max=1, aber Oxford existiert bereits")
    print(f"  → Status: {result_4.validation_status.value}")
    print(f"  → Historie: {[h['stage'] for h in result_4.validation_history]}")

    if result_4.conflicts:
        print(f"  → Konflikte: {len(result_4.conflicts)}")
        for c in result_4.conflicts:
            print(f"     - {c.conflict_type.value}: {c.description}")

    # =========================================================================
    # SZENARIO 5: Ungültiger Relationstyp OHNE Quelle
    # =========================================================================
    print("\n" + "-"*80)
    print("SZENARIO 5: UNGÜLTIGER RELATIONSTYP OHNE QUELLE")
    print("-"*80)

    triple_5 = Triple(
        subject=Entity(name="Florence Pugh", entity_type=EntityType.PERSON),
        predicate="FLIEGT_NACH",  # Nicht in valid_relation_types!
        object=Entity(name="Los Angeles", entity_type=EntityType.LOCATION),
        source_document_id=None,
        extraction_confidence=0.75
    )

    result_5 = orchestrator.process(triple_5)

    print(f"\n  Triple: {triple_5}")
    print(f"  Quelle: {triple_5.source_document_id}")
    print(f"  Problem: FLIEGT_NACH ist kein gültiger Relationstyp")
    print(f"  → Status: {result_5.validation_status.value}")
    print(f"  → Historie: {[h['stage'] for h in result_5.validation_history]}")

    # =========================================================================
    # ZUSAMMENFASSUNG
    # =========================================================================
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG")
    print("="*80)

    scenarios = [
        ("1. Richtiger Fakt MIT Quelle", result_1, "ACCEPTED"),
        ("2. Richtiger Fakt OHNE Quelle", result_2, "ACCEPTED (mit Penalty)"),
        ("3. Falscher Fakt OHNE Quelle", result_3, "REJECTED"),
        ("4. Falscher Fakt MIT Quelle", result_4, "REJECTED/CONFLICT"),
        ("5. Ungültiger Typ OHNE Quelle", result_5, "REJECTED"),
    ]

    print(f"""
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ Szenario                          │ Status        │ Erwartet               │
    ├───────────────────────────────────┼───────────────┼────────────────────────┤""")

    all_correct = True
    for name, result, expected in scenarios:
        status = result.validation_status.value.upper()
        match = "✓" if expected.split()[0].upper() in status.upper() else "✗"
        if match == "✗":
            all_correct = False
        print(f"    │ {name:<33} │ {status:<13} │ {expected:<22} │")

    print(f"    └────────────────────────────────────────────────────────────────────────────┘")

    # Provenance Statistiken
    print("\n--- Provenance Statistiken ---")
    prov_stats = orchestrator.get_provenance_statistics()
    print(f"  Quellen verarbeitet: {prov_stats.get('total_sources', 0)}")
    print(f"  Triples ohne Quelle: {prov_stats.get('missing_sources_count', 0)}")
    print(f"  Penalties angewendet: {prov_stats.get('missing_source_penalties_applied', 0)}")

    # Orchestrator Statistiken
    print("\n--- Pipeline Statistiken ---")
    stats = orchestrator.get_statistics()
    print(f"  Total verarbeitet: {stats['total_processed']}")
    print(f"  Akzeptiert: {stats['accepted']}")
    print(f"  Abgelehnt: {stats['rejected']}")
    print(f"  Needs Review: {stats['needs_review']}")

    # =========================================================================
    # ASSERTIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("ASSERTIONS")
    print("-"*80)

    # 1. Richtiger Fakt mit Quelle → ACCEPTED
    assert result_1.validation_status == ValidationStatus.ACCEPTED, \
        f"Szenario 1 sollte ACCEPTED sein: {result_1.validation_status}"
    print("✓ Szenario 1: Richtiger Fakt mit Quelle → ACCEPTED")

    # 2. Richtiger Fakt ohne Quelle → ACCEPTED (aber mit niedrigerer Konfidenz)
    assert result_2.validation_status == ValidationStatus.ACCEPTED, \
        f"Szenario 2 sollte ACCEPTED sein: {result_2.validation_status}"
    # Prüfe dass Missing Source Warning gesetzt wurde
    has_penalty = any(
        h.get("details", {}).get("missing_source_warning", False)
        for h in result_2.validation_history
    )
    assert has_penalty, "Szenario 2 sollte Missing Source Warning haben"
    print("✓ Szenario 2: Richtiger Fakt ohne Quelle → ACCEPTED (mit Penalty)")

    # 3. Falscher Fakt ohne Quelle (Schema-Verletzung) → REJECTED
    assert result_3.validation_status == ValidationStatus.REJECTED, \
        f"Szenario 3 sollte REJECTED sein: {result_3.validation_status}"
    print("✓ Szenario 3: Falscher Fakt (Schema-Verletzung) → REJECTED")

    # 4. Falscher Fakt mit Quelle (Kardinalitäts-Konflikt) → REJECTED oder CONFLICT
    assert result_4.validation_status in [ValidationStatus.REJECTED, ValidationStatus.CONFLICTING, ValidationStatus.NEEDS_REVIEW], \
        f"Szenario 4 sollte REJECTED/CONFLICTING sein: {result_4.validation_status}"
    print(f"✓ Szenario 4: Falscher Fakt (Kardinalität) → {result_4.validation_status.value}")

    # 5. Ungültiger Relationstyp → REJECTED
    assert result_5.validation_status == ValidationStatus.REJECTED, \
        f"Szenario 5 sollte REJECTED sein: {result_5.validation_status}"
    print("✓ Szenario 5: Ungültiger Relationstyp → REJECTED")

    # 6. Missing Source Penalty wurde korrekt gezählt
    # Hinweis: Nur Triples die Stage 2 erreichen werden gezählt
    # (Szenario 3 und 5 scheitern in Stage 1)
    assert prov_stats.get('missing_sources_count', 0) >= 1, \
        f"Mindestens 1 Triple ohne Quelle erwartet: {prov_stats.get('missing_sources_count')}"
    print(f"✓ Missing Source Count korrekt: {prov_stats.get('missing_sources_count')} "
          f"(nur Triples die Stage 2 erreichen)")

    print("\n" + "="*80)
    print("ALLE ASSERTIONS BESTANDEN")
    print("="*80)

    return True


def test_provenance_confidence_comparison():
    """
    Direkter Vergleich der Konfidenz-Werte mit und ohne Quelle.
    """

    print("\n" + "="*80)
    print("TEST: Konfidenz-Vergleich (Provenance-Level)")
    print("="*80)

    config = ProvenanceConfig(
        enable_missing_source_penalty=True,
        missing_source_penalty=0.7,
        enable_corroboration=False,
    )

    tracker = ProvenanceTracker(config=config)
    explainer = ProvenanceExplainer(tracker)

    # Gleiches Triple, unterschiedliche Quellen
    base_confidence = 0.85

    # MIT Wikipedia-Quelle
    triple_wiki = Triple(
        subject=Entity(name="Test", entity_type=EntityType.PERSON),
        predicate="KENNT",
        object=Entity(name="Test2", entity_type=EntityType.PERSON),
        source_document_id="wikipedia_test",
        extraction_confidence=base_confidence
    )

    # MIT unbekannter Quelle
    triple_unknown = Triple(
        subject=Entity(name="Test", entity_type=EntityType.PERSON),
        predicate="KENNT",
        object=Entity(name="Test2", entity_type=EntityType.PERSON),
        source_document_id="random_source_123",
        extraction_confidence=base_confidence
    )

    # OHNE Quelle
    triple_none = Triple(
        subject=Entity(name="Test", entity_type=EntityType.PERSON),
        predicate="KENNT",
        object=Entity(name="Test2", entity_type=EntityType.PERSON),
        source_document_id=None,
        extraction_confidence=base_confidence
    )

    conf_wiki, det_wiki = tracker.calculate_confidence(triple_wiki, base_confidence)
    conf_unknown, det_unknown = tracker.calculate_confidence(triple_unknown, base_confidence)
    conf_none, det_none = tracker.calculate_confidence(triple_none, base_confidence)

    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Quelle              │ Source Score │ Penalty │ Final Confidence    │
    ├─────────────────────┼──────────────┼─────────┼─────────────────────┤
    │ wikipedia_test      │ {det_wiki.get('source_score', 0)*100:5.1f}%       │ -       │ {conf_wiki*100:5.1f}%              │
    │ random_source_123   │ {det_unknown.get('source_score', 0)*100:5.1f}%       │ -       │ {conf_unknown*100:5.1f}%              │
    │ None (keine)        │ {det_none.get('source_score', 0)*100:5.1f}%       │ ×0.7    │ {conf_none*100:5.1f}%              │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    print("Erklärungen:")
    print(f"\n1. Wikipedia:\n{explainer.explain_confidence(triple_wiki, det_wiki)}")
    print(f"\n2. Unbekannte Quelle:\n{explainer.explain_confidence(triple_unknown, det_unknown)}")
    print(f"\n3. Keine Quelle:\n{explainer.explain_confidence(triple_none, det_none)}")

    # Assertions
    print("\n--- Assertions ---")
    assert conf_wiki > conf_unknown > conf_none, \
        f"Erwartete Reihenfolge: wiki > unknown > none: {conf_wiki} > {conf_unknown} > {conf_none}"
    print(f"✓ Konfidenz-Reihenfolge korrekt: {conf_wiki:.2%} > {conf_unknown:.2%} > {conf_none:.2%}")

    assert det_none.get("missing_source_penalty") == 0.7, \
        "Missing Source Penalty sollte 0.7 sein"
    print("✓ Missing Source Penalty = 0.7")

    assert det_none.get("missing_source_warning") == True, \
        "Missing Source Warning sollte gesetzt sein"
    print("✓ Missing Source Warning gesetzt")

    return True


if __name__ == "__main__":
    test_all_scenarios_full_pipeline()
    print()
    test_provenance_confidence_comparison()
