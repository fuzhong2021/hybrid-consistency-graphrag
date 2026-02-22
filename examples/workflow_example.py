#!/usr/bin/env python3
"""
Beispiel: Konsistenzmodul Workflow

Zeigt den kompletten Ablauf:
1. INITIAL: Knowledge Graph mit Basis-Daten füllen
2. UPDATE: Neue Triples mit Konsistenzprüfung hinzufügen
3. ERGEBNIS: Was wird akzeptiert/abgelehnt und warum

Verwendung:
    python examples/workflow_example.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig


def create_entity(name: str, entity_type: EntityType) -> Entity:
    """Hilfsfunktion: Entity erstellen."""
    return Entity(name=name, entity_type=entity_type)


def create_triple(subject: Entity, predicate: str, obj: Entity, source: str = "example") -> Triple:
    """Hilfsfunktion: Triple erstellen."""
    return Triple(
        subject=subject,
        predicate=predicate,
        object=obj,
        source_text=f"{subject.name} {predicate} {obj.name}",
        source_document_id=source,
        extraction_confidence=0.85,
    )


def main():
    print("=" * 70)
    print("KONSISTENZMODUL - WORKFLOW BEISPIEL")
    print("=" * 70)

    # =========================================================================
    # SCHRITT 1: SETUP
    # =========================================================================
    print("\n[1] SETUP: Graph und Konsistenzmodul initialisieren")
    print("-" * 50)

    # Leerer Graph
    graph = InMemoryGraphRepository()

    # Konsistenz-Konfiguration
    config = ConsistencyConfig(
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        similarity_threshold=0.85,
        valid_entity_types=["Person", "Organisation", "Ort", "Ereignis", "Konzept"],
        valid_relation_types=[
            "GEBOREN_IN", "ARBEITET_BEI", "WOHNT_IN", "GRUENDETE",
            "VERHEIRATET_MIT", "KENNT", "RELATED_TO", "MITGLIED_VON",
        ],
        cardinality_rules={
            "GEBOREN_IN": {"max": 1},  # Man kann nur an EINEM Ort geboren sein
        },
    )

    print(f"  Graph initialisiert: {len(graph._entities)} Entitäten, {len(graph._relations)} Relationen")
    print(f"  Kardinalitätsregeln: GEBOREN_IN max=1")

    # =========================================================================
    # SCHRITT 2: INITIAL SEED (ohne Konsistenzprüfung)
    # =========================================================================
    print("\n[2] INITIAL SEED: Basis-Daten laden (ohne Prüfung)")
    print("-" * 50)

    # Basis-Entitäten
    einstein = create_entity("Albert Einstein", EntityType.PERSON)
    ulm = create_entity("Ulm", EntityType.LOCATION)
    eth = create_entity("ETH Zürich", EntityType.ORGANIZATION)
    marie = create_entity("Marie Curie", EntityType.PERSON)

    # Basis-Triples (direkt speichern, keine Prüfung)
    seed_triples = [
        create_triple(einstein, "GEBOREN_IN", ulm, "wiki_einstein"),
        create_triple(einstein, "ARBEITET_BEI", eth, "wiki_einstein"),
        create_triple(marie, "KENNT", einstein, "wiki_curie"),
    ]

    for triple in seed_triples:
        triple.validation_status = ValidationStatus.ACCEPTED
        graph.save_triple(triple)
        print(f"  ✓ Gespeichert: {triple.subject.name} --[{triple.predicate}]--> {triple.object.name}")

    print(f"\n  Graph nach Seed: {len(graph._entities)} Entitäten, {len(graph._relations)} Relationen")

    # =========================================================================
    # SCHRITT 3: KONSISTENZMODUL AKTIVIEREN
    # =========================================================================
    print("\n[3] KONSISTENZMODUL AKTIVIEREN")
    print("-" * 50)

    orchestrator = ConsistencyOrchestrator(
        config=config,
        graph_repo=graph,
        embedding_model=None,  # Ohne Embedding für dieses Beispiel
        llm_client=None,       # Ohne LLM für dieses Beispiel
        enable_metrics=True,
        always_check_duplicates=True,
    )

    print("  Orchestrator initialisiert mit 3-Stufen-Kaskade:")
    print("    Stufe 1: Regelbasiert (Schema, Kardinalität, Self-Loops)")
    print("    Stufe 2: Embedding-basiert (Duplikate) - DEAKTIVIERT")
    print("    Stufe 3: LLM-Arbitration - DEAKTIVIERT")

    # =========================================================================
    # SCHRITT 4: UPDATES MIT KONSISTENZPRÜFUNG
    # =========================================================================
    print("\n[4] UPDATES: Neue Triples mit Konsistenzprüfung")
    print("-" * 50)

    # Verschiedene Update-Szenarien
    updates = [
        # ✓ GÜLTIG: Neue Information
        {
            "triple": create_triple(
                create_entity("Max Planck", EntityType.PERSON),
                "KENNT",
                einstein,
                "update_1"
            ),
            "erwartung": "ACCEPT",
            "grund": "Neue, gültige Information",
        },

        # ✗ SELF-LOOP: Subject = Object
        {
            "triple": create_triple(
                einstein,
                "KENNT",
                einstein,  # Gleiche Entity!
                "update_2"
            ),
            "erwartung": "REJECT",
            "grund": "Self-Loop (Einstein kennt Einstein)",
        },

        # ✗ KARDINALITÄT: Einstein kann nur an EINEM Ort geboren sein
        {
            "triple": create_triple(
                einstein,  # WICHTIG: Gleiche Entity-Instanz wie im Seed!
                "GEBOREN_IN",
                create_entity("München", EntityType.LOCATION),
                "update_3"
            ),
            "erwartung": "REJECT",
            "grund": "Kardinalitätsverletzung (bereits in Ulm geboren)",
        },

        # ✓ GÜLTIG: Andere Relation, gleiche Entitäten
        {
            "triple": create_triple(
                einstein,
                "WOHNT_IN",
                create_entity("Berlin", EntityType.LOCATION),
                "update_4"
            ),
            "erwartung": "ACCEPT",
            "grund": "Gültig (WOHNT_IN hat keine Kardinalitätsregel)",
        },

        # ✗ SCHEMA: Ungültige Relation
        {
            "triple": create_triple(
                einstein,
                "FLIEGT_NACH",  # Nicht in valid_relation_types!
                create_entity("Mond", EntityType.LOCATION),
                "update_5"
            ),
            "erwartung": "REJECT",
            "grund": "Ungültige Relation (FLIEGT_NACH nicht erlaubt)",
        },
    ]

    results = {"accepted": 0, "rejected": 0}

    for i, update in enumerate(updates, 1):
        triple = update["triple"]
        print(f"\n  Update {i}: {triple.subject.name} --[{triple.predicate}]--> {triple.object.name}")
        print(f"    Erwartung: {update['erwartung']} ({update['grund']})")

        # Konsistenzprüfung durchführen
        validated = orchestrator.process(triple)

        if validated.validation_status == ValidationStatus.ACCEPTED:
            print(f"    Ergebnis:  ✓ AKZEPTIERT")
            graph.save_triple(validated)
            results["accepted"] += 1
        else:
            print(f"    Ergebnis:  ✗ ABGELEHNT")
            if hasattr(validated, 'conflicts') and validated.conflicts:
                for conflict in validated.conflicts:
                    if hasattr(conflict, 'conflict_type'):
                        print(f"    Konflikt:  {conflict.conflict_type.value}")
            results["rejected"] += 1

    # =========================================================================
    # SCHRITT 5: ZUSAMMENFASSUNG
    # =========================================================================
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    print(f"\n  Seed-Phase:    3 Triples direkt gespeichert")
    print(f"  Update-Phase:  {len(updates)} Triples geprüft")
    print(f"    → Akzeptiert: {results['accepted']}")
    print(f"    → Abgelehnt:  {results['rejected']}")

    print(f"\n  Finaler Graph:")
    print(f"    Entitäten:  {len(graph._entities)}")
    print(f"    Relationen: {len(graph._relations)}")

    print("\n  Alle Relationen im Graph:")
    seen = set()
    for rel in graph._relations.values():
        # Relation hat source_id und target_id, nicht subject/object
        source = graph._entities.get(rel.source_id, None)
        target = graph._entities.get(rel.target_id, None)
        source_name = source.name if source else rel.source_id
        target_name = target.name if target else rel.target_id
        key = (source_name, rel.relation_type, target_name)
        if key not in seen:
            print(f"    • {source_name} --[{rel.relation_type}]--> {target_name}")
            seen.add(key)

    # Metriken ausgeben
    print("\n  Konsistenzmodul-Metriken:")
    if orchestrator.metrics:
        m = orchestrator.metrics
        print(f"    Verarbeitete Triples: {m.total_triples_processed}")
        print(f"    Akzeptiert:           {m.total_accepted}")
        print(f"    Abgelehnt:            {m.total_rejected}")
        print(f"    Akzeptanzrate:        {m.acceptance_rate:.1%}")

    print("\n" + "-" * 70)
    print("HINWEISE:")
    print("-" * 70)
    print("""
  • Self-Loops werden IMMER erkannt (Schema-Validierung)
  • Ungültige Relationen werden IMMER erkannt (Schema-Validierung)
  • Duplikaterkennung benötigt Embedding-Modell für Namens-Variationen
  • Kardinalitätsprüfung funktioniert nur bei gleicher Entity-ID
  • Für Produktion: Entity-Resolution VOR Konsistenzprüfung durchführen
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
