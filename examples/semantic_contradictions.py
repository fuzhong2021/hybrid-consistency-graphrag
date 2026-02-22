#!/usr/bin/env python3
"""
Beispiel: Widerspruchserkennung mit temporalen Regeln und semantischem Trigger

Zeigt wie das Konsistenzmodul verschiedene Widersprüche erkennt:

1. REGELBASIERT (Stage 1, ohne LLM):
   - Kardinalitätsverletzungen (z.B. zweiter Geburtsort)
   - Temporale Widersprüche (z.B. Preis nach dem Tod)
   - Self-Loops, Schema-Verletzungen

2. SEMANTISCHER TRIGGER (NEU):
   - Erkennt Rollen-Konflikte wie "Ehefrau = Mutter"
   - Ruft LLM nur bei potentiellen Widersprüchen auf
   - Reduziert LLM-Aufrufe auf 5-10% der Triples

3. LLM-BASIERT (Stage 3):
   - Wird automatisch durch semantischen Trigger aktiviert
   - Rein logische Widersprüche (z.B. Ehefrau = Mutter)
   - Komplexe semantische Inkonsistenzen

Verwendung:
    # Standard (semantischer Trigger aktiv, LLM bei Bedarf)
    python examples/semantic_contradictions.py --llm ollama

    # Ohne LLM (zeigt welche Widersprüche LLM benötigen)
    python examples/semantic_contradictions.py
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig


def create_entity(name: str, etype: EntityType = EntityType.PERSON) -> Entity:
    return Entity(name=name, entity_type=etype)


def create_triple(subj: Entity, pred: str, obj: Entity, source: str = "doc") -> Triple:
    return Triple(
        subject=subj,
        predicate=pred,
        object=obj,
        source_text=f"{subj.name} {pred} {obj.name}",
        source_document_id=source,
        extraction_confidence=0.85,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["ollama", "openai"], help="LLM aktivieren")
    parser.add_argument("--llm-model", default="llama3.1:8b")
    parser.add_argument("--disable-semantic-trigger", action="store_true",
                        help="Semantischen Trigger deaktivieren (zum Vergleich)")
    args = parser.parse_args()

    print("=" * 70)
    print("SEMANTISCHE WIDERSPRÜCHE - MIT SELEKTIVEM LLM-TRIGGER")
    print("=" * 70)

    # LLM laden falls gewünscht
    llm_client = None
    if args.llm == "ollama":
        try:
            from src.llm.ollama_client import OllamaClient
            llm_client = OllamaClient(model=args.llm_model)
            print(f"\n✓ LLM aktiv: {args.llm_model}")
        except Exception as e:
            print(f"\n✗ LLM nicht verfügbar: {e}")
    elif args.llm == "openai":
        try:
            from src.llm.openai_client import OpenAIClient
            llm_client = OpenAIClient()
            print(f"\n✓ LLM aktiv: OpenAI")
        except Exception as e:
            print(f"\n✗ LLM nicht verfügbar: {e}")
    else:
        print("\n⚠ Kein LLM - semantischer Trigger erkennt Konflikte, kann aber nicht prüfen!")
        print("  → Starte mit: python examples/semantic_contradictions.py --llm ollama")

    # Setup
    config = ConsistencyConfig(
        llm_model=args.llm_model,
        valid_relation_types=[
            "GEBOREN_IN", "GESTORBEN_IN", "GESTORBEN_AM",
            "VERHEIRATET_MIT", "GESCHIEDEN_VON",
            "IST_LEBENDIG", "IST_TOT",
            "VATER_VON", "MUTTER_VON", "KIND_VON",
            "ARBEITET_BEI", "ARBEITETE_BEI",
            "GEWANN", "ERHIELT",
        ],
        cardinality_rules={
            "GEBOREN_IN": {"max": 1},
            "GESTORBEN_IN": {"max": 1},
        },
        # Semantischer Trigger: aktivieren/deaktivieren
        enable_semantic_trigger=not args.disable_semantic_trigger,
    )

    graph = InMemoryGraphRepository()

    # ==========================================================================
    # SEED: Basis-Fakten laden
    # ==========================================================================
    print("\n" + "-" * 70)
    print("SEED: Basis-Fakten (werden direkt gespeichert)")
    print("-" * 70)

    # Entitäten
    einstein = create_entity("Albert Einstein")
    ulm = create_entity("Ulm", EntityType.LOCATION)
    princeton = create_entity("Princeton", EntityType.LOCATION)
    year_1955 = create_entity("1955", EntityType.EVENT)
    nobel = create_entity("Nobelpreis Physik 1921", EntityType.EVENT)
    elsa = create_entity("Elsa Einstein")

    seed_facts = [
        (einstein, "GEBOREN_IN", ulm, "Einstein wurde in Ulm geboren"),
        (einstein, "GESTORBEN_IN", princeton, "Einstein starb in Princeton"),
        (einstein, "GESTORBEN_AM", year_1955, "Einstein starb 1955"),
        (einstein, "VERHEIRATET_MIT", elsa, "Einstein war mit Elsa verheiratet"),
        (einstein, "GEWANN", nobel, "Einstein gewann den Nobelpreis 1921"),
    ]

    for subj, pred, obj, desc in seed_facts:
        t = create_triple(subj, pred, obj)
        t.validation_status = ValidationStatus.ACCEPTED
        graph.save_triple(t)
        print(f"  ✓ {subj.name} --[{pred}]--> {obj.name}")

    # Orchestrator NACH dem Seed
    orchestrator = ConsistencyOrchestrator(
        config=config,
        graph_repo=graph,
        llm_client=llm_client,
        enable_metrics=True,
    )

    # ==========================================================================
    # UPDATE: Semantische Widersprüche testen
    # ==========================================================================
    print("\n" + "-" * 70)
    print("UPDATE: Semantische Widersprüche prüfen")
    print("-" * 70)

    if not args.disable_semantic_trigger:
        print("\n  ✓ Semantischer Trigger AKTIV")
        print("    → LLM wird NUR bei erkannten Rollen-Konflikten aufgerufen")
    else:
        print("\n  ⚠ Semantischer Trigger DEAKTIVIERT")
        print("    → LLM nur bei niedriger Konfidenz aufgerufen")

    contradictions = [
        # 1. FAKTISCHER WIDERSPRUCH
        # Regel erkennt: Kardinalität (max 1 Geburtsort)
        {
            "triple": create_triple(
                einstein,  # GLEICHE Entity!
                "GEBOREN_IN",
                create_entity("München", EntityType.LOCATION)
            ),
            "beschreibung": "Faktisch: Zweiter Geburtsort",
            "erwartet": "REJECT (Kardinalität - Stufe 1)",
            "benoetigt_llm": False,
            "semantic_trigger_expected": False,
        },

        # 2. TEMPORALER WIDERSPRUCH
        # Temporale Regeln erkennen: Jahreszahl 1960 > Todesjahr 1955
        {
            "triple": create_triple(
                einstein,
                "GEWANN",
                create_entity("Nobelpreis Chemie 1960", EntityType.EVENT)
            ),
            "beschreibung": "Temporal: Preis nach dem Tod (1960 > 1955)",
            "erwartet": "REJECT (Temporale Regel)",
            "benoetigt_llm": False,
            "semantic_trigger_expected": False,
        },

        # 3. ZUSTANDSWIDERSPRUCH
        # Temporale Regeln erkennen: Jahreszahl 2020 > Todesjahr 1955
        {
            "triple": create_triple(
                einstein,
                "VERHEIRATET_MIT",
                create_entity("Neue Person 2020")
            ),
            "beschreibung": "Zustand: Heirat nach dem Tod (2020 > 1955)",
            "erwartet": "REJECT (Temporale Regel)",
            "benoetigt_llm": False,
            "semantic_trigger_expected": False,
        },

        # 4. LOGISCHER WIDERSPRUCH - SEMANTISCHER TRIGGER SOLLTE HIER GREIFEN!
        # Ehefrau kann nicht gleichzeitig Mutter sein
        {
            "triple": create_triple(
                elsa,
                "MUTTER_VON",
                einstein  # Elsa als Mutter von Einstein?
            ),
            "beschreibung": "Logisch: Ehefrau = Mutter? (SEMANTISCHER TRIGGER!)",
            "erwartet": "REJECT (Trigger → LLM erkennt Konflikt)",
            "benoetigt_llm": True,
            "semantic_trigger_expected": True,  # Hier sollte der Trigger greifen!
        },

        # 5. GÜLTIGES UPDATE (Kontrolle)
        {
            "triple": create_triple(
                create_entity("Max Planck"),
                "ARBEITET_BEI",
                create_entity("Berlin Universität", EntityType.ORGANIZATION)
            ),
            "beschreibung": "Gültig: Neuer unabhängiger Fakt",
            "erwartet": "ACCEPT",
            "benoetigt_llm": False,
            "semantic_trigger_expected": False,
        },
    ]

    results = {
        "accepted": 0,
        "rejected": 0,
        "llm_needed": 0,
        "semantic_trigger_fired": 0,
        "semantic_trigger_correct": 0,
    }

    for i, case in enumerate(contradictions, 1):
        triple = case["triple"]
        print(f"\n  [{i}] {case['beschreibung']}")
        print(f"      {triple.subject.name} --[{triple.predicate}]--> {triple.object.name}")
        print(f"      Erwartet: {case['erwartet']}")

        # Vor der Verarbeitung: Trigger-Stats merken
        trigger_stats_before = orchestrator.semantic_trigger.get_statistics()

        # Standard-Verarbeitung durch Orchestrator
        validated = orchestrator.process(triple)

        # Nach der Verarbeitung: Prüfen ob Trigger gefeuert hat
        trigger_stats_after = orchestrator.semantic_trigger.get_statistics()
        trigger_fired = trigger_stats_after["triggered"] > trigger_stats_before["triggered"]

        if trigger_fired:
            results["semantic_trigger_fired"] += 1
            if case["semantic_trigger_expected"]:
                results["semantic_trigger_correct"] += 1
                print(f"      → TRIGGER KORREKT AKTIVIERT!")

        if validated.validation_status == ValidationStatus.ACCEPTED:
            status = "✓ AKZEPTIERT"
            results["accepted"] += 1
        else:
            status = "✗ ABGELEHNT"
            results["rejected"] += 1

        print(f"      Ergebnis: {status}")

        if case["benoetigt_llm"]:
            results["llm_needed"] += 1
            if not llm_client and validated.validation_status == ValidationStatus.ACCEPTED:
                if trigger_fired:
                    print(f"      ⚠ Trigger hat Konflikt erkannt, aber kein LLM verfügbar!")
                else:
                    print(f"      ⚠ FALSCH AKZEPTIERT - LLM benötigt!")

        if validated.conflicts:
            for c in validated.conflicts:
                desc = c.description if hasattr(c, 'description') else str(c)
                print(f"      Konflikt: {desc[:60]}...")

    # ==========================================================================
    # ZUSAMMENFASSUNG
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    print(f"\n  Getestet:   {len(contradictions)} Triples")
    print(f"  Akzeptiert: {results['accepted']}")
    print(f"  Abgelehnt:  {results['rejected']}")

    # Semantischer Trigger Statistiken
    trigger_stats = orchestrator.semantic_trigger.get_statistics()
    print(f"\n  SEMANTISCHER TRIGGER:")
    print(f"    Geprüft: {trigger_stats['total_checked']} Triples")
    print(f"    Ausgelöst: {trigger_stats['triggered']} mal")
    print(f"    Trigger-Rate: {trigger_stats['trigger_rate']:.1%}")
    if trigger_stats["triggered"] > 0:
        print(f"    Korrekt (erwartet): {results['semantic_trigger_correct']}/{results['semantic_trigger_fired']}")

    # LLM-Nutzung
    print(f"\n  LLM-NUTZUNG:")
    print(f"    Triples die LLM benötigen: {results['llm_needed']}")
    if llm_client:
        print(f"    ✓ LLM war verbunden und konnte prüfen")
    else:
        print(f"    ⚠ Kein LLM - semantische Prüfung nicht möglich!")

    # Stufen-Statistik
    if orchestrator.metrics:
        m = orchestrator.metrics
        print(f"\n  STUFEN-STATISTIK:")
        print(f"    Stufe 1: {m.stage1_metrics.total_processed} verarbeitet")
        print(f"    Stufe 2: {m.stage2_metrics.total_processed} verarbeitet")
        print(f"    Stufe 3: {m.stage3_metrics.total_processed} verarbeitet (LLM-Aufrufe)")

        # Vergleich: Mit vs. ohne Trigger
        if not args.disable_semantic_trigger:
            total = m.stage1_metrics.total_processed
            stage3_calls = m.stage3_metrics.total_processed
            if total > 0:
                print(f"\n  EFFIZIENZ:")
                print(f"    LLM-Aufrufrate: {stage3_calls/total:.1%} ({stage3_calls}/{total} Triples)")
                print(f"    → Ohne Trigger wären 100% der Triples zu Stage 3 eskaliert!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
