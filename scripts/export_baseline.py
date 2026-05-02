#!/usr/bin/env python3
"""
Export Baseline zu verschiedenen Formaten.

Verwendung:
    python scripts/export_baseline.py data/baseline_graph_3.pkl --format json
    python scripts/export_baseline.py data/baseline_graph_3.pkl --format cypher
    python scripts/export_baseline.py data/baseline_graph_3.pkl --format neo4j --neo4j-uri bolt://localhost:7687
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime


def load_baseline(path: str) -> dict:
    """Lädt eine Baseline-Datei."""
    with open(path, "rb") as f:
        return pickle.load(f)


def export_json(data: dict, output_path: str):
    """Exportiert zu JSON (lesbar)."""
    export = {
        "metadata": {
            "saved_at": data.get("saved_at"),
            "stats": data["stats"],
        },
        "entities": [],
        "relations": [],
    }

    for entity_id, entity in data["entities"].items():
        export["entities"].append({
            "id": entity_id,
            "name": entity.name,
            "type": entity.entity_type.value,
            "confidence": entity.confidence,
            "source_document": entity.source_document,
        })

    for rel_id, rel in data["relations"].items():
        source = data["entities"].get(rel.source_id)
        target = data["entities"].get(rel.target_id)
        export["relations"].append({
            "id": rel_id,
            "source_id": rel.source_id,
            "source_name": source.name if source else None,
            "relation_type": rel.relation_type,
            "target_id": rel.target_id,
            "target_name": target.name if target else None,
            "confidence": rel.confidence,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON exportiert: {output_path}")
    print(f"  Entities: {len(export['entities'])}")
    print(f"  Relations: {len(export['relations'])}")


def export_cypher(data: dict, output_path: str):
    """Exportiert zu Cypher-Statements für Neo4j."""
    lines = [
        "// Baseline Export für Neo4j",
        f"// Erstellt: {datetime.now().isoformat()}",
        f"// Original: {data.get('saved_at', 'unbekannt')}",
        "",
        "// Lösche existierende Daten (optional)",
        "// MATCH (n) DETACH DELETE n;",
        "",
        "// === ENTITIES ===",
    ]

    # Entities
    for entity_id, entity in data["entities"].items():
        entity_type = entity.entity_type.value.upper()
        name_escaped = entity.name.replace("'", "\\'").replace('"', '\\"')

        cypher = (
            f"CREATE (:{entity_type} {{"
            f"id: '{entity_id}', "
            f"name: '{name_escaped}', "
            f"confidence: {entity.confidence}"
            f"}});"
        )
        lines.append(cypher)

    lines.append("")
    lines.append("// === RELATIONS ===")

    # Relations
    for rel_id, rel in data["relations"].items():
        source = data["entities"].get(rel.source_id)
        target = data["entities"].get(rel.target_id)

        if not source or not target:
            continue

        rel_type = rel.relation_type.upper().replace(" ", "_")

        cypher = (
            f"MATCH (a {{id: '{rel.source_id}'}}), (b {{id: '{rel.target_id}'}}) "
            f"CREATE (a)-[:{rel_type} {{id: '{rel_id}', confidence: {rel.confidence}}}]->(b);"
        )
        lines.append(cypher)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Cypher exportiert: {output_path}")
    print(f"  Statements: {len([l for l in lines if l.startswith('CREATE') or l.startswith('MATCH')])}")
    print(f"\nImport in Neo4j:")
    print(f"  1. Öffne Neo4j Browser")
    print(f"  2. Kopiere Inhalt von {output_path}")
    print(f"  3. Führe Statements aus")


def export_neo4j(data: dict, uri: str, user: str, password: str):
    """Exportiert direkt zu Neo4j."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("ERROR: neo4j package nicht installiert.")
        print("  pip install neo4j")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Lösche existierende Daten (optional)
        print("Lösche existierende Daten...")
        session.run("MATCH (n:BASELINE) DETACH DELETE n")

        # Entities erstellen
        print(f"Erstelle {len(data['entities'])} Entities...")
        for entity_id, entity in data["entities"].items():
            session.run(
                """
                CREATE (n:BASELINE {
                    id: $id,
                    name: $name,
                    type: $type,
                    confidence: $confidence
                })
                """,
                id=entity_id,
                name=entity.name,
                type=entity.entity_type.value,
                confidence=entity.confidence
            )

        # Relations erstellen
        print(f"Erstelle {len(data['relations'])} Relations...")
        for rel_id, rel in data["relations"].items():
            session.run(
                f"""
                MATCH (a:BASELINE {{id: $source_id}}), (b:BASELINE {{id: $target_id}})
                CREATE (a)-[r:{rel.relation_type.upper().replace(' ', '_')} {{
                    id: $rel_id,
                    confidence: $confidence
                }}]->(b)
                """,
                source_id=rel.source_id,
                target_id=rel.target_id,
                rel_id=rel_id,
                confidence=rel.confidence
            )

    driver.close()

    print(f"✓ Neo4j Export abgeschlossen!")
    print(f"  URI: {uri}")
    print(f"  Entities: {len(data['entities'])}")
    print(f"  Relations: {len(data['relations'])}")


def print_summary(data: dict):
    """Zeigt eine Zusammenfassung der Baseline."""
    print("\n" + "="*60)
    print("BASELINE ZUSAMMENFASSUNG")
    print("="*60)

    print(f"\nGespeichert: {data.get('saved_at', 'unbekannt')}")
    print(f"Entities:    {data['stats']['valid_entities']}")
    print(f"Relations:   {data['stats']['valid_relations']}")

    print("\n--- ENTITIES (erste 20) ---")
    for i, (entity_id, entity) in enumerate(data["entities"].items()):
        if i >= 20:
            print(f"  ... und {len(data['entities']) - 20} weitere")
            break
        print(f"  [{entity.entity_type.value:12}] {entity.name}")

    print("\n--- RELATIONS (erste 20) ---")
    for i, (rel_id, rel) in enumerate(data["relations"].items()):
        if i >= 20:
            print(f"  ... und {len(data['relations']) - 20} weitere")
            break
        source = data["entities"].get(rel.source_id)
        target = data["entities"].get(rel.target_id)
        source_name = source.name[:25] if source else "?"
        target_name = target.name[:25] if target else "?"
        print(f"  ({source_name}) --[{rel.relation_type}]--> ({target_name})")


def main():
    parser = argparse.ArgumentParser(description="Export Baseline zu verschiedenen Formaten")
    parser.add_argument("baseline_path", help="Pfad zur Baseline-Datei (.pkl)")
    parser.add_argument(
        "--format", choices=["json", "cypher", "neo4j", "summary"],
        default="summary", help="Ausgabeformat (default: summary)"
    )
    parser.add_argument("--output", "-o", help="Ausgabedatei (für json/cypher)")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j Benutzer")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j Passwort")

    args = parser.parse_args()

    if not Path(args.baseline_path).exists():
        print(f"ERROR: Datei nicht gefunden: {args.baseline_path}")
        sys.exit(1)

    data = load_baseline(args.baseline_path)

    if args.format == "summary":
        print_summary(data)

    elif args.format == "json":
        output = args.output or args.baseline_path.replace(".pkl", ".json")
        export_json(data, output)

    elif args.format == "cypher":
        output = args.output or args.baseline_path.replace(".pkl", ".cypher")
        export_cypher(data, output)

    elif args.format == "neo4j":
        export_neo4j(data, args.neo4j_uri, args.neo4j_user, args.neo4j_password)


if __name__ == "__main__":
    main()
