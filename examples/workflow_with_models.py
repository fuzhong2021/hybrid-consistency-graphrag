#!/usr/bin/env python3
"""
Beispiel: Konsistenzmodul mit konfigurierbaren Modellen

Zeigt wie man:
1. Embedding-Modell für Stufe 2 (Duplikaterkennung) konfiguriert
2. LLM für Stufe 3 (semantische Widersprüche) konfiguriert
3. Alles als CLI-Parameter übergibt

Verwendung:
    # Ohne Modelle (nur Stufe 1)
    python examples/workflow_with_models.py

    # Mit Ollama LLM
    python examples/workflow_with_models.py --llm ollama --llm-model llama3.1:8b

    # Mit OpenAI (API-Key in .env)
    python examples/workflow_with_models.py --llm openai --llm-model gpt-4

    # Mit Sentence-Transformers Embedding
    python examples/workflow_with_models.py --embedding sentence-transformers

    # Vollständig (alle 3 Stufen)
    python examples/workflow_with_models.py --llm ollama --embedding sentence-transformers
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig


# =============================================================================
# MODELL-FACTORIES
# =============================================================================

def create_embedding_model(provider: str = None, model_name: str = None):
    """
    Erstellt ein Embedding-Modell.

    Unterstützte Provider:
    - sentence-transformers: Lokale Modelle (z.B. all-MiniLM-L6-v2)
    - openai: OpenAI Embeddings (API-Key erforderlich)
    - ollama: Ollama Embeddings (lokal)

    Returns:
        Embedding-Modell mit .embed_query() Methode oder None
    """
    if not provider:
        print("  → Kein Embedding-Modell (Stufe 2 eingeschränkt)")
        return None

    if provider == "sentence-transformers":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            print(f"  → Lade Sentence-Transformers: {model_name}")
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            print("  ✗ sentence-transformers nicht installiert")
            print("    pip install sentence-transformers langchain-community")
            return None

    elif provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            print(f"  → Lade OpenAI Embeddings")
            return OpenAIEmbeddings()
        except ImportError:
            print("  ✗ langchain-openai nicht installiert")
            print("    pip install langchain-openai")
            return None

    elif provider == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            model_name = model_name or "nomic-embed-text"
            print(f"  → Lade Ollama Embeddings: {model_name}")
            return OllamaEmbeddings(model=model_name)
        except ImportError:
            print("  ✗ langchain-community nicht installiert")
            return None

    else:
        print(f"  ✗ Unbekannter Embedding-Provider: {provider}")
        return None


def create_llm_client(provider: str = None, model_name: str = None):
    """
    Erstellt einen LLM-Client.

    Unterstützte Provider:
    - ollama: Lokale LLMs via Ollama (empfohlen)
    - openai: OpenAI API (API-Key erforderlich)

    Returns:
        LLM-Client mit OpenAI-kompatiblem Interface oder None
    """
    if not provider:
        print("  → Kein LLM (Stufe 3 deaktiviert)")
        return None

    if provider == "ollama":
        try:
            from src.llm.ollama_client import OllamaClient
            model_name = model_name or "llama3.1:8b"
            print(f"  → Verbinde mit Ollama: {model_name}")

            client = OllamaClient(model=model_name)
            # Test-Anfrage
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print(f"  ✓ Ollama verbunden")
            return client
        except Exception as e:
            print(f"  ✗ Ollama nicht erreichbar: {e}")
            print("    Starte Ollama mit: ollama serve")
            return None

    elif provider == "openai":
        try:
            import openai
            import os
            if not os.getenv("OPENAI_API_KEY"):
                print("  ✗ OPENAI_API_KEY nicht gesetzt")
                return None
            print(f"  → Verbinde mit OpenAI: {model_name or 'gpt-4'}")
            return openai.OpenAI()
        except ImportError:
            print("  ✗ openai nicht installiert")
            print("    pip install openai")
            return None

    else:
        print(f"  ✗ Unbekannter LLM-Provider: {provider}")
        return None


# =============================================================================
# BEISPIEL-DATEN
# =============================================================================

def create_entity(name: str, entity_type: EntityType) -> Entity:
    return Entity(name=name, entity_type=entity_type)


def create_triple(subject: Entity, predicate: str, obj: Entity) -> Triple:
    return Triple(
        subject=subject,
        predicate=predicate,
        object=obj,
        source_text=f"{subject.name} {predicate} {obj.name}",
        source_document_id="example",
        extraction_confidence=0.85,
    )


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Konsistenzmodul mit konfigurierbaren Modellen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Nur Stufe 1 (keine Modelle)
  python examples/workflow_with_models.py

  # Mit Ollama LLM (Stufe 1 + 3)
  python examples/workflow_with_models.py --llm ollama

  # Mit Embeddings (Stufe 1 + 2)
  python examples/workflow_with_models.py --embedding sentence-transformers

  # Alle 3 Stufen
  python examples/workflow_with_models.py --llm ollama --embedding sentence-transformers
        """
    )

    parser.add_argument(
        "--embedding",
        choices=["sentence-transformers", "openai", "ollama"],
        help="Embedding-Provider für Stufe 2"
    )
    parser.add_argument(
        "--embedding-model",
        help="Spezifisches Embedding-Modell (optional)"
    )
    parser.add_argument(
        "--llm",
        choices=["ollama", "openai"],
        help="LLM-Provider für Stufe 3"
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.1:8b",
        help="LLM-Modell (default: llama3.1:8b)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("KONSISTENZMODUL - KONFIGURIERBARE MODELLE")
    print("=" * 70)

    # =========================================================================
    # MODELLE LADEN
    # =========================================================================
    print("\n[1] MODELLE INITIALISIEREN")
    print("-" * 50)

    embedding_model = create_embedding_model(args.embedding, args.embedding_model)
    llm_client = create_llm_client(args.llm, args.llm_model)

    # =========================================================================
    # KONSISTENZMODUL SETUP
    # =========================================================================
    print("\n[2] KONSISTENZMODUL KONFIGURIEREN")
    print("-" * 50)

    config = ConsistencyConfig(
        llm_model=args.llm_model,  # LLM-Modell für Stage 3
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        similarity_threshold=0.85,
        valid_entity_types=["Person", "Organisation", "Ort"],
        valid_relation_types=["GEBOREN_IN", "ARBEITET_BEI", "KENNT", "WOHNT_IN"],
        cardinality_rules={"GEBOREN_IN": {"max": 1}},
    )

    graph = InMemoryGraphRepository()

    # Seed-Daten
    einstein = create_entity("Albert Einstein", EntityType.PERSON)
    ulm = create_entity("Ulm", EntityType.LOCATION)
    eth = create_entity("ETH Zürich", EntityType.ORGANIZATION)

    for t in [
        create_triple(einstein, "GEBOREN_IN", ulm),
        create_triple(einstein, "ARBEITET_BEI", eth),
    ]:
        t.validation_status = ValidationStatus.ACCEPTED
        graph.save_triple(t)

    print(f"  Seed: {len(graph._relations)} Relationen geladen")

    # Orchestrator mit Modellen
    orchestrator = ConsistencyOrchestrator(
        config=config,
        graph_repo=graph,
        embedding_model=embedding_model,  # ← Stufe 2
        llm_client=llm_client,            # ← Stufe 3
        enable_metrics=True,
        always_check_duplicates=True,
    )

    print(f"\n  Aktive Stufen:")
    print(f"    Stufe 1 (Regeln):     ✓ Immer aktiv")
    print(f"    Stufe 2 (Embeddings): {'✓' if embedding_model else '✗'}")
    print(f"    Stufe 3 (LLM):        {'✓' if llm_client else '✗'}")

    # =========================================================================
    # TEST-UPDATES
    # =========================================================================
    print("\n[3] UPDATES TESTEN")
    print("-" * 50)

    test_cases = [
        # Sollte akzeptiert werden
        {
            "triple": create_triple(
                create_entity("Max Planck", EntityType.PERSON),
                "KENNT",
                einstein
            ),
            "beschreibung": "Neue gültige Relation",
        },
        # Self-Loop (Stufe 1 erkennt)
        {
            "triple": create_triple(einstein, "KENNT", einstein),
            "beschreibung": "Self-Loop",
        },
        # Kardinalitätsverletzung (Stufe 1 erkennt)
        {
            "triple": create_triple(
                einstein,
                "GEBOREN_IN",
                create_entity("München", EntityType.LOCATION)
            ),
            "beschreibung": "Kardinalität (schon in Ulm geboren)",
        },
        # Duplikat mit Variation (nur Stufe 2 erkennt)
        {
            "triple": create_triple(
                create_entity("A. Einstein", EntityType.PERSON),  # Variation!
                "ARBEITET_BEI",
                eth
            ),
            "beschreibung": "Duplikat-Variation (A. Einstein = Albert Einstein?)",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        triple = case["triple"]
        print(f"\n  Test {i}: {case['beschreibung']}")
        print(f"    Triple: {triple.subject.name} --[{triple.predicate}]--> {triple.object.name}")

        validated = orchestrator.process(triple)

        status = "✓ AKZEPTIERT" if validated.validation_status == ValidationStatus.ACCEPTED else "✗ ABGELEHNT"
        print(f"    Ergebnis: {status}")

        if validated.conflicts:
            for c in validated.conflicts:
                desc = c.description if hasattr(c, 'description') else str(c)
                print(f"    Konflikt: {desc[:60]}...")

    # =========================================================================
    # METRIKEN
    # =========================================================================
    print("\n" + "=" * 70)
    print("METRIKEN")
    print("=" * 70)

    if orchestrator.metrics:
        m = orchestrator.metrics
        print(f"\n  Verarbeitet: {m.total_triples_processed}")
        print(f"  Akzeptiert:  {m.total_accepted}")
        print(f"  Abgelehnt:   {m.total_rejected}")
        print(f"  Rate:        {m.acceptance_rate:.1%}")

        print(f"\n  Stufen-Durchlauf:")
        print(f"    Stufe 1: {m.stage1_metrics.total_processed} verarbeitet")
        print(f"    Stufe 2: {m.stage2_metrics.total_processed} verarbeitet")
        print(f"    Stufe 3: {m.stage3_metrics.total_processed} verarbeitet")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
