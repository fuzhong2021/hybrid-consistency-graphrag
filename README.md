# Hybrides Konsistenzmodul für GraphRAG

Masterarbeit: Design und Evaluation eines hybriden Konsistenzmoduls zur Datenqualitätssicherung in dynamischen GraphRAG-Pipelines.

## Kernfeatures

| Feature | Beschreibung | Ergebnis |
|---------|--------------|----------|
| **Dreistufige Validierung** | Regel → Embedding → LLM Kaskade | Effizient + Präzise |
| **Missing Source Penalty** | Konfidenz-Abzug für quellenlose Fakten | 25% Reduktion |
| **Source Verification** | Prüft ob Quelle den Claim belegt | 40% Reduktion bei Fake |
| **Kardinalitätsprüfung** | Erkennt widersprüchliche Fakten | 89% F1 Score |
| **Provenance Tracking** | Quellen-Reliabilität lernen | Automatisch |

## Schnellstart

```bash
# 1. Installation
pip install -r requirements.txt

# 2. Ollama starten (für LLM-Stufe)
ollama serve &
ollama pull llama3.1:8b

# 3. Evaluation ausführen
python evaluation/hotpotqa_realistic_evaluation.py --sample-size 50
```

Oder mit dem Setup-Script:

```bash
./scripts/setup_and_run.sh
```

## Architektur

```
Triple ──► Stufe 1: Regelbasiert ──► Stufe 2: Embedding ──► Stufe 3: LLM
              │                           │                      │
              ▼                           ▼                      ▼
         Schema-Check              Duplikaterkennung      Konfliktauflösung
         Kardinalität              Entity Resolution      Faktenverifikation
         Source Verification       Missing Source         Chain-of-Thought
```

## Evaluationsergebnisse (6-Phasen)

Getestet auf HotpotQA mit realistischen Kontradiktionen:

| Phase | Beschreibung | Ergebnis |
|-------|--------------|----------|
| Phase 1: Baseline | Korrekte Fakten aus GOLD | 100% akzeptiert |
| Phase 2: Mit Quelle | Fakten mit source_document_id | 100% akzeptiert |
| Phase 3: Ohne Quelle | Missing Source Penalty Test | **25% Reduktion** |
| Phase 4: Kardinalität | Widersprüchliche Antworten | **89% F1** |
| Phase 5: Cross-Question | Antworten von anderen Fragen | **89% F1** |
| Phase 6: Fake Source | Quelle unterstützt Claim nicht | **40% Reduktion** |

## Schutz vor Manipulation

| Angriff | Schutz | Effektivität |
|---------|--------|--------------|
| Keine Quelle angeben | Missing Source Penalty | -25% Konfidenz |
| Falsche Quelle angeben | Source Verification | -40% Konfidenz |
| Widersprüchliche Fakten | Kardinalitätsprüfung | 89% erkannt |

## Projektstruktur

```
src/
├── consistency/
│   ├── orchestrator.py         # Koordination der 3 Stufen
│   ├── rules/                  # Stufe 1: Regelbasierte Validierung
│   ├── embedding_validator.py  # Stufe 2: Embedding-basiert
│   ├── llm_arbitrator.py       # Stufe 3: LLM-Konfliktauflösung
│   ├── provenance.py           # Quellen-Tracking + Missing Source Penalty
│   └── source_verification.py  # Source Verification (NEU)
├── models/
│   └── entities.py             # Entity, Triple, ConflictSet
├── graph/
│   ├── memory_repository.py    # In-Memory Graph
│   └── repository.py           # Neo4j Anbindung
└── evaluation/
    └── benchmark_loader.py     # HotpotQA, MuSiQue Loader

evaluation/
└── hotpotqa_realistic_evaluation.py  # 6-Phasen Evaluation

scripts/
└── setup_and_run.sh            # Automatisches Setup + Evaluation
```

## Konfiguration

```python
from src.consistency.base import ConsistencyConfig
from src.consistency.orchestrator import ConsistencyOrchestrator

config = ConsistencyConfig(
    # Validierung
    valid_relation_types=["RELATED_TO", "SUPPORTS", "ANSWERS", "HAS_ANSWER"],
    high_confidence_threshold=0.9,
    medium_confidence_threshold=0.7,

    # Missing Source Penalty
    enable_missing_source_penalty=True,
    missing_source_penalty=0.7,  # 30% Abzug

    # Source Verification (NEU)
    enable_source_verification=True,
    source_verification_threshold=0.3,

    # Kardinalitätsregeln
    cardinality_rules={
        "HAS_ANSWER": {"max": 1},
        "ANSWERS": {"max": 1},
    },

    # LLM (Ollama)
    llm_model="llama3.1:8b",
)

orchestrator = ConsistencyOrchestrator(config=config)
result = orchestrator.process(triple)
```

## Dokumentation

| Dokument | Beschreibung |
|----------|--------------|
| [System Overview](docs/system_overview.md) | Architektur und Design |
| [Execution Guide](docs/EXECUTION_GUIDE.md) | Ausführungsanleitung |
| [Reproduktion](docs/REPRODUKTION_ANLEITUNG.md) | Schritt-für-Schritt Reproduktion |

## Wissenschaftliche Grundlagen

- **Graphiti** (Rasmussen et al., 2025): Bi-temporales Datenmodell
- **iText2KG** (Lairgi et al., 2024): Gewichtete Ähnlichkeit für Entity Resolution
- **HotpotQA** (Yang et al., 2018): Multi-Hop Reasoning Benchmark
- **SHACL**: Schema-Validierung für Knowledge Graphs

## Lizenz

Masterarbeit - Alle Rechte vorbehalten.
