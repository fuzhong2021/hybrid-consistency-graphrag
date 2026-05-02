# Hybrides Konsistenzmodul für GraphRAG (Doku mit Hilfe von KI erstellt)

Masterarbeit: *Konzeption und Evaluation eines hybriden Konsistenzmoduls zur Sicherung der Datenqualität in dynamischen GraphRAG-Pipelines.*

## Kernfeatures

| Feature | Beschreibung | Mechanismus |
|---------|--------------|-------------|
| **Vierstufige Validierung** | Regel → Embedding → NLI → LLM Kaskade | Konfidenzbasiertes Routing mit Fast-Path |
| **Missing Source Penalty** | Konfidenz-Multiplikator für quellenlose Fakten | `× 0.7` (≈30 % Reduktion) |
| **Source Verification** | Embedding-Ähnlichkeit oder NLI (Triple ↔ Quelltext) | Penalty bei Sim < 0.5 (Embedding) bzw. Contradiction (NLI) |
| **Kardinalitätsprüfung** | Erkennt widersprüchliche Fakten | Soft Penalty (× 0.6) → Eskalation an Stage 2+ |
| **NLI-Stufe** | DeBERTa-v3-xsmall Entailment/Contradiction | Default an; Ablations-Beleg siehe `results/analysis/` |
| **Provenance Boost** | Multiple Quellen erhöhen Konfidenz | × 1.1 (2 Quellen), × 1.2 (3+) |

> Quantitative Effekte siehe Abschnitt *Evaluationsergebnisse* — Werte aus dem Code (`src/consistency/base.py`) und den Result-JSONs in `results/`, nicht aus früheren README-Versionen.

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
./scripts/setup_and_run.sh          # 50 Beispiele
./scripts/setup_and_run.sh 7400     # Volle Evaluation
```

### Baseline-Persistenz

Die Baseline wird automatisch gespeichert und bei späteren Läufen geladen:

```bash
# Erster Lauf: Baseline wird erstellt
./scripts/setup_and_run.sh 7400

# Zweiter Lauf: Baseline wird geladen (Phase 1 übersprungen!)
./scripts/setup_and_run.sh 7400

# Baseline in Neo4j visualisieren
python scripts/export_baseline.py data/baseline_graph_7400.pkl --format cypher
```

## Architektur

```
Triple ──► Stufe 1: Regelbasiert ──► Stufe 2: Embedding ──► Stufe 3: NLI ──► Stufe 4: LLM
              │                           │                       │                │
              ▼                           ▼                       ▼                ▼
         Schema-Check              Duplikaterkennung        Entailment /     Konfliktauflösung
         Kardinalität              Entity Resolution        Contradiction    Faktenverifikation
         Source Verification       Missing Source           (DeBERTa-v3)     Chain-of-Thought
```

Ziel der Kaskade: Konsistenz erhöhen *und* LLM-Aufrufe minimieren. Jede Stufe entscheidet bei hoher Konfidenz selbstständig (Fast-Path). Die LLM-Stufe wird nur dann aufgerufen, wenn die Vorstufen einen Konflikt finden, den sie nicht eindeutig auflösen können — empirisch sind das zwischen 8 % (HotpotQA) und 40 % (FEVER) der Triples (siehe `results/analysis/stage_distribution.md`).

## Evaluationsergebnisse

### Hauptergebnisse — Multi-Dataset (51 635 Triples)

`results/multi_dataset_evaluation.json` (29 284 HotpotQA + 9 515 FEVER + 12 836 MuSiQue, mit LLM-Stufe):

| Datensatz | Triples | Precision | Recall | F1 | Accuracy |
|-----------|---------|-----------|--------|-----|----------|
| HotpotQA | 29 284 | 0.769 | 0.725 | **0.747** | 0.744 |
| FEVER | 9 515 | 0.762 | 0.743 | **0.753** | 0.727 |
| MuSiQue | 12 836 | 0.490 | 0.878 | **0.629** | 0.588 |
| **Ø** | 51 635 | — | — | **0.709** | 0.687 |

> **Beobachtung FEVER:** Ohne Stage-3-LLM (`results/full_evaluation_no_llm.json`) liegt der FEVER-F1 bei **0.883** (Recall 0.989). Die Stage-3-LLM-Stufe verschlechtert FEVER um ≈13 F1-Punkte. **Wichtig:** Die NLI-Stufe war in *beiden* Hauptläufen aktiv (`--nli`-Flag), die Begriffe "no_llm" / "with_llm" beziehen sich nur auf Stufe 3. Eine saubere NLI-vs-LLM-Trennung liefert `evaluation/run_nli_ablation.py`. Detailanalyse: `results/analysis/EXECUTIVE_SUMMARY.md`.
>
> **Beobachtung MuSiQue:** Precision 0.490 → das System produziert mehr False Positives als True Positives. Wahrscheinliche Ursache: deutsche Domain-Constraints in `ConsistencyConfig` greifen nicht für englische MuSiQue-Relationen (siehe Phase 2.2-Analyse).

### 6-Phasen-Sanity-Check (3 Beispiele, **nicht statistisch belastbar**)

`results/hotpotqa_realistic_evaluation.json` — sample_size = 3, dient ausschließlich als End-to-End-Smoke-Test:

| Phase | Beschreibung | Beobachtetes Verhalten | n |
|-------|--------------|------------------------|---|
| Phase 1: Baseline | Korrekte Fakten aus GOLD | 6/6 akzeptiert | 6 |
| Phase 2: Mit Quelle | Fakten mit `source_document_id` | 5/6 akzeptiert (1 FP) | 6 |
| Phase 3: Ohne Quelle | Missing Source Penalty wirksam | 5 Penalties, **30 % Konfidenz-Reduktion** | 6 |
| Phase 4: Kardinalität | Widersprüchliche Antworten | F1 = 1.00 (6/6 erkannt) | 6 |
| Phase 5: Cross-Question | Antworten anderer Fragen | F1 = 1.00 (3/3 erkannt) | 3 |
| Phase 6: Fake Source | Quelle stützt Claim nicht | 41 % Konfidenz-Reduktion | 3 |

> Mit n = 3–6 hat der Sanity-Check keine Aussagekraft über reale Performance. Belastbare Werte stehen in der Multi-Dataset-Tabelle oben.

## Schutz vor Manipulation (gemessene Mechanismus-Wirkung)

| Angriff | Schutz | Konfidenz-Effekt | Belegt in |
|---------|--------|------------------|-----------|
| Keine Quelle angeben | Missing Source Penalty (× 0.7) | **−30 %** | Phase 3, 6 Triples |
| Falsche Quelle angeben | Source Verification (Sim < 0.5 bzw. NLI Contradiction) | **−41 %** | Phase 6, 3 Triples |
| Widersprüchliche Fakten | Cardinality Soft Penalty (Stage 1) → Stage 2+ | F1 = 0.874 (n = 98) | Phase 4 (n=50) |

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
├── hotpotqa_realistic_evaluation.py  # 6-Phasen Evaluation
├── multi_dataset_evaluation.py       # Multi-Dataset (HotpotQA, FEVER, MuSiQue)
├── run_nli_ablation.py               # NLI-Ablation (mit/ohne NLI)
├── threshold_grid_search.py          # Schwellenwert-Rastersuche (θ_H, θ_M)
└── baselines/                        # Baseline-Vergleich (Random, Rules, Embedding, LLM, Full)

scripts/
├── setup_and_run.sh            # Automatisches Setup + Evaluation
└── export_baseline.py          # Baseline Export (JSON, Cypher, Neo4j)

data/
└── baseline_graph_{SIZE}.pkl   # Persistierte Baselines
```

## Konfiguration

> **Verbindlich** ist die Python-Klasse `ConsistencyConfig` in `src/consistency/base.py`.
> Die YAML-Datei `configs/config.yaml` beschreibt die übergeordnete GraphRAG-Pipeline (LLM-Provider, Chunking, Retrieval) und wird vom Konsistenzmodul *nicht* gelesen — Header der YAML-Datei verweist explizit darauf.
> Die in den Evaluationen tatsächlich verwendete Konfiguration steht inline in `evaluation/multi_dataset_evaluation.py` ab Zeile 102 und kombiniert englische Benchmark-Relationen (HotpotQA/FEVER) mit deutschen Standard-Relationen.

```python
from src.consistency.base import ConsistencyConfig
from src.consistency.orchestrator import ConsistencyOrchestrator

config = ConsistencyConfig(
    # Validierung — Mischung aus EN-Benchmark- und DE-Standardrelationen
    valid_relation_types=[
        "RELATED_TO", "SUPPORTS", "ANSWERS", "HAS_ANSWER", "CONFIRMS",
        "CLAIMS", "REFUTES", "SUPPORTS_CLAIM",
        "GEBOREN_IN", "ARBEITET_BEI", "LEITET",  # ...
    ],
    high_confidence_threshold=0.9,
    medium_confidence_threshold=0.7,

    # Missing Source Penalty (Multiplikator, kein Abzug)
    enable_missing_source_penalty=True,
    missing_source_penalty=0.7,  # → 30 % Reduktion der Konfidenz

    # Source Verification (Embedding-Sim Triple ↔ Quelltext)
    enable_source_verification=True,
    source_verification_threshold=0.5,  # oder method="nli" für NLI-basierte Verification

    # Kardinalitätsregeln (HotpotQA: 1 Antwort pro Frage)
    cardinality_rules={
        "HAS_ANSWER": {"max": 1},
        "ANSWERS": {"max": 1},
    },

    # NLI-Stufe — fester Bestandteil der Kaskade
    enable_nli=True,
    nli_model="cross-encoder/nli-deberta-v3-xsmall",

    # LLM (Ollama)
    llm_model="llama3.1:8b",
)

orchestrator = ConsistencyOrchestrator(config=config)
result = orchestrator.process(triple)
```

## Dokumentation

| Dokument | Beschreibung |
|----------|--------------|
| [Executive Summary](results/analysis/EXECUTIVE_SUMMARY.md) | Konsolidierter Befundbericht mit allen Kennzahlen, CIs und Satz-Bausteinen |
| [System Overview](docs/system_overview.md) | Architektur, Design und wissenschaftliche Grundlagen |
| [Konfidenzintervalle](results/analysis/confidence_intervals.md) | Wilson + Bootstrap 95 %-CIs für alle Hauptzahlen |
| [NLI-Ablation + McNemar + ECE](results/analysis/nli_ablation.md) | Signifikanztests und Kalibrierung |
| [Stage-Verteilung + Latenz](results/analysis/stage_distribution.md) | Empirische Stage-Verteilung pro Datensatz |
| [Kostenanalyse](results/analysis/cost_analysis.md) | GPT-4-Kostenschätzung aus LLM-Call-Counts |
| [Präsentationstext](docs/PRAESENTATION_TEXT.md) | Vortragsskript für die Verteidigung |

## Wissenschaftliche Grundlagen

- **Graphiti** (Rasmussen et al., 2025): Bi-temporales Datenmodell
- **iText2KG** (Lairgi et al., 2024): Gewichtete Ähnlichkeit für Entity Resolution
- **HotpotQA** (Yang et al., 2018): Multi-Hop Reasoning Benchmark
- **SHACL**: Schema-Validierung für Knowledge Graphs

## Lizenz

Masterarbeit - Alle Rechte vorbehalten.
