# Wissenschaftliche Dokumentation: Streaming Evaluation für GraphRAG Konsistenzmodul

**Version:** 1.0
**Autor:** Masterarbeit GraphRAG Konsistenzprüfung
**Datum:** 2024

---

## 1. Zusammenfassung

Dieses Dokument beschreibt das wissenschaftlich fundierte Evaluationsframework für das GraphRAG Konsistenzmodul. Das Framework testet systematisch **alle 10 identifizierten Konflikt-Kategorien** in Knowledge Graphs und ermöglicht einen fairen Vergleich zwischen verschiedenen Ansätzen.

### Kernbeiträge

1. **Taxonomie von 10 KG-Konflikt-Typen** mit formalen Definitionen
2. **21 wissenschaftliche Referenzen** als Grundlage
3. **Reproduzierbares Testframework** (seed=42)
4. **Keine Kardinalitätsregeln** für faire NLI-Evaluation

---

## 2. Konflikt-Taxonomie

### 2.1 Übersicht

| # | Kategorie | Definition | Erkennung | Ground Truth |
|---|-----------|------------|-----------|--------------|
| 1 | FACTUAL | Direkte faktische Widersprüche | NLI + Kardinalität | REJECT |
| 2 | TEMPORAL | Zeitlich getrennte Aussagen | Temporal Reasoning | ACCEPT (disjunkt) |
| 3 | GRANULARITY | Verschiedene Abstraktionsebenen | Ontologie-Hierarchie | MERGE |
| 4 | ENTITY_VARIANT | Koreferenz/Entity Resolution | Embedding + String | MERGE |
| 5 | IMPLICIT | Widersprüche durch Inferenz | LLM + Weltwissen | REJECT |
| 6 | NEGATION | Direkte Verneinung | Pattern + NLI | REJECT |
| 7 | MODALITY | Unterschiedliche Gewissheitsgrade | Modal Markers | WEIGHT |
| 8 | SOURCE_QUALITY | Quellen-Vertrauenswürdigkeit | Source Scoring | WEIGHT |
| 9 | SCHEMA | Schema-Heterogenität | Schema Alignment | MERGE |
| 10 | NUMERICAL | Numerische Präzision | Range Overlap | MERGE |

### 2.2 Schwierigkeitsverteilung

- **Easy (4):** FACTUAL, ENTITY_VARIANT, NEGATION, NUMERICAL
- **Medium (5):** TEMPORAL, GRANULARITY, MODALITY, SOURCE_QUALITY, SCHEMA
- **Hard (1):** IMPLICIT

---

## 3. Wissenschaftliche Referenzen

### 3.1 Fact Verification & NLI

| Referenz | Beitrag |
|----------|---------|
| Thorne et al. (2018) | FEVER Dataset - NAACL |
| Bowman et al. (2015) | SNLI Dataset - EMNLP |
| Williams et al. (2018) | MultiNLI - Cross-Domain NLI |

### 3.2 Entity Resolution

| Referenz | Beitrag |
|----------|---------|
| Lairgi et al. (2024) | iText2KG (α=0.6 Name, 0.4 Embedding) |
| Shen et al. (2015) | Entity Linking Survey - TKDE |

### 3.3 Temporal Knowledge Graphs

| Referenz | Beitrag |
|----------|---------|
| Leblay & Chekol (2018) | Temporal KG Validity - WWW |
| Lacroix et al. (2020) | Tensor Decomposition for TKG - ICLR |
| García-Durán et al. (2018) | Sequence Encoders for TKG - EMNLP |

### 3.4 Negative Knowledge

| Referenz | Beitrag |
|----------|---------|
| Razniewski et al. (2016) | Negative Statements - AKBC |
| Arnaout et al. (2021) | Negative Statements Considered Useful - JAIR |
| Arnaout et al. (2022) | Negative Knowledge in Wikidata - WWW |

### 3.5 Uncertainty & Probabilistic KGs

| Referenz | Beitrag |
|----------|---------|
| Safavi et al. (2020) | Calibration in KG Embeddings - EMNLP |
| Chen et al. (2019) | Uncertain Knowledge Graphs - AAAI |

### 3.6 Source Quality

| Referenz | Beitrag |
|----------|---------|
| Dong et al. (2015) | Knowledge Vault - VLDB |
| Pasternack & Roth (2013) | Source Credibility - WWW |

### 3.7 Schema & Ontology

| Referenz | Beitrag |
|----------|---------|
| Paulheim (2017) | KG Refinement Survey - SWJ |
| Euzenat & Shvaiko (2013) | Ontology Matching - Springer |
| Hobbs (1985) | Granularity - IJCAI |

### 3.8 Streaming KG

| Referenz | Beitrag |
|----------|---------|
| Heist & Paulheim (2019) | Streaming KG Construction |

---

## 4. Methodologie

### 4.1 Testdesign

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATIONSPIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [FEVER Dataset] ──► [Basis-Triples] ──► [10 Generatoren]   │
│                                              │               │
│                                              ▼               │
│                                    [Konflikt-Triples]        │
│                                              │               │
│                                              ▼               │
│                                       [Shuffle]              │
│                                              │               │
│                         ┌────────────────────┼────────────┐  │
│                         │                    │            │  │
│                         ▼                    ▼            ▼  │
│                    [Random]            [Rules-Only]   [HYBRID]│
│                         │                    │            │  │
│                         └────────────────────┼────────────┘  │
│                                              │               │
│                                              ▼               │
│                                    [Metriken pro Typ]        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Kritisches Designprinzip

**KEINE KARDINALITÄTSREGELN** für Hauptrelationen (CLAIMS, STATES, etc.)

Begründung:
- Kardinalitätsregeln bevorzugen regelbasierte Ansätze unfair
- Echte Widersprüche müssen semantisch erkannt werden
- Das testet die NLI/Embedding-Fähigkeiten des Hybrid-Systems

### 4.3 Ground Truth Definition

| Aktion | Definition | Beispiel |
|--------|------------|----------|
| ACCEPT | Triple in Graph aufnehmen | Korrekte Fakten |
| REJECT | Triple ablehnen | Faktische Widersprüche |
| MERGE | Mit existierendem Triple vereinen | Entity-Varianten |
| WEIGHT | Konfidenz anpassen | Modalität, Source |

---

## 5. Erwartete Ergebnisse

### 5.1 Baseline-Vergleich (Hypothesen)

| System | F1 (erwartet) | Stärken | Schwächen |
|--------|---------------|---------|-----------|
| Random | ~0.50 | - | Keine Analyse |
| Rules-Only | ~0.45 | Schema, (Kardinalität) | Keine Semantik |
| Embedding-Only | ~0.60 | Entity, Granularity | Keine Negation |
| NLI-Only | ~0.65 | Factual, Negation | Kein Entity Match |
| LLM-Only | ~0.70 | Implicit, Reasoning | Langsam, teuer |
| **HYBRID** | **~0.82** | Alle kombiniert | - |

### 5.2 Erwartete Per-Typ Performance

| Konflikt-Typ | Rules | Embedding | NLI | Hybrid |
|--------------|-------|-----------|-----|--------|
| FACTUAL | 0.30 | 0.40 | **0.85** | **0.90** |
| TEMPORAL | 0.20 | 0.30 | 0.50 | **0.70** |
| GRANULARITY | 0.20 | **0.75** | 0.40 | **0.80** |
| ENTITY_VARIANT | 0.30 | **0.85** | 0.50 | **0.90** |
| IMPLICIT | 0.10 | 0.20 | 0.60 | **0.75** |
| NEGATION | 0.20 | 0.30 | **0.80** | **0.85** |
| MODALITY | 0.40 | 0.50 | 0.60 | **0.70** |
| SOURCE_QUALITY | **0.60** | 0.50 | 0.40 | **0.70** |
| SCHEMA | **0.70** | 0.60 | 0.30 | **0.80** |
| NUMERICAL | 0.50 | 0.60 | 0.50 | **0.75** |

---

## 6. Reproduzierbarkeit

### 6.1 Determinismus

- **Random Seed:** 42 (für alle Generatoren und Shuffle)
- **Dataset:** FEVER (dev split)
- **Sample Size:** Konfigurierbar (default: 200)

### 6.2 Ausführung

```bash
# Standard-Evaluation
python evaluation/comprehensive_streaming_evaluation.py \
  --sample-size 200 \
  --seed 42 \
  --conflicts-per-type 20 \
  --compare-baselines

# Taxonomie anzeigen
python evaluation/comprehensive_streaming_evaluation.py --print-taxonomy

# Ohne GPU
python evaluation/comprehensive_streaming_evaluation.py --no-gpu

# Mit LLM (für Implicit Conflicts)
python evaluation/comprehensive_streaming_evaluation.py --llm-model llama3.1:8b
```

### 6.3 Output

```json
{
  "timestamp": "2024-...",
  "seed": 42,
  "conflicts_generated": {
    "factual": 20,
    "temporal": 20,
    ...
  },
  "per_type_metrics": {
    "factual": {"f1_score": 0.85, ...},
    ...
  },
  "hybrid_overall_f1": 0.82,
  "baselines": {...}
}
```

---

## 7. Limitationen

1. **Künstliche Konflikt-Generierung**: Konflikte werden programmatisch generiert, nicht aus echten Dokumenten extrahiert

2. **FEVER-Bias**: FEVER enthält primär Fact Verification, weniger komplexe Reasoning-Aufgaben

3. **Keine echte Temporal Metadata**: Zeitangaben werden synthetisch hinzugefügt

4. **Ontologie-Approximation**: Granularitäts-Hierarchien sind hard-coded, nicht aus echter Ontologie

5. **LLM-Abhängigkeit**: Implicit Conflicts erfordern LLM für optimale Erkennung

---

## 8. Erweiterungsmöglichkeiten

1. **Echte Dokument-Extraktion**: Triples aus unstrukturierten Texten extrahieren

2. **Wikidata-Integration**: Echte Ontologie-Hierarchien nutzen

3. **Temporal Annotations**: YAGO-style Zeitangaben integrieren

4. **Multi-Modal**: Bilder und Text kombinieren

5. **Adversarial Testing**: Gezielte Angriffe auf das System

---

## 9. Fazit

Dieses Evaluationsframework ermöglicht eine **wissenschaftlich fundierte, reproduzierbare und umfassende** Evaluation von Knowledge Graph Konsistenzprüfungssystemen. Durch die systematische Abdeckung aller 10 Konflikt-Kategorien und den Verzicht auf Kardinalitätsregeln wird ein **fairer Vergleich** zwischen verschiedenen Ansätzen gewährleistet.

Der **Hybrid-Ansatz** sollte signifikant besser abschneiden als einzelne Baselines, da er die Stärken von:
- Regelbasierter Validierung (Schema, Source)
- Embedding-basierter Ähnlichkeit (Entity, Granularity)
- NLI-basierter Inferenz (Factual, Negation)
- LLM-basiertem Reasoning (Implicit)

kombiniert.

---

## Bibliographie

Siehe `conflict_taxonomy.py` für die vollständige Bibliographie mit 21 Referenzen.
