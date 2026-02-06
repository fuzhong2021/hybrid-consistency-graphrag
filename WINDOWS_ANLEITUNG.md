# GraphRAG Masterarbeit - Windows Setup mit RTX 5070

## Voraussetzungen

1. **Windows 10/11** (64-bit)
2. **RTX 5070** mit aktuellen NVIDIA-Treibern
3. **Python 3.11+** - https://python.org
4. **Ollama** - https://ollama.com/download/windows
5. **Neo4j** (optional) - https://neo4j.com/download/

---

## Quick Start

### 1. Ollama installieren und Modell laden

```powershell
# Nach Ollama-Installation:
ollama pull llama3.1:8b
```

### 2. Projekt-Ordner öffnen

```powershell
cd C:\masterarbeit-graphrag
```

### 3. Setup ausführen

```
setup_windows.bat
```

### 4. Test starten

```
run_test.bat
```

### 5. Volle Evaluation starten

```
run_evaluation.bat
```

---

## Verfügbare Scripts

| Script | Beschreibung |
|--------|--------------|
| `setup_windows.bat` | Installiert alles automatisch |
| `run_test.bat` | Test mit 3 Beispielen (~12 Min) |
| `run_evaluation.bat` | Volle Evaluation 500 Beispiele (~8-16h) |
| `run_evaluation_background.bat` | Startet Evaluation im Hintergrund |
| `check_progress.bat` | Zeigt aktuellen Fortschritt |
| `stop_evaluation.bat` | Bricht Evaluation ab |

---

## Geschätzte Laufzeiten (RTX 5070)

| Beispiele | Pro Durchlauf | Ablation (2x) |
|-----------|---------------|---------------|
| 3 | ~6 Min | ~12 Min |
| 100 | ~1.5h | ~3h |
| 500 | ~8h | ~16h |

---

## Troubleshooting

### Ollama nicht erreichbar
```powershell
# Prüfen ob Ollama läuft:
curl http://localhost:11434/api/tags

# Ollama manuell starten:
ollama serve
```

### Neo4j nicht erreichbar
```powershell
# Option 1: Neo4j Desktop starten

# Option 2: Ohne Neo4j testen
python scripts/benchmark_evaluation.py --no-neo4j ...
```

### GPU wird nicht genutzt
```powershell
# NVIDIA-Treiber prüfen:
nvidia-smi

# Ollama GPU-Status:
ollama ps
```

### PowerShell ExecutionPolicy
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Ergebnisse

Nach der Evaluation findest du die Ergebnisse in:

```
results/
├── hotpotqa_llama3.1-8b_with_consistency_*.json
├── hotpotqa_llama3.1-8b_no_consistency_*.json
├── evaluation_500.log
└── extrinsic/
    ├── latex_tables.tex
    └── *.png (Visualisierungen)
```

---

## Neo4j Browser

Nach der Evaluation kannst du die Daten im Neo4j Browser ansehen:

1. Öffne http://localhost:7474
2. Login: `neo4j` / `masterarbeit2024`
3. Query: `MATCH (e:Entity)-[r]->(t:Entity) RETURN e, r, t LIMIT 100`
