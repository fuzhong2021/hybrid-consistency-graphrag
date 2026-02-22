#!/bin/bash
# Vollständiges Setup und Evaluation für RTX 2070 (oder CPU)
# Führt die 6-Phasen Evaluation des Konsistenzmoduls durch

set -e

echo "=============================================="
echo "  Hybrides Konsistenzmodul - Setup & Evaluation"
echo "=============================================="
echo ""

# Projektverzeichnis
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

echo "Projektverzeichnis: $PROJECT_DIR"
echo ""

# =============================================================================
# 1. GPU-Check
# =============================================================================
echo "=== 1. GPU-Check ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    GPU_AVAILABLE=true
else
    echo "nvidia-smi nicht gefunden. CPU-Modus wird verwendet."
    GPU_AVAILABLE=false
fi
echo ""

# =============================================================================
# 2. Python-Umgebung
# =============================================================================
echo "=== 2. Python-Umgebung ==="

if [ ! -d "venv" ]; then
    echo "Erstelle virtuelle Umgebung..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo ""

# =============================================================================
# 3. Abhängigkeiten
# =============================================================================
echo "=== 3. Abhängigkeiten installieren ==="
pip install -q -r requirements.txt

# Stelle sicher dass sentence-transformers installiert ist
pip install -q sentence-transformers
echo "Abhängigkeiten installiert."
echo ""

# =============================================================================
# 4. Ollama prüfen
# =============================================================================
echo "=== 4. Ollama prüfen ==="

if ! command -v ollama &> /dev/null; then
    echo "WARNUNG: Ollama nicht installiert."
    echo "  Installiere mit: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  LLM-Stufe wird deaktiviert."
    LLM_AVAILABLE=false
else
    # Prüfe ob Ollama läuft
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Starte Ollama..."
        ollama serve &
        sleep 5
    fi

    # Prüfe ob Modell verfügbar
    if ! ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        echo "Lade llama3.1:8b herunter (~4.7GB)..."
        ollama pull llama3.1:8b
    fi

    echo "Ollama bereit mit llama3.1:8b"
    LLM_AVAILABLE=true
fi
echo ""

# =============================================================================
# 5. Embedding-Modell prüfen
# =============================================================================
echo "=== 5. Embedding-Modell prüfen ==="
python -c "
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')
print(f'Embedding-Modell geladen auf: {model.device}')

if torch.cuda.is_available():
    print(f'GPU verfügbar: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('GPU nicht verfügbar - CPU wird verwendet')
"
echo ""

# =============================================================================
# 6. Evaluation starten
# =============================================================================
echo "=== 6. Evaluation starten ==="
echo "Starte 6-Phasen Evaluation..."
echo ""

SAMPLE_SIZE=${1:-50}

if [ "$LLM_AVAILABLE" = true ]; then
    echo "Mit LLM-Stufe (llama3.1:8b)"
    python evaluation/hotpotqa_realistic_evaluation.py --sample-size $SAMPLE_SIZE
else
    echo "Ohne LLM-Stufe (--no-llm)"
    python evaluation/hotpotqa_realistic_evaluation.py --sample-size $SAMPLE_SIZE --no-llm
fi

echo ""

# =============================================================================
# 7. Ergebnisse
# =============================================================================
echo "=== 7. Ergebnisse ==="
echo ""

if [ -f "results/hotpotqa_realistic_evaluation.json" ]; then
    echo "Ergebnisse gespeichert: results/hotpotqa_realistic_evaluation.json"
    echo ""
    echo "Zusammenfassung:"
    python -c "
import json
with open('results/hotpotqa_realistic_evaluation.json') as f:
    r = json.load(f)

print(f'  Sample Size: {r[\"sample_size\"]}')
print(f'  Missing Source Penalty: {r[\"comparison_metrics\"][\"missing_source_penalty_effectiveness\"]*100:.1f}%')
print(f'  Contradiction Detection F1: {r[\"comparison_metrics\"][\"contradiction_detection_f1\"]*100:.1f}%')
print(f'  Cross-Question F1: {r[\"comparison_metrics\"][\"cross_question_detection_f1\"]*100:.1f}%')
print(f'  Fake Source Detection: {r[\"comparison_metrics\"].get(\"fake_source_detection_effectiveness\", 0)*100:.1f}%')
"
fi

echo ""
echo "=============================================="
echo "  Evaluation abgeschlossen!"
echo "=============================================="
