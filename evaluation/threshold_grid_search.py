#!/usr/bin/env python3
# evaluation/threshold_grid_search.py
"""
Rastersuche über Konfidenz-Schwellenwerte (θ_H, θ_M) auf einem
10%-Validierungsausschnitt von HotpotQA.

Bestimmt die optimale Kombination aus High-Confidence-Threshold (θ_H,
Fast-Path Accept) und Medium-Confidence-Threshold (θ_M, Routing zu
Stufe 3/LLM). Optimierungsziel: F1-Score bei minimalem LLM-Anteil.

Methodik:
  - 10% der HotpotQA-Validierungsdaten (n ≈ 2.928 Triples) werden als
    Kalibrierungsset verwendet, die restlichen 90% als Testset.
  - Für jedes (θ_H, θ_M)-Paar wird das vollständige Konsistenzmodul
    mit den entsprechenden Schwellenwerten ausgeführt.
  - Ergebnis: Tabelle mit F1, Precision, Recall und LLM-Anteil pro
    Schwellenwertpaar.

Nutzung:
  python evaluation/threshold_grid_search.py --gpu --output results/threshold_grid_search.json

Ergebnisse werden in der Thesis in Tabelle 'Sensitivitätsanalyse der
Konfidenz-Schwellenwerte' (Abschnitt Schwellenwert-Kalibrierung) verwendet.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.models.entities import ValidationStatus, Relation
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.graph.memory_repository import InMemoryGraphRepository
from evaluation.multi_dataset_evaluation import MultiDatasetEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GRID = [
    (0.95, 0.75),
    (0.90, 0.70),
    (0.85, 0.65),
    (0.80, 0.60),
]

VALIDATION_FRACTION = 0.10


def run_single_config(
    theta_h: float,
    theta_m: float,
    data: list,
    embedding_model,
    llm_client,
) -> dict:
    config = ConsistencyConfig(
        high_confidence_threshold=theta_h,
        medium_confidence_threshold=theta_m,
        valid_relation_types=[
            "RELATED_TO", "SUPPORTS", "ANSWERS", "HAS_ANSWER", "CONFIRMS",
            "CLAIMS", "REFUTES", "SUPPORTS_CLAIM",
            "GEBOREN_IN", "GESTORBEN_IN", "WOHNT_IN", "ARBEITET_BEI",
            "STUDIERT_AN", "LEITET", "TEIL_VON", "BEFINDET_SICH_IN",
            "VERHEIRATET_MIT", "KIND_VON", "KENNT",
        ],
        cardinality_rules={"HAS_ANSWER": {"max": 1}, "ANSWERS": {"max": 1}},
        llm_model="llama3.1:8b",
    )

    graph_repo = InMemoryGraphRepository()
    orchestrator = ConsistencyOrchestrator(
        config=config,
        graph_repo=graph_repo,
        embedding_model=embedding_model,
        llm_client=llm_client,
        enable_metrics=True,
    )

    baseline_data = [d for d in data if d["ground_truth_accept"]]
    contradiction_data = [d for d in data if not d["ground_truth_accept"]]

    tp, fp, tn, fn = 0, 0, 0, 0
    llm_calls = 0
    total = 0

    for item in baseline_data:
        triple = item["triple"]
        result = orchestrator.process(triple)
        total += 1
        if result.validation_status == ValidationStatus.ACCEPTED:
            tn += 1
            try:
                graph_repo.create_entity(triple.subject)
                graph_repo.create_entity(triple.object)
                rel = Relation(
                    source_id=triple.subject.id,
                    target_id=triple.object.id,
                    relation_type=triple.predicate,
                    confidence=triple.extraction_confidence,
                )
                graph_repo.create_relation(rel)
            except Exception:
                pass
        else:
            fp += 1
        if result.validation_history:
            last_stage = result.validation_history[-1].get("stage", "")
            if "llm" in last_stage.lower() or "arbitr" in last_stage.lower():
                llm_calls += 1

    for item in contradiction_data:
        triple = item["triple"]
        result = orchestrator.process(triple)
        total += 1
        if result.validation_status == ValidationStatus.REJECTED:
            tp += 1
        else:
            fn += 1
        if result.validation_history:
            last_stage = result.validation_history[-1].get("stage", "")
            if "llm" in last_stage.lower() or "arbitr" in last_stage.lower():
                llm_calls += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    llm_share = llm_calls / total if total else 0.0

    return {
        "theta_h": theta_h,
        "theta_m": theta_m,
        "f1": round(f1, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "llm_share": round(llm_share, 3),
        "n": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "llm_calls": llm_calls,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Schwellenwert-Rastersuche")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--output", default="results/threshold_grid_search.json")
    ap.add_argument("--sample-size", type=int, default=3000,
                    help="HotpotQA-Beispiele laden (10%% davon = Validierungsset)")
    args = ap.parse_args()

    # Embedding-Modell
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        if args.gpu:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = "cpu"
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(f"Embedding-Modell auf {device}")
    except Exception as e:
        logger.warning(f"Kein Embedding-Modell: {e}")

    # LLM
    llm_client = None
    try:
        from src.llm.ollama_client import OllamaClient
        llm_client = OllamaClient(model="llama3.1:8b")
        logger.info("LLM-Client initialisiert")
    except Exception as e:
        logger.warning(f"Kein LLM: {e}")

    # Daten laden
    evaluator = MultiDatasetEvaluator(
        embedding_model=embedding_model, llm_client=llm_client)
    all_data = evaluator.load_dataset("hotpotqa", args.sample_size)
    logger.info(f"Geladen: {len(all_data)} Triples")

    # 10% Validierungssplit (deterministisch: erste 10%)
    split_idx = int(len(all_data) * VALIDATION_FRACTION)
    val_data = all_data[:split_idx]
    logger.info(f"Validierungsset: {len(val_data)} Triples ({VALIDATION_FRACTION:.0%})")

    # Grid Search
    results = []
    for theta_h, theta_m in GRID:
        logger.info(f"\n=== θ_H={theta_h}, θ_M={theta_m} ===")
        start = time.time()
        r = run_single_config(theta_h, theta_m, val_data, embedding_model, llm_client)
        r["wallclock_s"] = round(time.time() - start, 1)
        results.append(r)
        logger.info(f"  F1={r['f1']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                     f"LLM={r['llm_share']:.1%}  ({r['wallclock_s']}s)")

    # Beste Konfiguration
    best = max(results, key=lambda x: x["f1"])
    logger.info(f"\nBeste Konfiguration: θ_H={best['theta_h']}, θ_M={best['theta_m']} "
                f"(F1={best['f1']:.3f}, LLM={best['llm_share']:.1%})")

    # Speichern
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation_set_size": len(val_data),
        "validation_fraction": VALIDATION_FRACTION,
        "grid": results,
        "best": best,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    logger.info(f"Ergebnisse: {args.output}")

    # Tabelle für Thesis
    print("\n% LaTeX-Tabelle für Thesis:")
    print("% \\begin{tabular}{ccccr}")
    for r in results:
        bold = r == best
        fmt = "\\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.3f} & \\textbf{%.3f} & \\textbf{%d\\,\\%%}" if bold \
            else "%.2f & %.2f & %.3f & %.3f & %d\\,\\%%"
        print(f"        {fmt % (r['theta_h'], r['theta_m'], r['f1'], r['precision'], round(r['llm_share']*100))} \\\\")


if __name__ == "__main__":
    main()
