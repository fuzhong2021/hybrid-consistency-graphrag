#!/usr/bin/env python3
# evaluation/analysis/compute_confidence_intervals.py
"""
Wilson- und Bootstrap-Konfidenzintervalle für aggregierte Result-JSONs.

Liest die TP/FP/TN/FN-Counts aus den vorhandenen Result-Dateien und berechnet:
- Wilson-95%-CIs für Precision, Recall, Accuracy (geschlossene Form)
- Bootstrap-95%-CIs für F1 (1000 Resamples aus synthetisierten per-example Labels)

Phase 1.2 des Evaluationsplans. Belegt jede Hauptkennzahl mit einem CI,
ohne dass ein neuer Evaluationslauf nötig wäre.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis"

BOOTSTRAP_RESAMPLES = 1000
RNG_SEED = 42


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson-Score-Konfidenzintervall für Binomialanteil (Wilson 1927)."""
    if n == 0:
        return (0.0, 0.0)
    if confidence == 0.95:
        z = 1.959963984540054
    else:
        from scipy.stats import norm  # type: ignore
        z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def synthesize_labels(tp: int, fp: int, tn: int, fn: int) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetisiert per-example y_true/y_pred Arrays aus aggregierten Counts.

    Bootstrap auf den synthetisierten Arrays liefert mathematisch identische
    Resampling-Verteilungen wie auf den Originalbeispielen, da jede Metrik
    eine reine Funktion der Konfusionsmatrix ist.
    """
    y_true = np.concatenate([
        np.ones(tp, dtype=np.int8),
        np.zeros(fp, dtype=np.int8),
        np.zeros(tn, dtype=np.int8),
        np.ones(fn, dtype=np.int8),
    ])
    y_pred = np.concatenate([
        np.ones(tp, dtype=np.int8),
        np.ones(fp, dtype=np.int8),
        np.zeros(tn, dtype=np.int8),
        np.zeros(fn, dtype=np.int8),
    ])
    return y_true, y_pred


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """Percentile-Bootstrap für eine Metrik. Gibt (point, lo, hi) zurück."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = metrics_from_arrays(y_true, y_pred)[metric]
    samples = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = metrics_from_arrays(y_true[idx], y_pred[idx])[metric]
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return (point, float(lo), float(hi))


def analyze_record(record: Dict, label: str) -> Dict:
    tp = int(record.get("true_positives", 0))
    fp = int(record.get("false_positives", 0))
    tn = int(record.get("true_negatives", 0))
    fn = int(record.get("false_negatives", 0))
    n = tp + fp + tn + fn
    if n == 0:
        return {"label": label, "n": 0, "skipped": True}

    y_true, y_pred = synthesize_labels(tp, fp, tn, fn)
    rng = np.random.default_rng(RNG_SEED)

    p_lo, p_hi = wilson_ci(tp, tp + fp) if (tp + fp) else (0.0, 0.0)
    r_lo, r_hi = wilson_ci(tp, tp + fn) if (tp + fn) else (0.0, 0.0)
    a_lo, a_hi = wilson_ci(tp + tn, n)
    f1_point, f1_lo, f1_hi = bootstrap_metric_ci(y_true, y_pred, "f1", rng=rng)

    return {
        "label": label,
        "n": n,
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": {"point": tp / (tp + fp) if (tp + fp) else 0.0, "ci95": [p_lo, p_hi], "method": "wilson"},
        "recall": {"point": tp / (tp + fn) if (tp + fn) else 0.0, "ci95": [r_lo, r_hi], "method": "wilson"},
        "accuracy": {"point": (tp + tn) / n, "ci95": [a_lo, a_hi], "method": "wilson"},
        "f1": {"point": f1_point, "ci95": [f1_lo, f1_hi], "method": f"bootstrap_n={BOOTSTRAP_RESAMPLES}"},
    }


def collect_records() -> List[Dict]:
    """Sammelt alle Aggregat-Datensätze aus den Result-JSONs."""
    records: List[Dict] = []

    # Multi-dataset (Hauptergebnisse, mit LLM)
    path = RESULTS_DIR / "multi_dataset_evaluation.json"
    if path.exists():
        data = json.loads(path.read_text())
        for ds in data.get("datasets", []):
            records.append(analyze_record(ds, f"multi_dataset/with_llm/{ds['dataset_name']}"))

    # Full evaluation mit LLM
    path = RESULTS_DIR / "full_evaluation_with_llm.json"
    if path.exists():
        data = json.loads(path.read_text())
        for ds in data.get("datasets", []):
            records.append(analyze_record(ds, f"full_eval/with_llm/{ds['dataset_name']}"))

    # Full evaluation ohne LLM (FEVER-Vergleich!)
    path = RESULTS_DIR / "full_evaluation_no_llm.json"
    if path.exists():
        data = json.loads(path.read_text())
        for ds in data.get("datasets", []):
            records.append(analyze_record(ds, f"full_eval/no_llm/{ds['dataset_name']}"))

    # Baseline-Comparison
    path = RESULTS_DIR / "baseline_comparison.json"
    if path.exists():
        data = json.loads(path.read_text())
        for bl in data.get("baselines", []):
            records.append(analyze_record(bl, f"baselines/{bl['baseline_name']}"))
        if "full_system" in data:
            records.append(analyze_record(data["full_system"], "baselines/full_system"))

    # MuSiQue solo
    path = RESULTS_DIR / "musique_evaluation.json"
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "true_positives" in data:
            records.append(analyze_record(data, "musique_solo"))
        elif isinstance(data, dict) and "datasets" in data:
            for ds in data["datasets"]:
                records.append(analyze_record(ds, f"musique_solo/{ds.get('dataset_name', 'musique')}"))

    return [r for r in records if not r.get("skipped")]


def render_markdown(records: List[Dict]) -> str:
    lines = [
        "# Konfidenzintervalle (Phase 1.2)",
        "",
        f"Generiert aus aggregierten TP/FP/TN/FN-Counts mit Wilson-Score (Precision/Recall/Accuracy) "
        f"und Percentile-Bootstrap (F1, n={BOOTSTRAP_RESAMPLES}, Seed={RNG_SEED}). "
        "95 %-CIs.",
        "",
        "| Konfiguration | n | Precision (95 % CI) | Recall (95 % CI) | F1 (95 % CI) | Accuracy (95 % CI) |",
        "|---|---:|---|---|---|---|",
    ]
    for r in records:
        def fmt(metric: Dict) -> str:
            lo, hi = metric["ci95"]
            return f"{metric['point']:.3f} [{lo:.3f}, {hi:.3f}]"
        lines.append(
            f"| `{r['label']}` | {r['n']:,} | {fmt(r['precision'])} | "
            f"{fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['accuracy'])} |"
        )
    lines.append("")
    lines.append(
        "> **Methodik:** Da nur aggregierte Konfusionsmatrix-Counts vorliegen, werden "
        "per-example Labels aus den Counts synthetisiert (TP→(1,1), FP→(0,1), TN→(0,0), FN→(1,0)). "
        "Bootstrap-Resampling auf diesem synthetisierten Array liefert mathematisch identische "
        "Verteilungen wie Resampling auf den Originalbeispielen, da F1/Precision/Recall/Accuracy "
        "reine Funktionen der Konfusionsmatrix sind."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    (OUTPUT_DIR / "confidence_intervals.json").write_text(json.dumps(records, indent=2))
    (OUTPUT_DIR / "confidence_intervals.md").write_text(render_markdown(records))
    print(f"Wrote CIs for {len(records)} configurations to {OUTPUT_DIR}/confidence_intervals.{{json,md}}")


if __name__ == "__main__":
    main()
