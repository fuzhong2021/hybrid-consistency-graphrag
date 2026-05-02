#!/usr/bin/env python3
# evaluation/analysis/mcnemar_and_ece.py
"""
McNemar-Test (Phase 1.3) und Expected-Calibration-Error (Phase 1.4)
auf per-example JSONL-Dateien aus instrumentierten Eval-Läufen.

Voraussetzung: der Patch in evaluation/multi_dataset_evaluation.py
schreibt `<per_example_dir>/<dataset>.jsonl` mit Feldern:
  triple_id, dataset, y_true, y_pred, stage_decided, confidence

Vergleichsstrategie: Jede per-example JSONL landet in
results/per_example/<config>/<dataset>.jsonl (config = with_nli /
without_nli / …). Dieses Skript nimmt zwei Konfigurationen und liefert
McNemar-p-Werte + ECE pro Datensatz.

Nutzung:
  python evaluation/analysis/mcnemar_and_ece.py \
      --config-a results/ablation_nli/with_nli/per_example \
      --config-b results/ablation_nli/without_nli/per_example \
      --output   results/analysis/nli_ablation.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Repo-root in sys.path, damit `evaluation.metrics...` auch beim direkten
# Skriptaufruf außerhalb von `python -m` aufgelöst wird
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from evaluation.metrics.calibration import compute_calibration_metrics
from evaluation.metrics.significance import mcnemar_test


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def align(a: List[Dict], b: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Verknüpft Per-Example-Records nach triple_id."""
    a_by_id = {r["triple_id"]: r for r in a}
    b_by_id = {r["triple_id"]: r for r in b}
    common = sorted(set(a_by_id) & set(b_by_id))
    return [a_by_id[i] for i in common], [b_by_id[i] for i in common]


def ece_section(records: List[Dict]) -> Dict:
    """Ruft evaluation.metrics.calibration.compute_calibration_metrics mit
    korrekter Argument-Reihenfolge auf (y_true, y_prob, n_bins).
    Liefert Keys: ECE, MCE, Brier, n_samples, ECE_interpretation.
    """
    if not records:
        return {"n": 0}
    y_true = [r["y_true"] == r["y_pred"] for r in records]   # korrekt ja/nein
    y_prob = [float(r["confidence"]) for r in records]
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)
    return {
        "n": int(metrics.get("n_samples", len(records))),
        "ece": float(metrics["ECE"]),
        "mce": float(metrics["MCE"]),
        "brier": float(metrics["Brier"]),
        "interpretation": metrics.get("ECE_interpretation", ""),
    }


def mcnemar_section(a_records: List[Dict], b_records: List[Dict]) -> Dict:
    a_correct = [r["y_true"] == r["y_pred"] for r in a_records]
    b_correct = [r["y_true"] == r["y_pred"] for r in b_records]
    gt = [r["y_true"] == 1 for r in a_records]
    return mcnemar_test(a_correct, b_correct, gt, exact=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", required=True, help="Verzeichnis mit <ds>.jsonl (Konfig A)")
    ap.add_argument("--config-b", required=True, help="Verzeichnis mit <ds>.jsonl (Konfig B)")
    ap.add_argument("--output", default="results/analysis/mcnemar_ece.md")
    args = ap.parse_args()

    dir_a = Path(args.config_a)
    dir_b = Path(args.config_b)
    datasets = sorted({p.stem for p in dir_a.glob("*.jsonl")} & {p.stem for p in dir_b.glob("*.jsonl")})

    md_lines = [
        f"# McNemar + ECE: `{dir_a.name}` vs `{dir_b.name}`",
        "",
        "## McNemar-Test (Phase 1.3)",
        "",
        "Nullhypothese: beide Konfigurationen haben die gleiche Fehlerrate.",
        "",
        "| Dataset | n (aligned) | b (A richtig, B falsch) | c (A falsch, B richtig) | Statistik | p-Wert | sig. α=0.05 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    ece_lines = [
        "",
        "## Expected Calibration Error (Phase 1.4)",
        "",
        "Referenz: Guo et al. (2017). Niedriger = besser kalibriert.",
        "",
        "| Dataset | Konfig | n | ECE | MCE | Brier | Qualität |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    all_raw: Dict[str, Dict] = {}

    for ds in datasets:
        a = load_jsonl(dir_a / f"{ds}.jsonl")
        b = load_jsonl(dir_b / f"{ds}.jsonl")
        a_aligned, b_aligned = align(a, b)
        if not a_aligned:
            md_lines.append(f"| {ds} | 0 | – | – | – | – | – |")
            continue

        mc = mcnemar_section(a_aligned, b_aligned)
        md_lines.append(
            f"| {ds} | {len(a_aligned):,} | {mc.get('b', 0)} | {mc.get('c', 0)} | "
            f"{mc.get('statistic', float('nan')):.3f} | "
            f"{mc.get('p_value', float('nan')):.4f} | "
            f"{'✓' if mc.get('significant_0.05') else '–'} |"
        )

        ece_a = ece_section(a_aligned)
        ece_b = ece_section(b_aligned)
        for label, ece in [(dir_a.name, ece_a), (dir_b.name, ece_b)]:
            ece_lines.append(
                f"| {ds} | `{label}` | {ece.get('n', 0):,} | "
                f"{ece.get('ece', float('nan')):.4f} | "
                f"{ece.get('mce', float('nan')):.4f} | "
                f"{ece.get('brier', float('nan')):.4f} | "
                f"{ece.get('interpretation', '')} |"
            )

        all_raw[ds] = {"mcnemar": mc, "ece_a": ece_a, "ece_b": ece_b}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(md_lines + ece_lines) + "\n")
    raw_path = Path(args.output).with_suffix(".json")
    raw_path.write_text(json.dumps(all_raw, indent=2, default=str))
    print(f"Wrote {args.output} and {raw_path}")


if __name__ == "__main__":
    main()
