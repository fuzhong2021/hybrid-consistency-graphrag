#!/usr/bin/env python3
# evaluation/analysis/finalize_six_phase.py
"""
Post-processor für den hochskalierten 6-Phasen-Lauf (Schritt 1).

Liest results/six_phase_n50.json und produziert:
- Wilson-CIs auf Precision/Recall/F1/Accuracy pro Phase
- Vergleichstabelle n=3 (Sanity) vs n=50 (belastbar)
- Markdown-Report results/analysis/six_phase_n50.md

Nutzung:
  python evaluation/analysis/finalize_six_phase.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.analysis.compute_confidence_intervals import (
    analyze_record,
    render_markdown,
)

RESULTS_DIR = _REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis"

OLD = RESULTS_DIR / "hotpotqa_realistic_evaluation.json"   # n=3
NEW = RESULTS_DIR / "six_phase_n50.json"                   # n=50


def extract_phase_records(path: Path, label_prefix: str):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    recs = []
    for phase_key, phase in data.get("phases", {}).items():
        # Einige Phasen haben TP/FP/TN/FN, andere nur acceptance_rate
        if phase.get("true_positives", 0) + phase.get("false_positives", 0) \
           + phase.get("true_negatives", 0) + phase.get("false_negatives", 0) > 0:
            rec = analyze_record(phase, f"{label_prefix}/{phase_key}")
            if not rec.get("skipped"):
                recs.append(rec)
    return recs


def extract_comparison_metrics(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("comparison_metrics", {})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old_recs = extract_phase_records(OLD, "n=3_sanity")
    new_recs = extract_phase_records(NEW, "n=50_scaled")
    old_comp = extract_comparison_metrics(OLD)
    new_comp = extract_comparison_metrics(NEW)

    lines = [
        "# 6-Phasen-Evaluation — n=3 (Sanity) vs n=50 (belastbar)",
        "",
        "Wilson-95 %-CI für Precision/Recall/Accuracy, Bootstrap-95 %-CI für F1.",
        "",
        "## Comparison-Metriken (Konfidenz-Effekte)",
        "",
        "| Metrik | n=3 | n=50 |",
        "|---|---:|---:|",
    ]
    for key in ["missing_source_penalty_effectiveness", "contradiction_detection_f1",
                "cross_question_detection_f1", "fake_source_detection_effectiveness"]:
        old_val = old_comp.get(key, float("nan"))
        new_val = new_comp.get(key, float("nan"))
        lines.append(f"| `{key}` | {old_val:.3f} | {new_val:.3f} |")

    lines += ["", "## Phasen-Metriken mit Konfidenzintervallen", ""]
    if new_recs:
        lines.append(render_markdown(new_recs))
    else:
        lines.append("_noch keine n=50-Daten vorhanden_")

    if old_recs:
        lines += ["", "### Historischer Sanity-Check (n=3)", "", render_markdown(old_recs)]

    (OUTPUT_DIR / "six_phase_comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT_DIR / 'six_phase_comparison.md'}")


if __name__ == "__main__":
    main()
