#!/usr/bin/env python3
# evaluation/run_source_verification_comparison.py
"""
Vergleich: Embedding-basierte vs. NLI-basierte Source Verification.

Läuft die 6-Phasen-Evaluation zweimal:
  1. method="embedding" (Cosine-Similarity, neue Schwellenwerte 0.3/0.5/0.7)
  2. method="nli" (DeBERTa Entailment/Contradiction)

Fokus: Phase 6 (Fake Source Attack) — dort zeigt sich der Unterschied.

Nutzung:
  python evaluation/run_source_verification_comparison.py --sample-size 50 --gpu
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "hotpotqa_realistic_evaluation.py"


def run_variant(label: str, method: str, sample_size: int, gpu: bool) -> Path:
    out = Path(f"results/source_verification_{label}.json")
    cmd = [
        sys.executable, str(SCRIPT),
        "--sample-size", str(sample_size),
        "--output", str(out),
    ]
    if gpu:
        cmd.append("--gpu")

    # Source Verification Methode via Umgebungsvariable durchreichen
    env = {**__import__("os").environ, "SOURCE_VERIFICATION_METHOD": method}
    print(f"\n=== Source Verification: {label} (method={method}) ===")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    return out


def compare(path_emb: Path, path_nli: Path) -> None:
    d_emb = json.loads(path_emb.read_text())
    d_nli = json.loads(path_nli.read_text())

    print("\n" + "=" * 70)
    print("SOURCE VERIFICATION VERGLEICH: Embedding vs. NLI")
    print("=" * 70)

    for phase_key in ["phase6_fake_source_attack", "phase4_distractor_contradictions",
                       "phase5_cross_question_confusion", "phase1_baseline"]:
        p_emb = d_emb.get("phases", {}).get(phase_key, {})
        p_nli = d_nli.get("phases", {}).get(phase_key, {})
        if not p_emb or not p_nli:
            continue
        name = p_emb.get("phase_name", phase_key)
        f1_e = p_emb.get("f1_score", 0)
        f1_n = p_nli.get("f1_score", 0)
        r_e = p_emb.get("recall", 0)
        r_n = p_nli.get("recall", 0)
        print(f"\n{name}:")
        print(f"  Embedding: F1={f1_e:.3f}  Recall={r_e:.3f}")
        print(f"  NLI:       F1={f1_n:.3f}  Recall={r_n:.3f}")
        print(f"  Δ F1: {f1_n - f1_e:+.3f}")

    # Comparison Metrics
    print("\nComparison Metrics:")
    for key in ["fake_source_detection_effectiveness", "missing_source_penalty_effectiveness",
                 "contradiction_detection_f1", "cross_question_detection_f1"]:
        v_e = d_emb.get("comparison_metrics", {}).get(key, 0)
        v_n = d_nli.get("comparison_metrics", {}).get(key, 0)
        print(f"  {key}: Embedding={v_e:.3f}  NLI={v_n:.3f}  Δ={v_n - v_e:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    # Embedding-Lauf existiert möglicherweise schon
    path_emb = Path("results/six_phase_n50_threshold_fix.json")
    if not path_emb.exists():
        path_emb = run_variant("embedding", "embedding", args.sample_size, args.gpu)

    path_nli = run_variant("nli", "nli", args.sample_size, args.gpu)

    compare(path_emb, path_nli)
    print(f"\nErgebnisse: {path_emb} / {path_nli}")


if __name__ == "__main__":
    main()
