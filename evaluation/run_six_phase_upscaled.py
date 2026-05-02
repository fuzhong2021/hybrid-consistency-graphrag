#!/usr/bin/env python3
# evaluation/run_six_phase_upscaled.py
"""
Phase 1.1: 6-Phasen-Evaluation hochskaliert.

Wrapper um hotpotqa_realistic_evaluation.py, der die Stichprobengröße von
3 auf >= 50 hochzieht und die Ergebnisse in einer eigenen Datei ablegt,
damit die ursprüngliche Datei (n=3) als Sanity-Check erhalten bleibt.

Nutzung (Laufzeit: stark abhängig von Embedding/LLM-Verfügbarkeit):
  python evaluation/run_six_phase_upscaled.py --sample-size 50 --seed 42
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "hotpotqa_realistic_evaluation.py"


def main() -> None:
    # Hinweis: hotpotqa_realistic_evaluation.py samplet deterministisch per
    # Index (keine randomisierte Stichprobe), daher ist kein --seed nötig —
    # Reproduzierbarkeit ergibt sich aus sample-size + Split.
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--output", default="results/six_phase_n50.json")
    args = ap.parse_args()

    cmd = [
        sys.executable, str(SCRIPT),
        "--sample-size", str(args.sample_size),
        "--output", args.output,
    ]
    if args.gpu:
        cmd.append("--gpu")
    print("Run: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone. Result: {args.output}")
    print("Folgemessungen: 95%-CI via evaluation/analysis/compute_confidence_intervals.py")


if __name__ == "__main__":
    main()
