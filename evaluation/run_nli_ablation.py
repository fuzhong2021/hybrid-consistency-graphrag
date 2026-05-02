#!/usr/bin/env python3
# evaluation/run_nli_ablation.py
"""
Phase 2.3: NLI-Ablation.

Läuft die Multi-Dataset-Evaluation zweimal (mit/ohne NLI) und legt die
Per-Example-JSONLs + Result-JSONs in getrennten Verzeichnissen ab,
sodass evaluation/analysis/mcnemar_and_ece.py direkt den paarweisen
McNemar-Vergleich rechnen kann.

Wichtig: die bisherigen Logs `full_evaluation_with_llm.log` und
`full_evaluation_no_llm.log` hatten BEIDE NLI aktiviert (via `--nli`).
Diese Ablation trennt NLI sauber vom Stage-3-LLM-Effekt.

Nutzung (Laufzeit: Stunden):
  python evaluation/run_nli_ablation.py --sample-size 2000 --gpu

Ausgabe:
  results/ablation_nli/with_nli/multi_dataset_evaluation.json
  results/ablation_nli/with_nli/per_example/<dataset>.jsonl
  results/ablation_nli/without_nli/multi_dataset_evaluation.json
  results/ablation_nli/without_nli/per_example/<dataset>.jsonl
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "multi_dataset_evaluation.py"


def run_variant(variant: str, enable_nli: bool, sample_size: int, gpu: bool, no_llm: bool) -> None:
    out_dir = Path("results/ablation_nli") / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    per_example_dir = out_dir / "per_example"
    per_example_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT),
        "--sample-size", str(sample_size),
        "--output", str(out_dir / "multi_dataset_evaluation.json"),
        "--figures-dir", str(out_dir / "figures"),
    ]
    # NLI ist Default-on; wir geben nur explizit --no-nli für die Ablations-Variante
    if not enable_nli:
        cmd.append("--no-nli")
    if gpu:
        cmd.append("--gpu")
    if no_llm:
        cmd.append("--no-llm")

    env = {"PER_EXAMPLE_DIR": str(per_example_dir)}
    print(f"\n== Ablation variant: {variant} ==")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, env={**__import__("os").environ, **env}, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="Beide Varianten ohne Stage-3-LLM (rein Rules+Embedding±NLI)")
    args = ap.parse_args()

    # Hinweis: der Patch in multi_dataset_evaluation.py schreibt per-example
    # nach results/per_example/<dataset>.jsonl. Für Ablations-Trennung setzen
    # wir per-example-Dir via self.per_example_dir. Diese Variante erfordert,
    # dass main() in multi_dataset_evaluation.py die Env-Variable ausliest –
    # siehe README-Hinweis unten oder direkte Patch-Ergänzung.
    run_variant("with_nli", enable_nli=True, sample_size=args.sample_size,
                gpu=args.gpu, no_llm=args.no_llm)
    run_variant("without_nli", enable_nli=False, sample_size=args.sample_size,
                gpu=args.gpu, no_llm=args.no_llm)

    print("\nAblation abgeschlossen.")
    print("Nächster Schritt:")
    print("  python evaluation/analysis/mcnemar_and_ece.py \\")
    print("      --config-a results/ablation_nli/with_nli/per_example \\")
    print("      --config-b results/ablation_nli/without_nli/per_example \\")
    print("      --output   results/analysis/nli_ablation.md")


if __name__ == "__main__":
    main()
