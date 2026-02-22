#!/usr/bin/env python3
"""
Hybrid Evaluation Framework.

Kombiniert zwei wissenschaftlich komplementäre Ansätze:

TEIL A: Kontrolliertes Experiment (Synthetische Konflikte)
- Perfekte Ground Truth durch generierte Mutationen
- Misst P/R/F1 pro Konflikt-Kategorie
- Beweist: "Das Modul erkennt Fehler korrekt"

TEIL B: Realistisches Experiment (Supporting Facts vs. All)
- Nutzt HotpotQA's supporting_facts Labels als Proxy
- Misst Signal Preservation und Noise Rejection
- Beweist: "Das Modul verbessert Graph-Qualität in der Praxis"

Verwendung:
  # Beide Teile (Standard)
  python scripts/run_hybrid_evaluation.py --sample-size 100

  # Nur Teil A (Kontrolliert)
  python scripts/run_hybrid_evaluation.py --part-a-only --sample-size 50

  # Nur Teil B (Realistisch)
  python scripts/run_hybrid_evaluation.py --part-b-only --sample-size 50

  # Mit LLM
  python scripts/run_hybrid_evaluation.py --with-llm --sample-size 50
"""

import sys
from pathlib import Path

# Projektroot zum Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
from datetime import datetime

from src.consistency.base import ConsistencyConfig
from src.evaluation.hybrid_evaluator import (
    HybridEvaluator,
    HybridEvaluationResult,
    generate_latex_tables,
)
from src.utils.gpu_utils import print_gpu_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_consistency_config() -> ConsistencyConfig:
    """Standard ConsistencyConfig für Evaluation."""
    return ConsistencyConfig(
        high_confidence_threshold=0.9,
        medium_confidence_threshold=0.7,
        similarity_threshold=0.85,
        valid_entity_types=[
            "Person", "Organisation", "Ort", "Ereignis", "Konzept",
        ],
        valid_relation_types=[
            "WOHNT_IN", "ARBEITET_BEI", "KENNT", "BETEILIGT_AN",
            "BEFINDET_SICH_IN", "HAT_BEZIEHUNG_ZU", "TEIL_VON",
            "GEBOREN_IN", "GESTORBEN_IN", "GRUENDETE", "MITGLIED_VON",
            "REGIE_BEI", "VERHEIRATET_MIT", "VERWANDT_MIT", "SPIELT_FUER",
            "ASSOZIIERT_MIT", "VERBUNDEN_MIT", "TEILNAHME_AN", "RELATED_TO",
            "STUDIERT_AN", "ERHIELT", "ENTWICKELTE", "SCHRIEB",
        ],
        cardinality_rules={
            "GEBOREN_IN": {"max": 1},
            "GESTORBEN_IN": {"max": 1},
        },
    )


def create_llm_client(model: str = "llama3.1:8b"):
    """Create Ollama client if available."""
    try:
        from src.llm.ollama_client import OllamaClient
        return OllamaClient(model=model)
    except Exception as e:
        logger.warning(f"Could not create Ollama client: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Evaluation: Kontrolliert + Realistisch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard (beide Teile)
  python scripts/run_hybrid_evaluation.py --sample-size 100

  # Nur kontrolliertes Experiment (Teil A)
  python scripts/run_hybrid_evaluation.py --part-a-only --sample-size 50

  # Nur realistisches Experiment (Teil B)
  python scripts/run_hybrid_evaluation.py --part-b-only --sample-size 50

  # Mit LLM für Validierung
  python scripts/run_hybrid_evaluation.py --with-llm --sample-size 50

  # Schnelltest
  python scripts/run_hybrid_evaluation.py --sample-size 10
        """,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Anzahl der QA-Beispiele (default: 100)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["hotpotqa", "musique"],
        default="hotpotqa",
        help="Benchmark-Dataset (default: hotpotqa)",
    )
    parser.add_argument(
        "--part-a-only",
        action="store_true",
        help="Nur Teil A (kontrolliertes Experiment) ausführen",
    )
    parser.add_argument(
        "--part-b-only",
        action="store_true",
        help="Nur Teil B (realistisches Experiment) ausführen",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="LLM für Stage 3 aktivieren",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama-Modell (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed für Reproduzierbarkeit (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hybrid",
        help="Ausgabeverzeichnis (default: results/hybrid)",
    )

    args = parser.parse_args()

    # Bestimme welche Teile ausgeführt werden
    run_part_a = not args.part_b_only
    run_part_b = not args.part_a_only

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("\n" + "=" * 70)
    print("HYBRID EVALUATION FRAMEWORK")
    print("=" * 70)
    print(f"Sample Size:    {args.sample_size}")
    print(f"Benchmark:      {args.benchmark}")
    print(f"Part A:         {'Yes' if run_part_a else 'No'} (Kontrolliert)")
    print(f"Part B:         {'Yes' if run_part_b else 'No'} (Realistisch)")
    print(f"With LLM:       {args.with_llm}")
    print(f"Random Seed:    {args.seed}")
    print(f"Output:         {args.output_dir}")
    print("=" * 70)

    # LLM client if requested
    llm_client = None
    if args.with_llm:
        print_gpu_status()
        llm_client = create_llm_client(args.llm_model)
        if not llm_client:
            logger.warning("LLM client not available - Stage 3 disabled")

    # Create evaluator
    evaluator = HybridEvaluator(
        consistency_config=get_consistency_config(),
        sample_size=args.sample_size,
        benchmark=args.benchmark,
        llm_client=llm_client,
        seed=args.seed,
    )

    # Run evaluation
    result = evaluator.run(run_part_a=run_part_a, run_part_b=run_part_b)

    # Save results
    result_dict = result.to_dict()
    result_dict["timestamp"] = datetime.now().isoformat()
    result_dict["args"] = vars(args)

    # JSON
    json_path = output_dir / "hybrid_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {json_path}")

    # LaTeX
    latex_path = output_dir / "hybrid_tables.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(generate_latex_tables(result))
    logger.info(f"LaTeX tables saved: {latex_path}")

    # Print generated files
    print(f"\nGenerated files in {output_dir}/:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
