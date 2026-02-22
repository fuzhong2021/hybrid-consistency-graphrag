#!/usr/bin/env python3
"""
Seed + Update Evaluation — Realistische Konsistenzmodul-Evaluation.

Simuliert ein realistisches Szenario:
1. SEED: Initialer Knowledge Graph wird aufgebaut (ohne Prüfung)
2. UPDATE: Neue Dokumente werden mit voller Konsistenzprüfung hinzugefügt

Verwendung:
  # Standard (50/50 Split)
  python scripts/run_seed_update_evaluation.py --total-examples 100

  # Mehr Seed-Daten (70% seed, 30% update)
  python scripts/run_seed_update_evaluation.py --seed-ratio 0.7 --total-examples 200

  # Mit LLM-Extraktion
  python scripts/run_seed_update_evaluation.py --with-llm --total-examples 50
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
from src.evaluation.seed_update_evaluator import (
    SeedUpdateConfig,
    SeedUpdateEvaluator,
    DataSource,
    generate_latex_table,
)
from src.evaluation.comprehensive import EnhancedTripleExtractor
from src.utils.gpu_utils import print_gpu_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_consistency_config() -> ConsistencyConfig:
    """Get ConsistencyConfig for evaluation."""
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
    """Create Ollama client."""
    try:
        from src.llm.ollama_client import OllamaClient
        return OllamaClient(model=model)
    except Exception as e:
        logger.warning(f"Could not create Ollama client: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Seed + Update Evaluation des Konsistenzmoduls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard (50/50 Split, 100 Beispiele)
  python scripts/run_seed_update_evaluation.py

  # Mehr Seed-Daten (70/30 Split)
  python scripts/run_seed_update_evaluation.py --seed-ratio 0.7 --total-examples 200

  # Mit LLM-Extraktion
  python scripts/run_seed_update_evaluation.py --with-llm --total-examples 50

  # Vollständig (große Evaluation)
  python scripts/run_seed_update_evaluation.py --seed-ratio 0.6 --total-examples 500 --with-llm
        """,
    )

    parser.add_argument(
        "--seed-ratio",
        type=float,
        default=0.5,
        help="Anteil der Beispiele für Seed-Phase (default: 0.5 = 50%%)",
    )
    parser.add_argument(
        "--total-examples",
        type=int,
        default=100,
        help="Gesamtzahl der QA-Beispiele (default: 100)",
    )
    parser.add_argument(
        "--validate-seed",
        action="store_true",
        help="Auch Seed-Triples minimal validieren (Schema)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["hotpotqa", "musique"],
        default="hotpotqa",
        help="Benchmark-Dataset (default: hotpotqa)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="LLM-basierte Triple-Extraktion aktivieren",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama-Modell (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/seed_update",
        help="Ausgabeverzeichnis (default: results/seed_update)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("\n" + "=" * 70)
    print("SEED + UPDATE EVALUATION")
    print("=" * 70)
    print(f"Seed Ratio:      {args.seed_ratio:.0%}")
    print(f"Total Examples:  {args.total_examples}")
    print(f"Validate Seed:   {args.validate_seed}")
    print(f"Benchmark:       {args.benchmark}")
    print(f"With LLM:        {args.with_llm}")
    print(f"Output:          {args.output_dir}")
    print("=" * 70)

    # GPU status if using LLM
    llm_client = None
    if args.with_llm:
        print_gpu_status()
        llm_client = create_llm_client(args.llm_model)
        if not llm_client:
            logger.warning("LLM client not available - using heuristic extraction")

    # Create extractor
    extractor = EnhancedTripleExtractor(
        llm_client=llm_client,
        use_cache=True,
        cache_dir=output_dir / "cache",
    )

    # Create configs
    seed_update_config = SeedUpdateConfig(
        seed_ratio=args.seed_ratio,
        total_examples=args.total_examples,
        validate_seed=args.validate_seed,
        benchmark=args.benchmark,
    )

    consistency_config = get_consistency_config()

    # Run evaluation
    evaluator = SeedUpdateEvaluator(
        config=seed_update_config,
        consistency_config=consistency_config,
        extractor=extractor,
        llm_client=llm_client,
    )

    result = evaluator.run()

    # Save results
    result_dict = result.to_dict()
    result_dict["timestamp"] = datetime.now().isoformat()
    result_dict["args"] = vars(args)

    # Save JSON
    json_path = output_dir / "seed_update_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {json_path}")

    # Save LaTeX
    latex_path = output_dir / "seed_update_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(generate_latex_table(result))
    logger.info(f"LaTeX table saved: {latex_path}")

    # Print extraction stats
    stats = extractor.get_stats()
    if stats["total_paragraphs"] > 0:
        print("\n" + "-" * 40)
        print("EXTRACTION STATISTICS")
        print("-" * 40)
        print(f"Total paragraphs:    {stats['total_paragraphs']}")
        print(f"From cache:          {stats['paragraphs_from_cache']}")
        print(f"From LLM:            {stats['paragraphs_from_llm']}")
        print(f"Cache hit rate:      {stats['cache_hit_rate']:.1%}")
        print(f"Total triples:       {stats['total_triples']}")
        if stats['paragraphs_from_llm'] > 0:
            print(f"Avg LLM time:        {stats['avg_llm_time_per_paragraph']:.2f}s/paragraph")
        print("-" * 40)

    # Final output
    print(f"\nGenerated files in {output_dir}/:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
