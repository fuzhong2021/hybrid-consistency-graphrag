# src/evaluation/seed_update_evaluator.py
"""
Seed + Update Evaluation Framework.

Realistische Evaluation des Konsistenzmoduls:
1. SEED-Phase: Initiale Daten ohne Prüfung in den Graph laden
2. UPDATE-Phase: Neue Daten mit voller Konsistenzprüfung einfügen

Dies simuliert ein realistisches Szenario, wo ein existierender
Knowledge Graph mit neuen Dokumenten aktualisiert wird.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum

from src.models.entities import Entity, EntityType, Triple, ValidationStatus
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.orchestrator import ConsistencyOrchestrator
from src.consistency.base import ConsistencyConfig
from src.evaluation.benchmark_loader import BenchmarkLoader, QAExample

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Source of seed data."""
    HOTPOTQA_SPLIT = "hotpotqa_split"  # Split HotpotQA into seed/update
    JSON_FILE = "json_file"            # Load from JSON file
    WIKIDATA = "wikidata"              # Load from Wikidata (future)


@dataclass
class SeedUpdateConfig:
    """Configuration for seed + update evaluation."""
    # Data split
    seed_ratio: float = 0.5  # 50% seed, 50% update
    total_examples: int = 200

    # Seed options
    seed_source: DataSource = DataSource.HOTPOTQA_SPLIT
    seed_file: Optional[str] = None  # For JSON_FILE source

    # Validation during seed
    validate_seed: bool = False  # If True, also validate seed triples

    # Benchmark
    benchmark: str = "hotpotqa"


@dataclass
class PhaseResult:
    """Result of a single phase (seed or update)."""
    phase: str
    total_triples: int = 0
    accepted: int = 0
    rejected: int = 0

    # By rejection reason
    rejected_schema: int = 0
    rejected_duplicate: int = 0
    rejected_cardinality: int = 0
    rejected_temporal: int = 0
    rejected_other: int = 0

    # Timing
    duration_seconds: float = 0.0

    def acceptance_rate(self) -> float:
        if self.total_triples == 0:
            return 0.0
        return self.accepted / self.total_triples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "total_triples": self.total_triples,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.acceptance_rate(),
            "rejected_by_reason": {
                "schema": self.rejected_schema,
                "duplicate": self.rejected_duplicate,
                "cardinality": self.rejected_cardinality,
                "temporal": self.rejected_temporal,
                "other": self.rejected_other,
            },
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SeedUpdateResult:
    """Combined result of seed + update evaluation."""
    seed_result: PhaseResult
    update_result: PhaseResult

    # Graph stats
    final_entities: int = 0
    final_relations: int = 0

    # Config used
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed_result.to_dict(),
            "update": self.update_result.to_dict(),
            "graph": {
                "final_entities": self.final_entities,
                "final_relations": self.final_relations,
            },
            "config": self.config,
            "summary": {
                "seed_acceptance_rate": self.seed_result.acceptance_rate(),
                "update_acceptance_rate": self.update_result.acceptance_rate(),
                "update_rejection_rate": 1 - self.update_result.acceptance_rate(),
            }
        }


class SeedUpdateEvaluator:
    """
    Evaluates the consistency module in a realistic seed + update scenario.

    Workflow:
    1. Load seed data (e.g., first 50% of examples)
    2. Extract triples from seed data
    3. Store seed triples directly (no or minimal validation)
    4. Load update data (remaining 50%)
    5. Extract triples from update data
    6. Validate each update triple against the seeded graph
    7. Measure acceptance/rejection rates and reasons
    """

    def __init__(
        self,
        config: SeedUpdateConfig,
        consistency_config: ConsistencyConfig,
        extractor: Any = None,  # EnhancedTripleExtractor
        llm_client: Any = None,
    ):
        self.config = config
        self.consistency_config = consistency_config
        self.extractor = extractor
        self.llm_client = llm_client

        # Will be initialized during run
        self.graph_repo: Optional[InMemoryGraphRepository] = None
        self.orchestrator: Optional[ConsistencyOrchestrator] = None

    def run(self) -> SeedUpdateResult:
        """Run the complete seed + update evaluation."""
        logger.info("=" * 60)
        logger.info("SEED + UPDATE EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Seed Ratio: {self.config.seed_ratio:.0%}")
        logger.info(f"Total Examples: {self.config.total_examples}")
        logger.info(f"Validate Seed: {self.config.validate_seed}")

        # Load examples
        seed_examples, update_examples = self._load_and_split_examples()

        # Initialize fresh graph
        self.graph_repo = InMemoryGraphRepository()

        # Phase 1: Seed
        logger.info("\n" + "-" * 40)
        logger.info("PHASE 1: SEED (Initial Data Load)")
        logger.info("-" * 40)
        seed_result = self._run_seed_phase(seed_examples)

        # Initialize orchestrator AFTER seeding (so it checks against seed data)
        self.orchestrator = ConsistencyOrchestrator(
            config=self.consistency_config,
            graph_repo=self.graph_repo,
            embedding_model=None,
            llm_client=self.llm_client,
            enable_metrics=True,
            always_check_duplicates=True,
        )

        # Phase 2: Update
        logger.info("\n" + "-" * 40)
        logger.info("PHASE 2: UPDATE (With Consistency Check)")
        logger.info("-" * 40)
        update_result = self._run_update_phase(update_examples)

        # Compile results
        result = SeedUpdateResult(
            seed_result=seed_result,
            update_result=update_result,
            final_entities=len(self.graph_repo._entities),
            final_relations=len(self.graph_repo._relations),
            config={
                "seed_ratio": self.config.seed_ratio,
                "total_examples": self.config.total_examples,
                "validate_seed": self.config.validate_seed,
                "benchmark": self.config.benchmark,
            }
        )

        self._print_summary(result)

        return result

    def _load_and_split_examples(self) -> Tuple[List[QAExample], List[QAExample]]:
        """Load benchmark examples and split into seed/update."""
        loader = BenchmarkLoader()

        if self.config.benchmark.lower() == "hotpotqa":
            examples = loader.load_hotpotqa(
                split="validation",
                sample_size=self.config.total_examples
            )
        elif self.config.benchmark.lower() == "musique":
            examples = loader.load_musique(
                split="validation",
                sample_size=self.config.total_examples
            )
        else:
            raise ValueError(f"Unknown benchmark: {self.config.benchmark}")

        # Split
        split_idx = int(len(examples) * self.config.seed_ratio)
        seed_examples = examples[:split_idx]
        update_examples = examples[split_idx:]

        logger.info(f"Loaded {len(examples)} examples")
        logger.info(f"  Seed: {len(seed_examples)} examples")
        logger.info(f"  Update: {len(update_examples)} examples")

        return seed_examples, update_examples

    def _run_seed_phase(self, examples: List[QAExample]) -> PhaseResult:
        """Run seed phase: extract and store triples (minimal validation)."""
        result = PhaseResult(phase="seed")
        start_time = time.time()

        if self.extractor is None:
            from src.evaluation.comprehensive import EnhancedTripleExtractor
            self.extractor = EnhancedTripleExtractor()

        for i, example in enumerate(examples):
            if (i + 1) % 20 == 0:
                logger.info(f"  Seed progress: {i + 1}/{len(examples)}")

            tagged_triples = self.extractor.extract_all_triples(example)

            for tagged in tagged_triples:
                result.total_triples += 1
                triple = tagged.triple

                if self.config.validate_seed:
                    # Minimal validation (schema only)
                    if not self._basic_schema_check(triple):
                        result.rejected += 1
                        result.rejected_schema += 1
                        continue

                # Store directly
                try:
                    # Create fresh triple to avoid reference issues
                    fresh = Triple(
                        subject=Entity(
                            name=triple.subject.name,
                            entity_type=triple.subject.entity_type,
                            source_document=triple.subject.source_document,
                        ),
                        predicate=triple.predicate,
                        object=Entity(
                            name=triple.object.name,
                            entity_type=triple.object.entity_type,
                            source_document=triple.object.source_document,
                        ),
                        source_text=triple.source_text,
                        source_document_id=triple.source_document_id,
                        extraction_confidence=triple.extraction_confidence,
                        validation_status=ValidationStatus.ACCEPTED,
                    )
                    self.graph_repo.save_triple(fresh)
                    result.accepted += 1
                except Exception as e:
                    logger.debug(f"Could not save seed triple: {e}")
                    result.rejected += 1
                    result.rejected_other += 1

        result.duration_seconds = time.time() - start_time

        logger.info(f"Seed complete: {result.accepted}/{result.total_triples} "
                   f"triples stored ({result.acceptance_rate():.1%})")
        logger.info(f"Graph now has {len(self.graph_repo._entities)} entities, "
                   f"{len(self.graph_repo._relations)} relations")

        return result

    def _run_update_phase(self, examples: List[QAExample]) -> PhaseResult:
        """Run update phase: extract and validate triples against seeded graph."""
        result = PhaseResult(phase="update")
        start_time = time.time()

        for i, example in enumerate(examples):
            if (i + 1) % 20 == 0:
                logger.info(f"  Update progress: {i + 1}/{len(examples)}")

            tagged_triples = self.extractor.extract_all_triples(example)

            for tagged in tagged_triples:
                result.total_triples += 1
                triple = tagged.triple

                # Create fresh triple
                fresh = Triple(
                    subject=Entity(
                        name=triple.subject.name,
                        entity_type=triple.subject.entity_type,
                        source_document=triple.subject.source_document,
                    ),
                    predicate=triple.predicate,
                    object=Entity(
                        name=triple.object.name,
                        entity_type=triple.object.entity_type,
                        source_document=triple.object.source_document,
                    ),
                    source_text=triple.source_text,
                    source_document_id=triple.source_document_id,
                    extraction_confidence=triple.extraction_confidence,
                )

                # Full validation
                validated = self.orchestrator.process(fresh)

                if validated.validation_status == ValidationStatus.ACCEPTED:
                    result.accepted += 1
                    try:
                        self.graph_repo.save_triple(validated)
                    except Exception:
                        pass
                else:
                    result.rejected += 1
                    reason = self._categorize_rejection(validated)
                    if reason == "schema":
                        result.rejected_schema += 1
                    elif reason == "duplicate":
                        result.rejected_duplicate += 1
                    elif reason == "cardinality":
                        result.rejected_cardinality += 1
                    elif reason == "temporal":
                        result.rejected_temporal += 1
                    else:
                        result.rejected_other += 1

        result.duration_seconds = time.time() - start_time

        logger.info(f"Update complete: {result.accepted}/{result.total_triples} "
                   f"accepted ({result.acceptance_rate():.1%})")
        logger.info(f"Rejected: {result.rejected} "
                   f"(schema={result.rejected_schema}, "
                   f"duplicate={result.rejected_duplicate}, "
                   f"cardinality={result.rejected_cardinality}, "
                   f"temporal={result.rejected_temporal}, "
                   f"other={result.rejected_other})")

        return result

    def _basic_schema_check(self, triple: Triple) -> bool:
        """Basic schema validation for seed phase."""
        # Check entity types
        valid_types = {"Person", "Organisation", "Ort", "Ereignis", "Konzept",
                       "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"}

        if triple.subject.entity_type.value not in valid_types:
            return False
        if triple.object.entity_type.value not in valid_types:
            return False

        return True

    def _categorize_rejection(self, triple: Triple) -> str:
        """Categorize the reason for rejection based on conflicts."""
        if not hasattr(triple, 'conflicts') or not triple.conflicts:
            return "other"

        for conflict in triple.conflicts:
            # ConflictSet has conflict_type attribute (ConflictType enum)
            if hasattr(conflict, 'conflict_type'):
                conflict_type = conflict.conflict_type.value.lower()
            elif hasattr(conflict, 'get'):
                conflict_type = conflict.get("type", "").lower()
            else:
                conflict_type = str(conflict).lower()

            if "schema" in conflict_type:
                return "schema"
            if "duplicate" in conflict_type or "entity_match" in conflict_type:
                return "duplicate"
            if "cardinality" in conflict_type:
                return "cardinality"
            if "temporal" in conflict_type:
                return "temporal"

        return "other"

    def _print_summary(self, result: SeedUpdateResult):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("SEED + UPDATE EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nPHASE 1: SEED")
        print(f"  Triples loaded:    {result.seed_result.accepted}")
        print(f"  Acceptance rate:   {result.seed_result.acceptance_rate():.1%}")
        print(f"  Duration:          {result.seed_result.duration_seconds:.1f}s")

        print(f"\nPHASE 2: UPDATE (with consistency check)")
        print(f"  Total triples:     {result.update_result.total_triples}")
        print(f"  Accepted:          {result.update_result.accepted}")
        print(f"  Rejected:          {result.update_result.rejected}")
        print(f"  Acceptance rate:   {result.update_result.acceptance_rate():.1%}")
        print(f"  Rejection rate:    {1 - result.update_result.acceptance_rate():.1%}")

        print(f"\n  Rejections by reason:")
        print(f"    Schema:          {result.update_result.rejected_schema}")
        print(f"    Duplicate:       {result.update_result.rejected_duplicate}")
        print(f"    Cardinality:     {result.update_result.rejected_cardinality}")
        print(f"    Temporal:        {result.update_result.rejected_temporal}")
        print(f"    Other:           {result.update_result.rejected_other}")

        print(f"\nFINAL GRAPH")
        print(f"  Entities:          {result.final_entities}")
        print(f"  Relations:         {result.final_relations}")

        print("=" * 60)


def generate_latex_table(result: SeedUpdateResult) -> str:
    """Generate LaTeX table for seed+update results."""
    seed = result.seed_result
    update = result.update_result

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Seed + Update Evaluation: Konsistenzmodul-Wirksamkeit}",
        r"\label{tab:seed-update-evaluation}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Metrik} & \textbf{Seed} & \textbf{Update} \\",
        r"\midrule",
        f"Triples verarbeitet & {seed.total_triples} & {update.total_triples} \\\\",
        f"Akzeptiert & {seed.accepted} & {update.accepted} \\\\",
        f"Abgelehnt & {seed.rejected} & {update.rejected} \\\\",
        f"Akzeptanzrate & {seed.acceptance_rate():.1%} & {update.acceptance_rate():.1%} \\\\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Ablehnungsgründe (Update):}} \\",
        f"\\quad Schema-Verletzung & — & {update.rejected_schema} \\\\",
        f"\\quad Duplikat erkannt & — & {update.rejected_duplicate} \\\\",
        f"\\quad Kardinalität & — & {update.rejected_cardinality} \\\\",
        f"\\quad Temporal & — & {update.rejected_temporal} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)
