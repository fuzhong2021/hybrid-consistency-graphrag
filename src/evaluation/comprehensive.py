# src/evaluation/comprehensive.py
"""
Comprehensive Evaluation Framework for the Consistency Module.

Implements 4 complementary evaluation strategies:
1. Intrinsic Evaluation — synthetic error detection (P/R/F1)
2. Manual Annotation — human agreement (Cohen's Kappa)
3. Graph Quality Metrics — structural improvement (before/after)
4. Ablation + Information Quality — signal/noise separation

Used by: scripts/run_comprehensive_evaluation.py
"""

import json
import math
import logging
import random
import uuid
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, TYPE_CHECKING
from collections import defaultdict
from pathlib import Path

from src.models.entities import (
    Entity, EntityType, Triple, ValidationStatus, Relation
)
from src.evaluation.benchmark_loader import QAExample
from src.graph.memory_repository import InMemoryGraphRepository
from src.consistency.base import ConsistencyConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Triple Cache for LLM Extractions
# =============================================================================

class TripleCache:
    """
    Caches extracted triples to avoid repeated LLM calls.

    Cache is stored as JSON file and keyed by content hash.
    """

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("results/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "triple_extraction_cache.json"
        self._cache: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached extractions")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    def _hash_content(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def get(self, example_id: str, paragraph_text: str) -> Optional[List[Dict]]:
        """Get cached triples for a paragraph."""
        key = f"{example_id}:{self._hash_content(paragraph_text)}"
        return self._cache.get(key)

    def set(self, example_id: str, paragraph_text: str, triples: List[Dict]):
        """Cache triples for a paragraph."""
        key = f"{example_id}:{self._hash_content(paragraph_text)}"
        self._cache[key] = triples
        # Save periodically (every 10 new entries)
        if len(self._cache) % 10 == 0:
            self._save_cache()

    def save(self):
        """Force save cache to disk."""
        self._save_cache()

    @property
    def size(self) -> int:
        return len(self._cache)


# =============================================================================
# TaggedTriple & Enhanced Extraction
# =============================================================================

@dataclass
class TaggedTriple:
    """A triple tagged with its provenance (supporting fact or distractor)."""
    triple: Triple
    is_from_supporting_fact: bool  # True = Signal, False = Noise
    source_paragraph_title: str = ""


@dataclass
class ExtractionStats:
    """Statistics for triple extraction."""
    total_paragraphs: int = 0
    paragraphs_from_cache: int = 0
    paragraphs_from_llm: int = 0
    total_triples: int = 0
    llm_time_seconds: float = 0.0


class EnhancedTripleExtractor:
    """
    Extracts triples from ALL paragraphs in a QA example using LLM.

    Features:
    - Uses real LLM-based extraction (TripleExtractor) when llm_client is provided
    - Falls back to heuristic extraction without LLM
    - Caches extractions to avoid repeated LLM calls
    - Tags each triple with provenance (supporting fact or distractor)

    HotpotQA has ~10 paragraphs per example (2 supporting, 8 distractor).
    """

    def __init__(
        self,
        llm_client: Any = None,
        use_cache: bool = True,
        cache_dir: Path = None
    ):
        """
        Initialize the extractor.

        Args:
            llm_client: OpenAI-compatible LLM client (e.g., OllamaClient)
            use_cache: Whether to cache LLM extractions
            cache_dir: Directory for cache files
        """
        self.llm_client = llm_client
        self.use_cache = use_cache
        self.cache = TripleCache(cache_dir) if use_cache else None
        self._triple_extractor = None
        self.stats = ExtractionStats()

        # Initialize real TripleExtractor if LLM client available
        if llm_client:
            try:
                from src.extraction.triple_extractor import TripleExtractor, ExtractionConfig
                config = ExtractionConfig(
                    model=getattr(llm_client, 'model', 'llama3.1:8b'),
                    temperature=0.0,
                    use_few_shot=True,
                    min_confidence=0.3,
                )
                self._triple_extractor = TripleExtractor(config, llm_client)
                logger.info("EnhancedTripleExtractor: Using LLM-based extraction")
            except Exception as e:
                logger.warning(f"Could not initialize TripleExtractor: {e}")
                logger.info("EnhancedTripleExtractor: Falling back to heuristic extraction")
        else:
            logger.info("EnhancedTripleExtractor: Using heuristic extraction (no LLM)")

    def extract_all_triples(self, example: QAExample) -> List[TaggedTriple]:
        """
        Extract tagged triples from all paragraphs of a QA example.

        If LLM client is available, uses real LLM-based extraction.
        Otherwise, falls back to heuristic extraction.
        """
        if self._triple_extractor:
            return self._extract_with_llm(example)
        else:
            return self._extract_heuristic(example)

    def _extract_with_llm(self, example: QAExample) -> List[TaggedTriple]:
        """Extract triples using LLM-based TripleExtractor."""
        tagged_triples = []
        sf_titles: Set[str] = {sf.title for sf in example.supporting_facts}

        for para in example.context_paragraphs:
            title = para.get("title", "")
            sentences = para.get("sentences", [])
            text = " ".join(sentences) if sentences else ""

            if not text.strip():
                continue

            self.stats.total_paragraphs += 1
            is_supporting = title in sf_titles

            # Check cache first
            cached = self.cache.get(example.id, text) if self.cache else None

            if cached is not None:
                self.stats.paragraphs_from_cache += 1
                triples_data = cached
            else:
                # Extract with LLM
                self.stats.paragraphs_from_llm += 1
                start_time = time.time()

                try:
                    result = self._triple_extractor.extract(
                        text=text,
                        document_id=example.id
                    )
                    triples_data = [
                        {
                            "subject": t.subject.name,
                            "subject_type": t.subject.entity_type.value,
                            "predicate": t.predicate,
                            "object": t.object.name,
                            "object_type": t.object.entity_type.value,
                            "confidence": t.extraction_confidence,
                            "source_text": t.source_text,
                        }
                        for t in result.triples
                    ]
                except Exception as e:
                    logger.warning(f"LLM extraction failed for '{title}': {e}")
                    triples_data = []

                self.stats.llm_time_seconds += time.time() - start_time

                # Cache the result
                if self.cache:
                    self.cache.set(example.id, text, triples_data)

            # Convert to TaggedTriple
            for td in triples_data:
                try:
                    triple = Triple(
                        subject=Entity(
                            name=td["subject"],
                            entity_type=EntityType.from_string(td.get("subject_type", "Konzept")),
                            source_document=example.id,
                        ),
                        predicate=td["predicate"],
                        object=Entity(
                            name=td["object"],
                            entity_type=EntityType.from_string(td.get("object_type", "Konzept")),
                            source_document=example.id,
                        ),
                        source_text=td.get("source_text", text[:200]),
                        source_document_id=example.id,
                        extraction_confidence=td.get("confidence", 0.7),
                    )

                    tagged_triples.append(TaggedTriple(
                        triple=triple,
                        is_from_supporting_fact=is_supporting,
                        source_paragraph_title=title,
                    ))
                    self.stats.total_triples += 1

                except Exception as e:
                    logger.debug(f"Could not create triple: {e}")

        # Save cache at the end
        if self.cache:
            self.cache.save()

        return tagged_triples

    def _extract_heuristic(self, example: QAExample) -> List[TaggedTriple]:
        """
        Fallback heuristic extraction (no LLM required).

        Creates triples from paragraph titles and answer connections.
        Less accurate but fast and deterministic.
        """
        tagged_triples = []
        entity_cache: Dict[str, Entity] = {}

        # Determine which titles are supporting facts
        sf_titles: Set[str] = {sf.title for sf in example.supporting_facts}

        # 1. Create entities from ALL paragraph titles
        for para in example.context_paragraphs:
            title = para.get("title", "")
            if not title or title in entity_cache:
                continue

            entity_type = self._infer_entity_type(title)
            entity = Entity(
                name=title,
                entity_type=entity_type,
                source_document=example.id
            )
            entity_cache[title] = entity

        # Get ordered list of all titles (preserving paragraph order)
        all_titles = []
        for para in example.context_paragraphs:
            title = para.get("title", "")
            if title and title not in all_titles:
                all_titles.append(title)

        # 2. Create triples between consecutive paragraphs (ALL, not just SF)
        for i in range(len(all_titles) - 1):
            title1 = all_titles[i]
            title2 = all_titles[i + 1]

            if title1 not in entity_cache or title2 not in entity_cache:
                continue

            relation_type = self._infer_relation_type(
                entity_cache[title1],
                entity_cache[title2],
                example
            )

            # A triple is "from supporting fact" if BOTH entities are SF paragraphs
            is_supporting = title1 in sf_titles and title2 in sf_titles

            triple = Triple(
                subject=Entity(
                    name=entity_cache[title1].name,
                    entity_type=entity_cache[title1].entity_type,
                    source_document=example.id,
                ),
                predicate=relation_type,
                object=Entity(
                    name=entity_cache[title2].name,
                    entity_type=entity_cache[title2].entity_type,
                    source_document=example.id,
                ),
                source_text=example.question,
                source_document_id=example.id,
                extraction_confidence=0.8 if is_supporting else 0.5,
            )

            tagged_triples.append(TaggedTriple(
                triple=triple,
                is_from_supporting_fact=is_supporting,
                source_paragraph_title=f"{title1} -> {title2}",
            ))

        # 3. Connect ALL paragraph entities to the answer entity
        answer = example.answer
        if answer and len(answer) > 2:
            answer_type = self._infer_entity_type(answer)

            for title in all_titles:
                if title not in entity_cache:
                    continue

                is_supporting = title in sf_titles

                triple = Triple(
                    subject=Entity(
                        name=entity_cache[title].name,
                        entity_type=entity_cache[title].entity_type,
                        source_document=example.id,
                    ),
                    predicate="RELATED_TO",
                    object=Entity(
                        name=answer,
                        entity_type=answer_type,
                        source_document=example.id,
                    ),
                    source_text=example.question,
                    source_document_id=example.id,
                    extraction_confidence=0.7 if is_supporting else 0.4,
                )

                tagged_triples.append(TaggedTriple(
                    triple=triple,
                    is_from_supporting_fact=is_supporting,
                    source_paragraph_title=title,
                ))

        return tagged_triples

    def _infer_entity_type(self, name: str) -> EntityType:
        """Infer entity type from name (heuristic)."""
        name_lower = name.lower()

        location_keywords = [
            "city", "country", "river", "mountain", "lake",
            "island", "state", "county", "village", "town",
        ]
        if any(kw in name_lower for kw in location_keywords):
            return EntityType.LOCATION

        org_keywords = [
            "university", "company", "corporation", "inc", "ltd",
            "organization", "institute", "foundation", "college",
            "school", "hospital", "museum", "library", "bank",
        ]
        if any(kw in name_lower for kw in org_keywords):
            return EntityType.ORGANIZATION

        event_keywords = [
            "war", "battle", "election", "festival", "championship",
            "world cup", "olympics", "conference", "summit",
        ]
        if any(kw in name_lower for kw in event_keywords):
            return EntityType.EVENT

        return EntityType.PERSON

    def _infer_relation_type(
        self,
        subject: Entity,
        obj: Entity,
        example: QAExample,
    ) -> str:
        """Infer relation type from context (heuristic)."""
        question_lower = example.question.lower()

        if "born" in question_lower or "birthplace" in question_lower:
            return "GEBOREN_IN"
        if "died" in question_lower or "death" in question_lower:
            return "GESTORBEN_IN"
        if "work" in question_lower or "employ" in question_lower:
            return "ARBEITET_BEI"
        if "direct" in question_lower or "film" in question_lower or "movie" in question_lower:
            return "REGIE_BEI"
        if "found" in question_lower or "establish" in question_lower:
            return "GRUENDETE"
        if "locat" in question_lower or "where" in question_lower:
            return "BEFINDET_SICH_IN"
        if "member" in question_lower or "belong" in question_lower:
            return "MITGLIED_VON"
        if "play" in question_lower or "sport" in question_lower:
            return "SPIELT_FUER"
        if "marr" in question_lower or "spouse" in question_lower:
            return "VERHEIRATET_MIT"
        if ("child" in question_lower or "parent" in question_lower
                or "son" in question_lower or "daughter" in question_lower):
            return "VERWANDT_MIT"

        if subject.entity_type == EntityType.PERSON:
            if obj.entity_type == EntityType.ORGANIZATION:
                return "ASSOZIIERT_MIT"
            if obj.entity_type == EntityType.LOCATION:
                return "VERBUNDEN_MIT"
            if obj.entity_type == EntityType.EVENT:
                return "TEILNAHME_AN"

        return "RELATED_TO"

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "total_paragraphs": self.stats.total_paragraphs,
            "paragraphs_from_cache": self.stats.paragraphs_from_cache,
            "paragraphs_from_llm": self.stats.paragraphs_from_llm,
            "cache_hit_rate": (
                self.stats.paragraphs_from_cache / max(1, self.stats.total_paragraphs)
            ),
            "total_triples": self.stats.total_triples,
            "llm_time_seconds": self.stats.llm_time_seconds,
            "avg_llm_time_per_paragraph": (
                self.stats.llm_time_seconds / max(1, self.stats.paragraphs_from_llm)
            ),
            "cache_size": self.cache.size if self.cache else 0,
            "using_llm": self._triple_extractor is not None,
        }


# =============================================================================
# Strategy 3: Graph Quality Metrics
# =============================================================================

@dataclass
class GraphQualityMetrics:
    """Structural quality metrics for a knowledge graph."""
    # Schema compliance
    schema_compliant_entities: int = 0
    total_entities: int = 0
    schema_compliant_relations: int = 0
    total_relations: int = 0

    # Self-loops
    self_loop_count: int = 0

    # Duplication
    unique_fingerprints: int = 0
    total_entity_count: int = 0

    # Domain constraints
    domain_constraint_violations: int = 0

    # Degree statistics
    avg_degree: float = 0.0
    max_degree: int = 0
    degree_std: float = 0.0

    # Graph density
    density: float = 0.0

    # Isolated entities
    isolated_entities: int = 0

    @property
    def schema_compliance_rate(self) -> float:
        total = self.total_entities + self.total_relations
        compliant = self.schema_compliant_entities + self.schema_compliant_relations
        return compliant / total if total > 0 else 0.0

    @property
    def entity_duplication_rate(self) -> float:
        if self.total_entity_count == 0:
            return 0.0
        return 1.0 - (self.unique_fingerprints / self.total_entity_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_compliance_rate": round(self.schema_compliance_rate, 4),
            "schema_compliant_entities": self.schema_compliant_entities,
            "total_entities": self.total_entities,
            "schema_compliant_relations": self.schema_compliant_relations,
            "total_relations": self.total_relations,
            "self_loop_count": self.self_loop_count,
            "entity_duplication_rate": round(self.entity_duplication_rate, 4),
            "unique_fingerprints": self.unique_fingerprints,
            "total_entity_count": self.total_entity_count,
            "domain_constraint_violations": self.domain_constraint_violations,
            "avg_degree": round(self.avg_degree, 4),
            "max_degree": self.max_degree,
            "degree_std": round(self.degree_std, 4),
            "density": round(self.density, 6),
            "isolated_entities": self.isolated_entities,
        }


class GraphQualityAnalyzer:
    """Analyzes structural quality metrics of a knowledge graph."""

    def __init__(self, config: ConsistencyConfig):
        self.config = config

    def analyze(self, graph_repo: InMemoryGraphRepository) -> GraphQualityMetrics:
        """Compute quality metrics for a graph repository."""
        metrics = GraphQualityMetrics()

        all_entities = graph_repo.find_all_entities(include_invalid=False)
        all_relations = graph_repo.find_relations(include_invalid=False)

        metrics.total_entities = len(all_entities)
        metrics.total_relations = len(all_relations)
        metrics.total_entity_count = len(all_entities)

        # Schema compliance — entities
        valid_types = set(self.config.valid_entity_types)
        for entity in all_entities:
            if entity.entity_type.value in valid_types:
                metrics.schema_compliant_entities += 1

        # Schema compliance — relations + self-loops + domain constraints
        valid_rel_types = set(self.config.valid_relation_types)
        for rel_info in all_relations:
            rel = rel_info.get("relation")
            if rel is None:
                continue

            if rel.relation_type in valid_rel_types:
                metrics.schema_compliant_relations += 1

            # Self-loop check
            if rel.source_id == rel.target_id:
                metrics.self_loop_count += 1

            # Domain constraint check
            source_entity = graph_repo.get_entity(rel.source_id)
            target_entity = graph_repo.get_entity(rel.target_id)
            if source_entity and target_entity:
                constraint = self.config.domain_constraints.get(rel.relation_type)
                if constraint:
                    subj_types = constraint.get("subject_types", [])
                    obj_types = constraint.get("object_types", [])
                    if subj_types and source_entity.entity_type.value not in subj_types:
                        metrics.domain_constraint_violations += 1
                    elif obj_types and target_entity.entity_type.value not in obj_types:
                        metrics.domain_constraint_violations += 1

        # Entity duplication (unique fingerprints)
        fingerprints = set()
        for entity in all_entities:
            fingerprints.add(entity.fingerprint)
        metrics.unique_fingerprints = len(fingerprints)

        # Degree statistics
        degree_map: Dict[str, int] = {}
        for entity in all_entities:
            degree_map[entity.id] = 0

        for rel_info in all_relations:
            rel = rel_info.get("relation")
            if rel:
                degree_map[rel.source_id] = degree_map.get(rel.source_id, 0) + 1
                degree_map[rel.target_id] = degree_map.get(rel.target_id, 0) + 1

        if degree_map:
            degrees = list(degree_map.values())
            metrics.avg_degree = sum(degrees) / len(degrees)
            metrics.max_degree = max(degrees)

            if len(degrees) > 1:
                mean = metrics.avg_degree
                variance = sum((d - mean) ** 2 for d in degrees) / len(degrees)
                metrics.degree_std = math.sqrt(variance)

            metrics.isolated_entities = sum(1 for d in degrees if d == 0)

        # Graph density: E / (V * (V-1))
        v = metrics.total_entities
        e = metrics.total_relations
        if v > 1:
            metrics.density = e / (v * (v - 1))

        return metrics

    def compare(
        self,
        unfiltered: GraphQualityMetrics,
        filtered: GraphQualityMetrics,
    ) -> Dict[str, Any]:
        """Compare unfiltered (Graph A) vs. filtered (Graph B) quality."""

        def delta(a: float, b: float) -> float:
            return round(b - a, 6)

        def pct_change(a: float, b: float) -> str:
            if a == 0:
                return "N/A"
            return f"{((b - a) / a) * 100:+.1f}%"

        return {
            "before": unfiltered.to_dict(),
            "after": filtered.to_dict(),
            "delta": {
                "schema_compliance_rate": delta(
                    unfiltered.schema_compliance_rate, filtered.schema_compliance_rate
                ),
                "self_loop_count": delta(
                    unfiltered.self_loop_count, filtered.self_loop_count
                ),
                "entity_duplication_rate": delta(
                    unfiltered.entity_duplication_rate, filtered.entity_duplication_rate
                ),
                "domain_constraint_violations": delta(
                    unfiltered.domain_constraint_violations,
                    filtered.domain_constraint_violations,
                ),
                "avg_degree": delta(unfiltered.avg_degree, filtered.avg_degree),
                "density": delta(unfiltered.density, filtered.density),
                "isolated_entities": delta(
                    unfiltered.isolated_entities, filtered.isolated_entities
                ),
                "total_entities": delta(
                    unfiltered.total_entities, filtered.total_entities
                ),
                "total_relations": delta(
                    unfiltered.total_relations, filtered.total_relations
                ),
            },
            "pct_change": {
                "schema_compliance_rate": pct_change(
                    unfiltered.schema_compliance_rate, filtered.schema_compliance_rate
                ),
                "self_loop_count": pct_change(
                    unfiltered.self_loop_count, filtered.self_loop_count
                ),
                "domain_constraint_violations": pct_change(
                    unfiltered.domain_constraint_violations,
                    filtered.domain_constraint_violations,
                ),
                "density": pct_change(unfiltered.density, filtered.density),
            },
        }


# =============================================================================
# Strategy 2: Manual Annotation
# =============================================================================

@dataclass
class AnnotationSample:
    """A sample triple for manual annotation."""
    id: str
    subject_name: str
    subject_type: str
    predicate: str
    object_name: str
    object_type: str
    source_text: str
    validation_status: str  # "accepted" or "rejected"
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    # Human annotation (filled after manual annotation)
    human_label: Optional[str] = None  # "correct", "incorrect", "uncertain"
    annotator_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject_name": self.subject_name,
            "subject_type": self.subject_type,
            "predicate": self.predicate,
            "object_name": self.object_name,
            "object_type": self.object_type,
            "source_text": self.source_text,
            "validation_status": self.validation_status,
            "validation_history": self.validation_history,
            "human_label": self.human_label,
            "annotator_notes": self.annotator_notes,
        }


class AnnotationManager:
    """Manages annotation sample creation, export, import, and agreement metrics."""

    def __init__(self, sample_size: int = 100):
        self.sample_size = sample_size

    def create_annotation_sample(
        self,
        validated_triples: List[Tuple[Triple, str]],
    ) -> List[AnnotationSample]:
        """
        Create a stratified annotation sample.

        Args:
            validated_triples: List of (triple, validation_status_string) tuples

        Returns:
            Stratified sample (target: 50 accepted, 50 rejected)
        """
        accepted = [(t, s) for t, s in validated_triples if s == "accepted"]
        rejected = [
            (t, s) for t, s in validated_triples
            if s in ("rejected", "needs_review", "conflicting")
        ]

        half = self.sample_size // 2

        accepted_sample = random.sample(accepted, min(half, len(accepted)))
        rejected_sample = random.sample(rejected, min(half, len(rejected)))

        # If one group is too small, take more from the other
        if len(accepted_sample) < half and len(rejected) > half:
            extra = half - len(accepted_sample)
            remaining = [r for r in rejected if r not in rejected_sample]
            rejected_sample.extend(
                random.sample(remaining, min(extra, len(remaining)))
            )
        elif len(rejected_sample) < half and len(accepted) > half:
            extra = half - len(rejected_sample)
            remaining = [a for a in accepted if a not in accepted_sample]
            accepted_sample.extend(
                random.sample(remaining, min(extra, len(remaining)))
            )

        samples = []
        for triple, status in accepted_sample + rejected_sample:
            sample = AnnotationSample(
                id=str(uuid.uuid4())[:8],
                subject_name=triple.subject.name,
                subject_type=triple.subject.entity_type.value,
                predicate=triple.predicate,
                object_name=triple.object.name,
                object_type=triple.object.entity_type.value,
                source_text=triple.source_text or "",
                validation_status=status,
                validation_history=triple.validation_history,
            )
            samples.append(sample)

        random.shuffle(samples)
        logger.info(
            f"Created annotation sample: {len(samples)} triples "
            f"({len(accepted_sample)} accepted, {len(rejected_sample)} rejected)"
        )
        return samples

    def export_for_annotation(
        self,
        samples: List[AnnotationSample],
        path: str,
    ) -> None:
        """Export annotation samples as JSON with instructions."""
        export_data = {
            "instructions": {
                "task": (
                    "Bitte bewerten Sie jedes Triple als "
                    "'correct', 'incorrect' oder 'uncertain'."
                ),
                "correct": "Das Triple ist faktisch korrekt und sinnvoll.",
                "incorrect": (
                    "Das Triple ist faktisch falsch, unsinnig oder "
                    "schlecht typisiert."
                ),
                "uncertain": (
                    "Die Korrektheit kann nicht eindeutig bestimmt werden."
                ),
                "field_to_fill": "human_label",
                "optional_field": "annotator_notes",
            },
            "total_samples": len(samples),
            "samples": [s.to_dict() for s in samples],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported {len(samples)} samples for annotation to {path}")

    def import_annotations(self, path: str) -> List[AnnotationSample]:
        """Import annotated samples from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data.get("samples", []):
            sample = AnnotationSample(
                id=item["id"],
                subject_name=item["subject_name"],
                subject_type=item["subject_type"],
                predicate=item["predicate"],
                object_name=item["object_name"],
                object_type=item["object_type"],
                source_text=item.get("source_text", ""),
                validation_status=item["validation_status"],
                validation_history=item.get("validation_history", []),
                human_label=item.get("human_label"),
                annotator_notes=item.get("annotator_notes", ""),
            )
            samples.append(sample)

        annotated = sum(1 for s in samples if s.human_label is not None)
        logger.info(f"Imported {len(samples)} samples ({annotated} annotated)")
        return samples

    def compute_agreement_metrics(
        self,
        samples: List[AnnotationSample],
    ) -> Dict[str, Any]:
        """
        Compute agreement between system decisions and human annotations.

        Mapping:
        - System "accepted" → system says "correct"
        - System "rejected/needs_review/conflicting" → system says "incorrect"
        - Human "correct"/"incorrect" → used for P/R/F1
        - Human "uncertain" → excluded from P/R/F1 but included in Kappa
        """
        annotated = [s for s in samples if s.human_label is not None]

        if not annotated:
            return {"error": "No annotated samples found"}

        # For P/R/F1: exclude "uncertain"
        evaluable = [
            s for s in annotated if s.human_label in ("correct", "incorrect")
        ]

        tp = fp = fn = tn = 0
        for s in evaluable:
            system_correct = s.validation_status == "accepted"
            human_correct = s.human_label == "correct"

            if system_correct and human_correct:
                tp += 1
            elif system_correct and not human_correct:
                fp += 1
            elif not system_correct and human_correct:
                fn += 1
            else:
                tn += 1

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / total if total > 0 else 0.0

        kappa = self._compute_cohens_kappa(annotated)

        return {
            "total_annotated": len(annotated),
            "total_evaluable": len(evaluable),
            "uncertain_excluded": len(annotated) - len(evaluable),
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "cohens_kappa": round(kappa, 4),
        }

    def _compute_cohens_kappa(self, samples: List[AnnotationSample]) -> float:
        """
        Compute Cohen's Kappa between system and human decisions.

        Binary classification: correct (1) vs. not-correct (0).
        """
        n = len(samples)
        if n == 0:
            return 0.0

        a = b = c = d = 0
        for s in samples:
            sys_pos = s.validation_status == "accepted"
            hum_pos = s.human_label == "correct"

            if sys_pos and hum_pos:
                a += 1
            elif sys_pos and not hum_pos:
                b += 1
            elif not sys_pos and hum_pos:
                c += 1
            else:
                d += 1

        # Observed agreement
        po = (a + d) / n

        # Expected agreement
        p_sys_pos = (a + b) / n
        p_hum_pos = (a + c) / n
        pe = p_sys_pos * p_hum_pos + (1 - p_sys_pos) * (1 - p_hum_pos)

        if pe == 1.0:
            return 1.0

        return (po - pe) / (1 - pe)


# =============================================================================
# Strategy 4: Information Quality
# =============================================================================

@dataclass
class InformationQualityResult:
    """Results of information quality evaluation for one variant."""
    variant_name: str

    # Counts
    total_supporting: int = 0
    total_distractor: int = 0
    accepted_supporting: int = 0
    accepted_distractor: int = 0
    rejected_supporting: int = 0
    rejected_distractor: int = 0

    @property
    def supporting_preservation_rate(self) -> float:
        """Rate of supporting (signal) triples preserved."""
        if self.total_supporting == 0:
            return 0.0
        return self.accepted_supporting / self.total_supporting

    @property
    def distractor_removal_rate(self) -> float:
        """Rate of distractor (noise) triples removed."""
        if self.total_distractor == 0:
            return 0.0
        return self.rejected_distractor / self.total_distractor

    @property
    def information_precision(self) -> float:
        """Precision: supporting among accepted."""
        total_accepted = self.accepted_supporting + self.accepted_distractor
        if total_accepted == 0:
            return 0.0
        return self.accepted_supporting / total_accepted

    @property
    def information_recall(self) -> float:
        """Recall: same as supporting preservation rate."""
        return self.supporting_preservation_rate

    @property
    def information_f1(self) -> float:
        """F1 score combining precision and recall."""
        p = self.information_precision
        r = self.information_recall
        if (p + r) == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "total_supporting": self.total_supporting,
            "total_distractor": self.total_distractor,
            "accepted_supporting": self.accepted_supporting,
            "accepted_distractor": self.accepted_distractor,
            "rejected_supporting": self.rejected_supporting,
            "rejected_distractor": self.rejected_distractor,
            "supporting_preservation_rate": round(
                self.supporting_preservation_rate, 4
            ),
            "distractor_removal_rate": round(self.distractor_removal_rate, 4),
            "information_precision": round(self.information_precision, 4),
            "information_recall": round(self.information_recall, 4),
            "information_f1": round(self.information_f1, 4),
        }


class InformationQualityEvaluator:
    """Evaluates information quality by measuring signal/noise separation."""

    def __init__(
        self,
        config: ConsistencyConfig,
        extractor: EnhancedTripleExtractor = None,
    ):
        self.config = config
        self.extractor = extractor or EnhancedTripleExtractor()

    def evaluate_variant(
        self,
        examples: List[QAExample],
        variant_name: str,
        orchestrator: Optional[Any] = None,
        graph_repo: Optional[InMemoryGraphRepository] = None,
    ) -> InformationQualityResult:
        """
        Evaluate a single variant on signal/noise separation.

        Args:
            examples: QA examples to process
            variant_name: Name of the variant (BASELINE, STAGE1_ONLY, etc.)
            orchestrator: ConsistencyOrchestrator (None for BASELINE)
            graph_repo: Graph repository for storing accepted triples
        """
        result = InformationQualityResult(variant_name=variant_name)

        for example in examples:
            tagged_triples = self.extractor.extract_all_triples(example)

            for tagged in tagged_triples:
                if tagged.is_from_supporting_fact:
                    result.total_supporting += 1
                else:
                    result.total_distractor += 1

                if variant_name == "BASELINE" or orchestrator is None:
                    # Baseline: accept everything
                    if tagged.is_from_supporting_fact:
                        result.accepted_supporting += 1
                    else:
                        result.accepted_distractor += 1
                else:
                    # Create fresh triple to avoid mutation issues
                    fresh_triple = Triple(
                        subject=Entity(
                            name=tagged.triple.subject.name,
                            entity_type=tagged.triple.subject.entity_type,
                            source_document=tagged.triple.subject.source_document,
                        ),
                        predicate=tagged.triple.predicate,
                        object=Entity(
                            name=tagged.triple.object.name,
                            entity_type=tagged.triple.object.entity_type,
                            source_document=tagged.triple.object.source_document,
                        ),
                        source_text=tagged.triple.source_text,
                        source_document_id=tagged.triple.source_document_id,
                        extraction_confidence=tagged.triple.extraction_confidence,
                    )

                    validated = orchestrator.process(fresh_triple)
                    accepted = validated.validation_status == ValidationStatus.ACCEPTED

                    if tagged.is_from_supporting_fact:
                        if accepted:
                            result.accepted_supporting += 1
                        else:
                            result.rejected_supporting += 1
                    else:
                        if accepted:
                            result.accepted_distractor += 1
                        else:
                            result.rejected_distractor += 1

                    # Save accepted triples for subsequent duplicate checks
                    if accepted and graph_repo is not None:
                        try:
                            graph_repo.save_triple(validated)
                        except Exception:
                            pass

        return result


# =============================================================================
# LaTeX Table Generator
# =============================================================================

class LaTeXTableGenerator:
    """Generates LaTeX tables for thesis output."""

    @staticmethod
    def _pct(val: float) -> str:
        """Format a float (0.0–1.0) as LaTeX percentage string."""
        return f"{val * 100:.1f}\\%"

    @staticmethod
    def intrinsic_table(data: Dict[str, Any]) -> str:
        """Generate LaTeX table for intrinsic evaluation results."""
        metrics = data.get("metrics", {})
        per_type = data.get("per_type", {})
        pct = LaTeXTableGenerator._pct

        latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Intrinsische Evaluation: Fehlererkennung pro Fehlertyp}\n"
            "\\label{tab:intrinsic-evaluation}\n"
            "\\begin{tabular}{lrrrr}\n"
            "\\toprule\n"
            "\\textbf{Fehlertyp} & \\textbf{Total} & \\textbf{TP} "
            "& \\textbf{FN} & \\textbf{Recall} \\\\\n"
            "\\midrule\n"
        )

        for error_type, type_data in per_type.items():
            total = type_data.get("total", 0)
            tp = type_data.get("tp", 0)
            fn = type_data.get("fn", 0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            label = error_type.replace("_", " ").title()
            latex += f"{label} & {total} & {tp} & {fn} & {pct(recall)} \\\\\n"

        p = metrics.get("precision", 0)
        r = metrics.get("recall", 0)
        f1 = metrics.get("f1_score", 0)
        acc = metrics.get("accuracy", 0)

        latex += (
            "\\midrule\n"
            f"\\multicolumn{{5}}{{l}}{{\\textbf{{Gesamt:}} "
            f"Precision={pct(p)}, Recall={pct(r)}, "
            f"F1={pct(f1)}, Accuracy={pct(acc)}}} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        return latex

    @staticmethod
    def graph_quality_table(comparison: Dict[str, Any]) -> str:
        """Generate LaTeX table for graph quality comparison."""
        before = comparison.get("before", {})
        after = comparison.get("after", {})
        delta = comparison.get("delta", {})
        pct = LaTeXTableGenerator._pct

        latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Graph-Qualität: Vorher vs.\\ Nachher Konsistenzmodul}\n"
            "\\label{tab:graph-quality}\n"
            "\\begin{tabular}{lrrr}\n"
            "\\toprule\n"
            "\\textbf{Metrik} & \\textbf{Vorher} & \\textbf{Nachher} "
            "& \\textbf{$\\Delta$} \\\\\n"
            "\\midrule\n"
        )

        # (label, key, format_type)
        rows = [
            ("Schema-Compliance", "schema_compliance_rate", "pct"),
            ("Self-Loops", "self_loop_count", "int"),
            ("Entity-Duplikationsrate", "entity_duplication_rate", "pct"),
            ("Domain-Verletzungen", "domain_constraint_violations", "int"),
            ("$\\varnothing$ Knotengrad", "avg_degree", "float"),
            ("Graph-Dichte", "density", "float4"),
            ("Isolierte Entitäten", "isolated_entities", "int"),
            ("Entitäten", "total_entities", "int"),
            ("Relationen", "total_relations", "int"),
        ]

        for label, key, fmt in rows:
            b = before.get(key, 0)
            a = after.get(key, 0)
            d = delta.get(key, 0)

            if fmt == "pct":
                latex += (
                    f"{label} & {pct(b)} & {pct(a)} "
                    f"& {d * 100:+.1f}pp \\\\\n"
                )
            elif fmt == "int":
                latex += (
                    f"{label} & {int(b)} & {int(a)} & {int(d):+d} \\\\\n"
                )
            elif fmt == "float":
                latex += (
                    f"{label} & {b:.2f} & {a:.2f} & {d:+.2f} \\\\\n"
                )
            elif fmt == "float4":
                latex += (
                    f"{label} & {b:.4f} & {a:.4f} & {d:+.4f} \\\\\n"
                )

        latex += (
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        return latex

    @staticmethod
    def ablation_table(results: List[Dict[str, Any]]) -> str:
        """Generate LaTeX table for ablation study results."""
        pct = LaTeXTableGenerator._pct

        latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Ablation Study: Information Quality pro Variante}\n"
            "\\label{tab:ablation-info-quality}\n"
            "\\begin{tabular}{lrrrrr}\n"
            "\\toprule\n"
            "\\textbf{Variante} & \\textbf{Preservation} & \\textbf{Removal} "
            "& \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\\n"
            "\\midrule\n"
        )

        for r in results:
            name = r.get("variant_name", "?")
            pres = r.get("supporting_preservation_rate", 0)
            rem = r.get("distractor_removal_rate", 0)
            prec = r.get("information_precision", 0)
            rec = r.get("information_recall", 0)
            f1 = r.get("information_f1", 0)

            latex += (
                f"{name} & {pct(pres)} & {pct(rem)} & {pct(prec)} "
                f"& {pct(rec)} & {pct(f1)} \\\\\n"
            )

        latex += (
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        return latex

    @staticmethod
    def annotation_table(agreement: Dict[str, Any]) -> str:
        """Generate LaTeX table for annotation agreement results."""
        cm = agreement.get("confusion_matrix", {})
        pct = LaTeXTableGenerator._pct

        latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Manuelle Annotation: Übereinstimmung System vs.\\ Mensch}\n"
            "\\label{tab:annotation-agreement}\n"
            "\\begin{tabular}{lrr}\n"
            "\\toprule\n"
            " & \\textbf{Mensch: Korrekt} & \\textbf{Mensch: Inkorrekt} \\\\\n"
            "\\midrule\n"
        )

        latex += (
            f"System: Akzeptiert & {cm.get('tp', 0)} (TP) "
            f"& {cm.get('fp', 0)} (FP) \\\\\n"
        )
        latex += (
            f"System: Abgelehnt & {cm.get('fn', 0)} (FN) "
            f"& {cm.get('tn', 0)} (TN) \\\\\n"
        )

        latex += "\\midrule\n"
        latex += "\\multicolumn{3}{l}{"
        latex += (
            f"Precision: {pct(agreement.get('precision', 0))}, "
            f"Recall: {pct(agreement.get('recall', 0))}, "
            f"F1: {pct(agreement.get('f1', 0))}, "
            f"Cohen's $\\kappa$: {agreement.get('cohens_kappa', 0):.3f}"
        )
        latex += "} \\\\\n"
        latex += (
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        return latex
