#!/usr/bin/env python3
# evaluation/streaming/__init__.py
"""
Streaming Evaluation Module für wissenschaftlich vollständige KG-Konsistenzprüfung.

═══════════════════════════════════════════════════════════════════════════════
WISSENSCHAFTLICHE GRUNDLAGE (21 Referenzen)
═══════════════════════════════════════════════════════════════════════════════

Fact Verification:
- Thorne et al. (2018): FEVER Dataset - NAACL
- Bowman et al. (2015): SNLI - EMNLP
- Williams et al. (2018): MultiNLI - NAACL

Entity Resolution:
- Lairgi et al. (2024): iText2KG (α=0.6 Name, 0.4 Embedding)
- Shen et al. (2015): Entity Linking Survey - TKDE

Temporal KG:
- Leblay & Chekol (2018): Temporal KG Validity - WWW
- Lacroix et al. (2020): Tensor Decomposition for TKG - ICLR
- García-Durán et al. (2018): Sequence Encoders for TKG - EMNLP

Negative Knowledge:
- Razniewski et al. (2016): Negative Statements - AKBC
- Arnaout et al. (2021, 2022): Negative Knowledge - JAIR, WWW

Uncertainty:
- Safavi et al. (2020): Calibration in KG Embeddings - EMNLP
- Chen et al. (2019): Uncertain Knowledge Graphs - AAAI

Source Quality:
- Dong et al. (2015): Knowledge Vault - VLDB
- Pasternack & Roth (2013): Source Credibility - WWW

Schema/Ontology:
- Paulheim (2017): KG Refinement Survey - SWJ
- Euzenat & Shvaiko (2013): Ontology Matching - Springer
- Hobbs (1985): Granularity - IJCAI

Streaming KG:
- Heist & Paulheim (2019): Streaming KG Construction

═══════════════════════════════════════════════════════════════════════════════
10-KATEGORIEN KONFLIKT-TAXONOMIE
═══════════════════════════════════════════════════════════════════════════════

1. FACTUAL        - Direkte faktische Widersprüche
2. TEMPORAL       - Zeitlich getrennte Aussagen
3. GRANULARITY    - Verschiedene Abstraktionsebenen
4. ENTITY_VARIANT - Koreferenz/Entity Resolution
5. IMPLICIT       - Widersprüche durch Inferenz
6. NEGATION       - Direkte Verneinung
7. MODALITY       - Unterschiedliche Gewissheitsgrade
8. SOURCE_QUALITY - Quellen-Vertrauenswürdigkeit
9. SCHEMA         - Schema-Heterogenität
10. NUMERICAL     - Numerische Präzision

═══════════════════════════════════════════════════════════════════════════════
"""

# Core Components
from .triple_generator import (
    FEVERTripleGenerator,
    AnnotatedTriple,
    TripleCategory,
)
from .entity_variant_generator import (
    EntityVariantGenerator,
    EntityVariant,
)
from .cross_doc_generator import CrossDocConflictGenerator
from .shuffle_strategy import (
    StreamingShuffler,
    ShuffleStrategy,
    create_shuffler,
)

# Scientific Taxonomy
from .conflict_taxonomy import (
    ConflictType,
    GroundTruthAction,
    DetectionMethod,
    ConflictCategory,
    ScientificReference,
    CONFLICT_TAXONOMY,
    get_all_conflict_types,
    get_category,
    get_detection_methods,
    get_ground_truth_action,
    get_references,
    get_all_references,
    print_taxonomy_summary,
)

# Comprehensive Generator
from .comprehensive_conflict_generator import (
    ConflictAnnotation,
    AnnotatedConflictTriple,
    ConflictGenerator,
    FactualConflictGenerator,
    TemporalConflictGenerator,
    GranularityConflictGenerator,
    EntityVariantConflictGenerator,
    ImplicitConflictGenerator,
    NegationConflictGenerator,
    ModalityConflictGenerator,
    SourceQualityConflictGenerator,
    SchemaConflictGenerator,
    NumericalConflictGenerator,
    ComprehensiveConflictGenerator,
)

__all__ = [
    # Triple Generation
    "FEVERTripleGenerator",
    "AnnotatedTriple",
    "TripleCategory",

    # Entity Variants
    "EntityVariantGenerator",
    "EntityVariant",

    # Cross-Doc Conflicts
    "CrossDocConflictGenerator",

    # Shuffle
    "StreamingShuffler",
    "ShuffleStrategy",
    "create_shuffler",

    # Taxonomy
    "ConflictType",
    "GroundTruthAction",
    "DetectionMethod",
    "ConflictCategory",
    "ScientificReference",
    "CONFLICT_TAXONOMY",
    "get_all_conflict_types",
    "get_category",
    "get_detection_methods",
    "get_ground_truth_action",
    "get_references",
    "get_all_references",
    "print_taxonomy_summary",

    # Comprehensive Generator
    "ConflictAnnotation",
    "AnnotatedConflictTriple",
    "ComprehensiveConflictGenerator",
    "FactualConflictGenerator",
    "TemporalConflictGenerator",
    "GranularityConflictGenerator",
    "EntityVariantConflictGenerator",
    "ImplicitConflictGenerator",
    "NegationConflictGenerator",
    "ModalityConflictGenerator",
    "SourceQualityConflictGenerator",
    "SchemaConflictGenerator",
    "NumericalConflictGenerator",
]

__version__ = "1.0.0"
__author__ = "Masterarbeit GraphRAG Konsistenzprüfung"
