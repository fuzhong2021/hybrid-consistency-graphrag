# evaluation/baselines/__init__.py
"""
Baseline-Implementierungen für wissenschaftlichen Vergleich.

Baselines:
- RandomBaseline: Akzeptiert Triples mit konfigurierbarer Rate
- RulesOnlyBaseline: Nur regelbasierte Validierung (Stage 1)
- EmbeddingOnlyBaseline: Nur embedding-basierte Validierung (Stage 2)
- LLMOnlyBaseline: Direkt LLM-Validierung ohne Vorfilterung
- NLIBaseline: NLI-basierte Fact Verification (~44M Parameter)

Wissenschaftliche Referenz:
- Thorne et al. (2018): FEVER - Fact Extraction and VERification
- Akoglu et al. (2015): Graph Based Anomaly Detection and Description
- Bowman et al. (2015): A large annotated corpus for learning NLI
- He et al. (2021): DeBERTa: Decoding-enhanced BERT with Disentangled Attention
"""

from evaluation.baselines.random_baseline import RandomBaseline
from evaluation.baselines.rules_only_baseline import RulesOnlyBaseline
from evaluation.baselines.embedding_only_baseline import EmbeddingOnlyBaseline
from evaluation.baselines.llm_only_baseline import LLMOnlyBaseline
from evaluation.baselines.nli_baseline import NLIBaseline

__all__ = [
    "RandomBaseline",
    "RulesOnlyBaseline",
    "EmbeddingOnlyBaseline",
    "LLMOnlyBaseline",
    "NLIBaseline",
]
