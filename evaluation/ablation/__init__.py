# evaluation/ablation/__init__.py
"""
Ablation Study für Konsistenzmodul-Komponenten.

Systematische Analyse welche Komponenten wie viel zum
Gesamtergebnis beitragen.

Varianten:
- Ohne Stage 2 (Embedding)
- Ohne Stage 3 (LLM)
- Ohne Provenance-Boost
- Ohne Source Verification
- Ohne Semantischen Trigger
- Verschiedene Confidence-Kombinationsmethoden
"""

from evaluation.ablation.ablation_study import (
    AblationStudy,
    AblationVariant,
    run_ablation_study,
)

__all__ = [
    "AblationStudy",
    "AblationVariant",
    "run_ablation_study",
]
