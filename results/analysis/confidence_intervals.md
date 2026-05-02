# Konfidenzintervalle (Phase 1.2)

Generiert aus aggregierten TP/FP/TN/FN-Counts mit Wilson-Score (Precision/Recall/Accuracy) und Percentile-Bootstrap (F1, n=1000, Seed=42). 95 %-CIs.

| Konfiguration | n | Precision (95 % CI) | Recall (95 % CI) | F1 (95 % CI) | Accuracy (95 % CI) |
|---|---:|---|---|---|---|
| `multi_dataset/with_llm/fever` | 14 | 0.625 [0.306, 0.863] | 1.000 [0.566, 1.000] | 0.769 [0.444, 1.000] | 0.786 [0.524, 0.924] |
| `full_eval/with_llm/hotpotqa` | 27,955 | 0.769 [0.762, 0.776] | 0.725 [0.718, 0.732] | 0.747 [0.741, 0.752] | 0.744 [0.739, 0.749] |
| `full_eval/with_llm/fever` | 7,692 | 0.762 [0.749, 0.775] | 0.743 [0.730, 0.756] | 0.753 [0.742, 0.763] | 0.727 [0.717, 0.737] |
| `full_eval/with_llm/musique` | 11,893 | 0.490 [0.479, 0.500] | 0.878 [0.869, 0.887] | 0.629 [0.619, 0.638] | 0.588 [0.579, 0.597] |
| `full_eval/no_llm/hotpotqa` | 22,262 | 0.791 [0.783, 0.798] | 0.715 [0.706, 0.723] | 0.751 [0.744, 0.757] | 0.754 [0.748, 0.760] |
| `full_eval/no_llm/fever` | 4,707 | 0.797 [0.784, 0.809] | 0.989 [0.985, 0.992] | 0.883 [0.875, 0.891] | 0.818 [0.807, 0.829] |
| `baselines/random` | 127 | 0.507 [0.395, 0.618] | 0.552 [0.434, 0.665] | 0.529 [0.429, 0.617] | 0.480 [0.395, 0.567] |
| `baselines/rules_only` | 127 | 0.803 [0.700, 0.877] | 0.910 [0.818, 0.958] | 0.853 [0.790, 0.911] | 0.835 [0.760, 0.889] |
| `baselines/embedding_only` | 127 | 0.528 [0.441, 0.612] | 1.000 [0.946, 1.000] | 0.691 [0.612, 0.761] | 0.528 [0.441, 0.612] |
| `baselines/rules_embedding` | 127 | 0.803 [0.700, 0.877] | 0.910 [0.818, 0.958] | 0.853 [0.790, 0.911] | 0.835 [0.760, 0.889] |
| `baselines/rules_emb_no_penalty` | 127 | 0.803 [0.700, 0.877] | 0.910 [0.818, 0.958] | 0.853 [0.790, 0.911] | 0.835 [0.760, 0.889] |
| `baselines/rules_emb_no_srcver` | 127 | 0.803 [0.700, 0.877] | 0.910 [0.818, 0.958] | 0.853 [0.790, 0.911] | 0.835 [0.760, 0.889] |
| `baselines/full_system` | 127 | 0.807 [0.687, 0.889] | 0.687 [0.568, 0.785] | 0.742 [0.649, 0.825] | 0.748 [0.666, 0.815] |
| `musique_solo/musique` | 11,893 | 0.490 [0.479, 0.500] | 0.878 [0.869, 0.887] | 0.629 [0.619, 0.638] | 0.588 [0.579, 0.597] |

> **Methodik:** Da nur aggregierte Konfusionsmatrix-Counts vorliegen, werden per-example Labels aus den Counts synthetisiert (TP→(1,1), FP→(0,1), TN→(0,0), FN→(1,0)). Bootstrap-Resampling auf diesem synthetisierten Array liefert mathematisch identische Verteilungen wie Resampling auf den Originalbeispielen, da F1/Precision/Recall/Accuracy reine Funktionen der Konfusionsmatrix sind.
