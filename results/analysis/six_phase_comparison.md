# 6-Phasen-Evaluation — n=3 (Sanity) vs n=50 (belastbar)

Wilson-95 %-CI für Precision/Recall/Accuracy, Bootstrap-95 %-CI für F1.

## Comparison-Metriken (Konfidenz-Effekte)

| Metrik | n=3 | n=50 |
|---|---:|---:|
| `missing_source_penalty_effectiveness` | 0.300 | 0.565 |
| `contradiction_detection_f1` | 1.000 | 0.874 |
| `cross_question_detection_f1` | 1.000 | 0.851 |
| `fake_source_detection_effectiveness` | 0.412 | 0.404 |

## Phasen-Metriken mit Konfidenzintervallen

# Konfidenzintervalle (Phase 1.2)

Generiert aus aggregierten TP/FP/TN/FN-Counts mit Wilson-Score (Precision/Recall/Accuracy) und Percentile-Bootstrap (F1, n=1000, Seed=42). 95 %-CIs.

| Konfiguration | n | Precision (95 % CI) | Recall (95 % CI) | F1 (95 % CI) | Accuracy (95 % CI) |
|---|---:|---|---|---|---|
| `n=50_scaled/phase1_baseline` | 96 | 0.000 [0.000, 0.149] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.771 [0.677, 0.844] |
| `n=50_scaled/phase2_correct_with_source` | 96 | 0.000 [0.000, 0.242] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.875 [0.794, 0.927] |
| `n=50_scaled/phase3_correct_without_source` | 96 | 0.000 [0.000, 0.073] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.490 [0.392, 0.588] |
| `n=50_scaled/phase4_distractor_contradictions` | 98 | 1.000 [0.952, 1.000] | 0.776 [0.683, 0.847] | 0.874 [0.819, 0.923] | 0.776 [0.683, 0.847] |
| `n=50_scaled/phase5_cross_question_confusion` | 50 | 1.000 [0.906, 1.000] | 0.740 [0.604, 0.841] | 0.851 [0.765, 0.925] | 0.740 [0.604, 0.841] |
| `n=50_scaled/phase6_fake_source_attack` | 50 | 1.000 [0.851, 1.000] | 0.440 [0.312, 0.577] | 0.611 [0.462, 0.734] | 0.440 [0.312, 0.577] |

> **Methodik:** Da nur aggregierte Konfusionsmatrix-Counts vorliegen, werden per-example Labels aus den Counts synthetisiert (TP→(1,1), FP→(0,1), TN→(0,0), FN→(1,0)). Bootstrap-Resampling auf diesem synthetisierten Array liefert mathematisch identische Verteilungen wie Resampling auf den Originalbeispielen, da F1/Precision/Recall/Accuracy reine Funktionen der Konfusionsmatrix sind.


### Historischer Sanity-Check (n=3)

# Konfidenzintervalle (Phase 1.2)

Generiert aus aggregierten TP/FP/TN/FN-Counts mit Wilson-Score (Precision/Recall/Accuracy) und Percentile-Bootstrap (F1, n=1000, Seed=42). 95 %-CIs.

| Konfiguration | n | Precision (95 % CI) | Recall (95 % CI) | F1 (95 % CI) | Accuracy (95 % CI) |
|---|---:|---|---|---|---|
| `n=3_sanity/phase2_correct_with_source` | 6 | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.833 [0.436, 0.970] |
| `n=3_sanity/phase3_correct_without_source` | 6 | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.833 [0.436, 0.970] |
| `n=3_sanity/phase4_distractor_contradictions` | 6 | 1.000 [0.610, 1.000] | 1.000 [0.610, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.610, 1.000] |
| `n=3_sanity/phase5_cross_question_confusion` | 3 | 1.000 [0.439, 1.000] | 1.000 [0.439, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.439, 1.000] |
| `n=3_sanity/phase6_fake_source_attack` | 3 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.561] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.561] |

> **Methodik:** Da nur aggregierte Konfusionsmatrix-Counts vorliegen, werden per-example Labels aus den Counts synthetisiert (TP→(1,1), FP→(0,1), TN→(0,0), FN→(1,0)). Bootstrap-Resampling auf diesem synthetisierten Array liefert mathematisch identische Verteilungen wie Resampling auf den Originalbeispielen, da F1/Precision/Recall/Accuracy reine Funktionen der Konfusionsmatrix sind.

