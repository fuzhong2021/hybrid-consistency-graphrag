#!/usr/bin/env python3
# evaluation/metrics/significance.py
"""
Statistische Signifikanztests für Modellvergleiche.

Implementiert:
- McNemar's Test: Vergleich von zwei Klassifizierern
- Bootstrap Confidence Intervals: Konfidenzintervalle für Metriken
- Paired Permutation Test: Non-parametrischer Signifikanztest

Wissenschaftliche Referenzen:
- McNemar (1947): Note on the Sampling Error of the Difference Between Correlated Proportions
- Efron & Tibshirani (1993): An Introduction to the Bootstrap
- Dror et al. (2018): Deep Dominance - How to Properly Compare Deep Neural Models
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


def mcnemar_test(
    predictions_a: List[bool],
    predictions_b: List[bool],
    ground_truth: List[bool],
    exact: bool = True
) -> Dict[str, Any]:
    """
    McNemar's Test zum Vergleich zweier Klassifizierer.

    Testet ob die Unterschiede zwischen zwei Klassifizierern
    statistisch signifikant sind.

    Null-Hypothese: Beide Modelle haben die gleiche Fehlerrate.

    Args:
        predictions_a: Vorhersagen von Modell A
        predictions_b: Vorhersagen von Modell B
        ground_truth: Wahre Labels
        exact: True für exakten Test (langsamer, genauer)

    Returns:
        Dict mit Statistik, p-value und Interpretation

    Reference:
        McNemar (1947): Note on the Sampling Error of the Difference Between Correlated Proportions
    """
    pred_a = np.asarray(predictions_a)
    pred_b = np.asarray(predictions_b)
    truth = np.asarray(ground_truth)

    # Korrektheit berechnen
    correct_a = (pred_a == truth)
    correct_b = (pred_b == truth)

    # Kontingenztabelle
    # b: A richtig, B falsch
    # c: A falsch, B richtig
    b = np.sum(correct_a & ~correct_b)  # A correct, B incorrect
    c = np.sum(~correct_a & correct_b)  # A incorrect, B correct

    # McNemar's Test
    if b + c == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b": int(b),
            "c": int(c),
            "significant_0.05": False,
            "significant_0.01": False,
            "interpretation": "Keine Unterschiede zwischen den Modellen",
        }

    # scipy.stats.binom_test wurde in scipy>=1.12 entfernt; binomtest ist der
    # offizielle Nachfolger. Wir versuchen erst den neuen, dann den alten Namen.
    try:
        from scipy.stats import chi2
        try:
            from scipy.stats import binomtest as _binomtest_obj  # scipy >= 1.7
            def _two_sided_binom_p(k: int, n: int) -> float:
                return float(_binomtest_obj(k=k, n=n, p=0.5, alternative="two-sided").pvalue)
        except ImportError:
            from scipy.stats import binom_test as _binom_test  # scipy < 1.12
            def _two_sided_binom_p(k: int, n: int) -> float:
                return float(_binom_test(k, n=n, p=0.5))
    except ImportError:
        logger.warning("scipy nicht installiert - vereinfachter McNemar Test")
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
        p_value = np.exp(-statistic / 2)  # Sehr grobe Approximation
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "b": int(b),
            "c": int(c),
            "significant_0.05": p_value < 0.05,
            "significant_0.01": p_value < 0.01,
            "interpretation": "scipy nicht verfügbar - approximierte Werte",
        }

    if exact:
        # Exakter binomischer Test: P(X=k oder extremer | n=b+c, p=0.5)
        p_value = _two_sided_binom_p(int(min(b, c)), int(b + c))
        statistic = float(b - c)
    else:
        # Chi-Quadrat Approximation mit Kontinuitätskorrektur
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)

    # Interpretation
    if p_value < 0.001:
        interpretation = "Sehr signifikant (p < 0.001): Modelle unterscheiden sich stark"
    elif p_value < 0.01:
        interpretation = "Hoch signifikant (p < 0.01): Klare Unterschiede"
    elif p_value < 0.05:
        interpretation = "Signifikant (p < 0.05): Messbare Unterschiede"
    else:
        interpretation = "Nicht signifikant (p >= 0.05): Kein bedeutsamer Unterschied"

    # Welches Modell ist besser?
    if b > c:
        better_model = "A"
    elif c > b:
        better_model = "B"
    else:
        better_model = "Gleichwertig"

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "b": int(b),  # A richtig, B falsch
        "c": int(c),  # A falsch, B richtig
        "better_model": better_model,
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
        "interpretation": interpretation,
    }


def bootstrap_confidence_interval(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    y_true: List[bool],
    y_pred: List[bool],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Berechnet Bootstrap-Konfidenzintervall für eine Metrik.

    Args:
        metric_func: Funktion die (y_true, y_pred) → Metrik berechnet
        y_true: Wahre Labels
        y_pred: Vorhersagen
        n_bootstrap: Anzahl der Bootstrap-Samples
        confidence_level: Konfidenzniveau (z.B. 0.95 für 95%)
        seed: Random Seed für Reproduzierbarkeit

    Returns:
        Dict mit Punktschätzung, CI, und Standardfehler

    Reference:
        Efron & Tibshirani (1993): An Introduction to the Bootstrap
    """
    if seed is not None:
        np.random.seed(seed)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)

    # Punktschätzung
    point_estimate = metric_func(y_true, y_pred)

    # Bootstrap-Samples
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Sample mit Zurücklegen
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        try:
            estimate = metric_func(sample_true, sample_pred)
            bootstrap_estimates.append(estimate)
        except Exception:
            continue

    bootstrap_estimates = np.array(bootstrap_estimates)

    if len(bootstrap_estimates) == 0:
        return {
            "point_estimate": float(point_estimate),
            "ci_lower": float(point_estimate),
            "ci_upper": float(point_estimate),
            "std_error": 0.0,
            "n_successful_bootstrap": 0,
        }

    # Konfidenzintervall (Percentile-Methode)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)

    # Standardfehler
    std_error = np.std(bootstrap_estimates)

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_width": float(ci_upper - ci_lower),
        "std_error": float(std_error),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "n_successful_bootstrap": len(bootstrap_estimates),
    }


def paired_permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 10000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Paired Permutation Test zum Vergleich zweier Systeme.

    Non-parametrischer Test der keine Verteilungsannahmen macht.
    Gut für kleine Stichproben und nicht-normale Verteilungen.

    Args:
        scores_a: Scores von System A (z.B. F1 pro Sample)
        scores_b: Scores von System B
        n_permutations: Anzahl der Permutationen
        seed: Random Seed

    Returns:
        Dict mit p-value und Interpretation

    Reference:
        Dror et al. (2018): Deep Dominance
    """
    if seed is not None:
        np.random.seed(seed)

    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    # Differenzen
    diff = scores_a - scores_b
    observed_mean_diff = np.mean(diff)

    # Permutation Test
    count_extreme = 0
    for _ in range(n_permutations):
        # Zufällige Vorzeichen für Differenzen
        signs = np.random.choice([-1, 1], size=len(diff))
        permuted_diff = diff * signs
        permuted_mean = np.mean(permuted_diff)

        if abs(permuted_mean) >= abs(observed_mean_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    # Interpretation
    if p_value < 0.001:
        interpretation = "Sehr signifikant (p < 0.001)"
    elif p_value < 0.01:
        interpretation = "Hoch signifikant (p < 0.01)"
    elif p_value < 0.05:
        interpretation = "Signifikant (p < 0.05)"
    else:
        interpretation = "Nicht signifikant (p >= 0.05)"

    return {
        "observed_mean_diff": float(observed_mean_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "count_extreme": count_extreme,
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
        "better_system": "A" if observed_mean_diff > 0 else "B",
        "interpretation": interpretation,
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Bonferroni-Korrektur für multiple Tests.

    Korrigiert den Signifikanzlevel für mehrere simultane Tests
    um die Family-Wise Error Rate (FWER) zu kontrollieren.

    Args:
        p_values: Liste der p-values
        alpha: Gewünschtes Signifikanzniveau

    Returns:
        Dict mit korrigierten Ergebnissen
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    return {
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "n_tests": n_tests,
        "significant_tests": [i for i, p in enumerate(p_values) if p < corrected_alpha],
        "n_significant": sum(1 for p in p_values if p < corrected_alpha),
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Holm-Bonferroni-Korrektur (weniger konservativ als Bonferroni).

    Step-down Prozedur die mehr Power hat als die klassische
    Bonferroni-Korrektur.

    Args:
        p_values: Liste der p-values
        alpha: Gewünschtes Signifikanzniveau

    Returns:
        Dict mit korrigierten Ergebnissen
    """
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = np.array(p_values)[sorted_indices]

    significant = []
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_pvalues)):
        corrected_alpha = alpha / (n_tests - i)
        if p < corrected_alpha:
            significant.append(int(idx))
        else:
            break

    return {
        "original_alpha": alpha,
        "n_tests": n_tests,
        "significant_tests": significant,
        "n_significant": len(significant),
    }


# ===========================================================================
# Convenience Functions
# ===========================================================================

def accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Berechnet Accuracy."""
    return np.mean(y_true == y_pred)


def f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Berechnet F1 Score."""
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_significance_tests(
    predictions: Dict[str, List[bool]],
    ground_truth: List[bool],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Führt alle Signifikanztests für mehrere Modelle durch.

    Args:
        predictions: Dict mit Modellname → Vorhersagen
        ground_truth: Wahre Labels
        alpha: Signifikanzniveau

    Returns:
        Umfassendes Ergebnis mit allen Tests
    """
    model_names = list(predictions.keys())
    results = {
        "pairwise_mcnemar": {},
        "bootstrap_ci": {},
        "summary": {},
    }

    # Paarweise McNemar Tests
    p_values = []
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            test_name = f"{model_a}_vs_{model_b}"
            test_result = mcnemar_test(
                predictions[model_a],
                predictions[model_b],
                ground_truth
            )
            results["pairwise_mcnemar"][test_name] = test_result
            p_values.append(test_result["p_value"])

    # Bootstrap CIs für jedes Modell
    for model_name, preds in predictions.items():
        ci_result = bootstrap_confidence_interval(
            accuracy_metric,
            ground_truth,
            preds,
            n_bootstrap=1000
        )
        results["bootstrap_ci"][model_name] = ci_result

    # Multiple Testing Correction
    if p_values:
        results["bonferroni"] = bonferroni_correction(p_values, alpha)
        results["holm_bonferroni"] = holm_bonferroni_correction(p_values, alpha)

    # Summary
    accuracies = {
        name: accuracy_metric(np.array(ground_truth), np.array(preds))
        for name, preds in predictions.items()
    }
    best_model = max(accuracies, key=accuracies.get)
    results["summary"] = {
        "n_models": len(model_names),
        "n_samples": len(ground_truth),
        "accuracies": accuracies,
        "best_model": best_model,
        "best_accuracy": accuracies[best_model],
    }

    return results


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Führt Signifikanztests durch")
    parser.add_argument("--predictions", type=str, required=True,
                       help="JSON mit Vorhersagen (Dict: model -> predictions)")
    parser.add_argument("--ground-truth", type=str, required=True,
                       help="JSON mit Ground Truth")
    parser.add_argument("--output", type=str, default=None,
                       help="Output-Pfad für Ergebnisse")

    args = parser.parse_args()

    # Lade Daten
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)

    # Tests durchführen
    results = compute_all_significance_tests(predictions, ground_truth)

    # Ausgabe
    print("\n" + "=" * 60)
    print("SIGNIFIKANZTESTS")
    print("=" * 60)

    print("\nPaarweise McNemar Tests:")
    for test_name, test_result in results["pairwise_mcnemar"].items():
        sig = "*" if test_result["significant_0.05"] else ""
        print(f"  {test_name}: p={test_result['p_value']:.4f} {sig}")

    print("\nBootstrap 95% CIs (Accuracy):")
    for model_name, ci in results["bootstrap_ci"].items():
        print(f"  {model_name}: {ci['point_estimate']:.3f} "
              f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    print(f"\nBestes Modell: {results['summary']['best_model']} "
          f"(Accuracy: {results['summary']['best_accuracy']:.3f})")

    # Speichern
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nErgebnisse gespeichert: {args.output}")
