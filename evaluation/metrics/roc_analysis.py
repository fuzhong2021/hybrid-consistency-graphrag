#!/usr/bin/env python3
# evaluation/metrics/roc_analysis.py
"""
ROC und Precision-Recall Analyse für Konsistenzprüfung.

Implementiert:
- ROC-AUC: Area Under the ROC Curve
- PR-AUC: Area Under the Precision-Recall Curve
- Optimale Threshold-Bestimmung
- Visualisierungen

Wissenschaftliche Referenzen:
- Fawcett (2006): An Introduction to ROC Analysis
- Davis & Goadrich (2006): The Relationship Between Precision-Recall and ROC Curves
- Saito & Rehmsmeier (2015): The Precision-Recall Plot Is More Informative Than the ROC Plot
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet ROC-Kurve.

    Args:
        y_true: Wahre Labels (0/1)
        y_scores: Vorhersage-Scores (Wahrscheinlichkeiten)
        n_thresholds: Anzahl der Schwellenwerte

    Returns:
        Tuple von (fpr, tpr, thresholds)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Thresholds
    thresholds = np.linspace(0, 1, n_thresholds)

    fpr = []
    tpr = []

    n_positives = np.sum(y_true)
    n_negatives = len(y_true) - n_positives

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        tpr.append(tp / n_positives if n_positives > 0 else 0)
        fpr.append(fp / n_negatives if n_negatives > 0 else 0)

    return np.array(fpr), np.array(tpr), thresholds


def compute_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet Precision-Recall-Kurve.

    Args:
        y_true: Wahre Labels (0/1)
        y_scores: Vorhersage-Scores
        n_thresholds: Anzahl der Schwellenwerte

    Returns:
        Tuple von (precision, recall, thresholds)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    thresholds = np.linspace(0, 1, n_thresholds)

    precision = []
    recall = []

    n_positives = np.sum(y_true)

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / n_positives if n_positives > 0 else 0

        precision.append(prec)
        recall.append(rec)

    return np.array(precision), np.array(recall), thresholds


def compute_roc_auc(
    y_true: List[bool],
    y_scores: List[float]
) -> float:
    """
    Berechnet Area Under the ROC Curve (AUC-ROC).

    AUC-ROC misst die Fähigkeit des Modells, positive und negative
    Klassen zu unterscheiden.

    Args:
        y_true: Wahre Labels
        y_scores: Vorhersage-Scores (Wahrscheinlichkeiten)

    Returns:
        AUC-ROC Wert (0.5 = random, 1.0 = perfekt)

    Reference:
        Fawcett (2006): An Introduction to ROC Analysis
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores)

    # ROC-Kurve berechnen
    fpr, tpr, _ = compute_roc_curve(y_true, y_scores)

    # Sortiere nach FPR für korrekte AUC-Berechnung
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    # Trapezregel für AUC
    auc = np.trapz(tpr_sorted, fpr_sorted)

    return float(auc)


def compute_pr_auc(
    y_true: List[bool],
    y_scores: List[float]
) -> float:
    """
    Berechnet Area Under the Precision-Recall Curve (AUC-PR).

    AUC-PR ist besonders wichtig bei unbalancierten Datasets,
    da es nicht durch eine große Anzahl von True Negatives
    verzerrt wird.

    Args:
        y_true: Wahre Labels
        y_scores: Vorhersage-Scores

    Returns:
        AUC-PR Wert

    Reference:
        Saito & Rehmsmeier (2015): The Precision-Recall Plot Is More Informative
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores)

    precision, recall, _ = compute_pr_curve(y_true, y_scores)

    # Sortiere nach Recall für korrekte AUC-Berechnung
    sorted_indices = np.argsort(recall)
    recall_sorted = recall[sorted_indices]
    precision_sorted = precision[sorted_indices]

    # Trapezregel für AUC
    auc = np.trapz(precision_sorted, recall_sorted)

    return float(auc)


def compute_optimal_threshold(
    y_true: List[bool],
    y_scores: List[float],
    method: str = "youden"
) -> Dict[str, Any]:
    """
    Bestimmt den optimalen Schwellenwert.

    Methods:
        "youden": Maximiert Youden's J (TPR - FPR)
        "f1": Maximiert F1-Score
        "cost": Minimiert Kosten (custom weights)
        "balanced": Minimiert |TPR - TNR|

    Args:
        y_true: Wahre Labels
        y_scores: Vorhersage-Scores
        method: Optimierungsmethode

    Returns:
        Dict mit optimalem Threshold und Metriken
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores)

    thresholds = np.linspace(0, 1, 100)
    best_thresh = 0.5
    best_metric = -float('inf')
    best_metrics = {}

    n_positives = np.sum(y_true)
    n_negatives = len(y_true) - n_positives

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        tpr = tp / n_positives if n_positives > 0 else 0
        fpr = fp / n_negatives if n_negatives > 0 else 0
        tnr = tn / n_negatives if n_negatives > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Metrik basierend auf Methode
        if method == "youden":
            metric = tpr - fpr  # Youden's J Index
        elif method == "f1":
            metric = f1
        elif method == "balanced":
            metric = -abs(tpr - tnr)  # Minimiere Differenz
        else:
            metric = tpr - fpr  # Default: Youden

        if metric > best_metric:
            best_metric = metric
            best_thresh = thresh
            best_metrics = {
                "threshold": float(thresh),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "tnr": float(tnr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "youden_j": float(tpr - fpr),
            }

    return {
        "optimal_threshold": best_thresh,
        "method": method,
        "metrics_at_threshold": best_metrics,
    }


def compute_roc_metrics_per_stage(
    stage_results: Dict[str, Dict[str, List]],
    ground_truth: List[bool]
) -> Dict[str, Dict[str, float]]:
    """
    Berechnet ROC-Metriken für jede Validierungsstufe.

    Args:
        stage_results: Dict mit Stage-Namen → {"scores": [...], "predictions": [...]}
        ground_truth: Wahre Labels

    Returns:
        Dict mit Metriken pro Stage
    """
    results = {}

    for stage_name, stage_data in stage_results.items():
        scores = stage_data.get("scores", [])
        if not scores:
            continue

        roc_auc = compute_roc_auc(ground_truth, scores)
        pr_auc = compute_pr_auc(ground_truth, scores)
        optimal = compute_optimal_threshold(ground_truth, scores)

        results[stage_name] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": optimal["optimal_threshold"],
            "f1_at_optimal": optimal["metrics_at_threshold"]["f1"],
        }

    return results


def plot_roc_curve(
    y_true: List[bool],
    y_scores: List[float],
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show_optimal: bool = True
) -> Optional[Any]:
    """
    Erstellt ROC-Kurve Plot.

    Args:
        y_true: Wahre Labels
        y_scores: Vorhersage-Scores
        title: Plot-Titel
        save_path: Pfad zum Speichern
        show_optimal: Zeige optimalen Punkt

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib nicht installiert")
        return None

    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores)

    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    auc = compute_roc_auc(y_true.tolist(), y_scores.tolist())

    fig, ax = plt.subplots(figsize=(8, 8))

    # ROC-Kurve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')

    # Optimaler Punkt
    if show_optimal:
        optimal = compute_optimal_threshold(y_true.tolist(), y_scores.tolist())
        opt_thresh = optimal["optimal_threshold"]
        opt_metrics = optimal["metrics_at_threshold"]
        ax.scatter([opt_metrics["fpr"]], [opt_metrics["tpr"]],
                  c='red', s=100, marker='o', zorder=5,
                  label=f'Optimal (thresh={opt_thresh:.2f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve gespeichert: {save_path}")

    return fig


def plot_precision_recall_curve(
    y_true: List[bool],
    y_scores: List[float],
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Erstellt Precision-Recall-Kurve Plot.

    Args:
        y_true: Wahre Labels
        y_scores: Vorhersage-Scores
        title: Plot-Titel
        save_path: Pfad zum Speichern

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib nicht installiert")
        return None

    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores)

    precision, recall, thresholds = compute_pr_curve(y_true, y_scores)
    auc = compute_pr_auc(y_true.tolist(), y_scores.tolist())

    # Baseline (Anteil positiver Klasse)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(8, 8))

    # PR-Kurve
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC = {auc:.3f})')
    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"PR curve gespeichert: {save_path}")

    return fig


def plot_multi_stage_roc(
    stage_data: Dict[str, Tuple[List[bool], List[float]]],
    title: str = "ROC Curves per Validation Stage",
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Erstellt ROC-Kurven für mehrere Validierungsstufen.

    Args:
        stage_data: Dict mit Stage-Namen → (y_true, y_scores)
        title: Plot-Titel
        save_path: Pfad zum Speichern

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib nicht installiert")
        return None

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, (stage_name, (y_true, y_scores)) in enumerate(stage_data.items()):
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores)

        fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
        auc = compute_roc_auc(y_true.tolist(), y_scores.tolist())

        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2,
               label=f'{stage_name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Multi-stage ROC gespeichert: {save_path}")

    return fig


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ROC/PR Analyse")
    parser.add_argument("--predictions", type=str, required=True,
                       help="JSON mit y_true und y_scores")
    parser.add_argument("--roc-plot", type=str, default=None,
                       help="Pfad für ROC Plot")
    parser.add_argument("--pr-plot", type=str, default=None,
                       help="Pfad für PR Plot")
    parser.add_argument("--output", type=str, default=None,
                       help="Pfad für JSON-Ergebnisse")

    args = parser.parse_args()

    # Lade Daten
    with open(args.predictions, 'r') as f:
        data = json.load(f)

    y_true = data.get("y_true", [])
    y_scores = data.get("y_scores", data.get("y_prob", []))

    # Metriken berechnen
    roc_auc = compute_roc_auc(y_true, y_scores)
    pr_auc = compute_pr_auc(y_true, y_scores)
    optimal = compute_optimal_threshold(y_true, y_scores)

    print("\n" + "=" * 50)
    print("ROC/PR ANALYSE")
    print("=" * 50)
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Optimaler Threshold: {optimal['optimal_threshold']:.3f}")
    print(f"  F1 bei optimalem Threshold: {optimal['metrics_at_threshold']['f1']:.3f}")

    # Plots erstellen
    if args.roc_plot:
        plot_roc_curve(y_true, y_scores, save_path=args.roc_plot)

    if args.pr_plot:
        plot_precision_recall_curve(y_true, y_scores, save_path=args.pr_plot)

    # Ergebnisse speichern
    if args.output:
        results = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": optimal,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nErgebnisse gespeichert: {args.output}")
