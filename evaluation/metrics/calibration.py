#!/usr/bin/env python3
# evaluation/metrics/calibration.py
"""
Kalibrierungsmetriken für Konfidenz-Bewertung.

Misst ob die vorhergesagten Wahrscheinlichkeiten mit den tatsächlichen
Ergebnissen übereinstimmen.

Wissenschaftliche Referenzen:
- Guo et al. (2017): On Calibration of Modern Neural Networks
- Naeini et al. (2015): Obtaining Well Calibrated Probabilities Using Bayesian Binning
- Platt (1999): Probabilistic Outputs for Support Vector Machines

Metriken:
- ECE (Expected Calibration Error): Gewichtete Abweichung pro Bin
- Brier Score: Mittlere quadratische Abweichung der Wahrscheinlichkeiten
- Reliability Diagram: Visualisierung der Kalibrierung
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Berechnet Expected Calibration Error (ECE).

    ECE = Σ (|B_m|/n) * |acc(B_m) - conf(B_m)|

    wobei:
    - B_m ist der m-te Bin
    - acc(B_m) ist die Genauigkeit im Bin
    - conf(B_m) ist die durchschnittliche Konfidenz im Bin

    Args:
        y_true: Ground Truth Labels (0 oder 1)
        y_prob: Vorhergesagte Wahrscheinlichkeiten [0, 1]
        n_bins: Anzahl der Bins für die Kalibrierung

    Returns:
        ECE Wert (0 = perfekt kalibriert, 1 = komplett falsch)

    Reference:
        Guo et al. (2017): On Calibration of Modern Neural Networks
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return 0.0

    # Bin-Grenzen
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        # Finde Samples in diesem Bin
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Genauigkeit und Konfidenz im Bin
            accuracy_in_bin = y_true[in_bin].mean()
            confidence_in_bin = y_prob[in_bin].mean()

            # Gewichtete Abweichung
            ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Berechnet Maximum Calibration Error (MCE).

    MCE = max_m |acc(B_m) - conf(B_m)|

    Die maximale Abweichung über alle Bins.

    Args:
        y_true: Ground Truth Labels
        y_prob: Vorhergesagte Wahrscheinlichkeiten
        n_bins: Anzahl der Bins

    Returns:
        MCE Wert
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            confidence_in_bin = y_prob[in_bin].mean()
            error = np.abs(accuracy_in_bin - confidence_in_bin)
            max_error = max(max_error, error)

    return float(max_error)


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Berechnet Brier Score.

    Brier Score = (1/n) * Σ (p_i - y_i)²

    Misst die mittlere quadratische Abweichung zwischen
    vorhergesagten Wahrscheinlichkeiten und tatsächlichen Labels.

    Args:
        y_true: Ground Truth Labels (0 oder 1)
        y_prob: Vorhergesagte Wahrscheinlichkeiten [0, 1]

    Returns:
        Brier Score (0 = perfekt, 0.25 = random bei 50/50)

    Reference:
        Brier (1950): Verification of Forecasts Expressed in Terms of Probability
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean((y_prob - y_true) ** 2))


def compute_calibration_metrics(
    y_true: List[bool],
    y_prob: List[float],
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Berechnet alle Kalibrierungsmetriken.

    Args:
        y_true: Ground Truth Labels (True/False)
        y_prob: Vorhergesagte Wahrscheinlichkeiten [0, 1]
        n_bins: Anzahl der Bins

    Returns:
        Dict mit ECE, MCE, Brier Score und weiteren Metriken
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob)

    metrics = {
        "ECE": compute_ece(y_true, y_prob, n_bins),
        "MCE": compute_mce(y_true, y_prob, n_bins),
        "Brier": compute_brier_score(y_true, y_prob),
        "n_samples": len(y_true),
    }

    # Interpretationen
    if metrics["ECE"] < 0.05:
        metrics["ECE_interpretation"] = "excellent"
    elif metrics["ECE"] < 0.10:
        metrics["ECE_interpretation"] = "good"
    elif metrics["ECE"] < 0.15:
        metrics["ECE_interpretation"] = "moderate"
    else:
        metrics["ECE_interpretation"] = "poor"

    return metrics


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet Daten für Reliability Diagram.

    Args:
        y_true: Ground Truth Labels
        y_prob: Vorhergesagte Wahrscheinlichkeiten
        n_bins: Anzahl der Bins

    Returns:
        Tuple von (fraction_of_positives, mean_predicted_value, bin_counts)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        bin_counts[i] = in_bin.sum()

        if bin_counts[i] > 0:
            fraction_of_positives[i] = y_true[in_bin].mean()
            mean_predicted_value[i] = y_prob[in_bin].mean()
        else:
            fraction_of_positives[i] = np.nan
            mean_predicted_value[i] = bin_centers[i]

    return fraction_of_positives, mean_predicted_value, bin_counts


def plot_calibration_curve(
    y_true: List[bool],
    y_prob: List[float],
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Erstellt ein Reliability Diagram (Calibration Plot).

    Args:
        y_true: Ground Truth Labels
        y_prob: Vorhergesagte Wahrscheinlichkeiten
        n_bins: Anzahl der Bins
        title: Titel des Plots
        save_path: Pfad zum Speichern (optional)

    Returns:
        matplotlib Figure wenn verfügbar, sonst None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib nicht installiert - Plot nicht möglich")
        return None

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob)

    # Kalibrierungskurve berechnen
    fraction_of_positives, mean_predicted_value, bin_counts = compute_calibration_curve(
        y_true, y_prob, n_bins
    )

    # Metriken berechnen
    ece = compute_ece(y_true, y_prob, n_bins)
    brier = compute_brier_score(y_true, y_prob)

    # Plot erstellen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Reliability Diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')

    # Konfidenzintervalle als Schatten
    ax1.fill_between(
        mean_predicted_value,
        fraction_of_positives - 0.1,
        fraction_of_positives + 0.1,
        alpha=0.2
    )

    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'{title}\nECE={ece:.4f}, Brier={brier:.4f}')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Histogram der Vorhersagen
    ax2.bar(mean_predicted_value, bin_counts / bin_counts.sum(),
            width=1/n_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Fraction of Samples')
    ax2.set_title('Prediction Distribution')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Calibration plot gespeichert: {save_path}")

    return fig


def temperature_scaling_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    initial_temp: float = 1.5
) -> Tuple[float, np.ndarray]:
    """
    Führt Temperature Scaling zur Nachkalibrierung durch.

    Temperature Scaling ist eine einfache Methode zur Verbesserung
    der Kalibrierung durch Division der Logits durch eine Temperatur T.

    Args:
        y_true: Ground Truth Labels
        y_prob: Vorhergesagte Wahrscheinlichkeiten
        initial_temp: Initiale Temperatur für Optimierung

    Returns:
        Tuple von (optimale_temperatur, kalibrierte_wahrscheinlichkeiten)

    Reference:
        Guo et al. (2017): On Calibration of Modern Neural Networks
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        logger.warning("scipy nicht installiert - Temperature Scaling nicht möglich")
        return 1.0, y_prob

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Clip Wahrscheinlichkeiten für numerische Stabilität
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    # Logits berechnen
    logits = np.log(y_prob / (1 - y_prob))

    def nll_loss(temperature):
        """Negative Log-Likelihood Loss."""
        temp = temperature[0]
        if temp <= 0:
            return float('inf')
        scaled_logits = logits / temp
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
        nll = -np.mean(y_true * np.log(scaled_probs) + (1 - y_true) * np.log(1 - scaled_probs))
        return nll

    # Optimiere Temperatur
    result = minimize(nll_loss, [initial_temp], method='Nelder-Mead')
    optimal_temp = result.x[0]

    # Kalibrierte Wahrscheinlichkeiten berechnen
    calibrated_logits = logits / optimal_temp
    calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))

    logger.info(f"Temperature Scaling: T={optimal_temp:.4f}")
    logger.info(f"  ECE vorher: {compute_ece(y_true, y_prob):.4f}")
    logger.info(f"  ECE nachher: {compute_ece(y_true, calibrated_probs):.4f}")

    return float(optimal_temp), calibrated_probs


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Berechnet Kalibrierungsmetriken")
    parser.add_argument("--predictions", type=str, required=True,
                       help="JSON-Datei mit Vorhersagen")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--plot", type=str, default=None,
                       help="Pfad für Calibration Plot")
    parser.add_argument("--validate", action="store_true",
                       help="Prüfe ob ECE < 0.1")

    args = parser.parse_args()

    # Lade Vorhersagen
    with open(args.predictions, 'r') as f:
        data = json.load(f)

    y_true = data.get("y_true", [])
    y_prob = data.get("y_prob", [])

    if not y_true or not y_prob:
        print("Keine Daten in predictions-Datei gefunden!")
        exit(1)

    # Metriken berechnen
    metrics = compute_calibration_metrics(y_true, y_prob, args.n_bins)

    print("\n" + "=" * 50)
    print("KALIBRIERUNGSMETRIKEN")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Plot erstellen
    if args.plot:
        plot_calibration_curve(y_true, y_prob, args.n_bins, save_path=args.plot)

    # Validierung
    if args.validate:
        if metrics["ECE"] < 0.1:
            print("\n ECE < 0.1 - System ist gut kalibriert")
            exit(0)
        else:
            print(f"\n ECE = {metrics['ECE']:.4f} >= 0.1 - Kalibrierung verbesserungswürdig")
            exit(1)
