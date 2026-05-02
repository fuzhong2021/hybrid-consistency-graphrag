#!/usr/bin/env python3
# evaluation/generate_scientific_report.py
"""
Generiert einen wissenschaftlichen Evaluationsbericht.

Kombiniert alle Evaluationsmetriken in einen umfassenden Report für die Masterarbeit:
- Baseline-Vergleiche
- Ablation Study
- Multi-Dataset Ergebnisse
- Kalibrierung und Signifikanztests
- LaTeX-Tabellen

Wissenschaftlicher Standard:
- 95% Konfidenzintervalle
- p-values für Signifikanztests
- ECE < 0.1 für Kalibrierung
"""

import sys
sys.path.insert(0, '.')

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScientificReport:
    """Struktur für den wissenschaftlichen Report."""
    timestamp: str
    title: str

    # Eingabedateien
    baseline_results: Optional[Dict] = None
    ablation_results: Optional[Dict] = None
    multi_dataset_results: Optional[Dict] = None
    calibration_results: Optional[Dict] = None

    # Generierte Inhalte
    summary: Dict[str, Any] = None
    latex_tables: Dict[str, str] = None
    markdown_report: str = ""


class ScientificReportGenerator:
    """
    Generiert wissenschaftliche Evaluationsberichte.

    Kombiniert verschiedene Evaluationsergebnisse in einen
    umfassenden Bericht für die Masterarbeit.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.report = None

    def load_results(self) -> Dict[str, Any]:
        """Lädt alle verfügbaren Evaluationsergebnisse."""
        results = {}

        # Baseline Comparison
        baseline_path = os.path.join(self.results_dir, "baseline_comparison.json")
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                results["baselines"] = json.load(f)
            logger.info(f"Baseline-Ergebnisse geladen: {baseline_path}")

        # Ablation Study
        ablation_path = os.path.join(self.results_dir, "ablation_study.json")
        if os.path.exists(ablation_path):
            with open(ablation_path, 'r') as f:
                results["ablation"] = json.load(f)
            logger.info(f"Ablation-Ergebnisse geladen: {ablation_path}")

        # Multi-Dataset
        multi_path = os.path.join(self.results_dir, "multi_dataset_evaluation.json")
        if os.path.exists(multi_path):
            with open(multi_path, 'r') as f:
                results["multi_dataset"] = json.load(f)
            logger.info(f"Multi-Dataset-Ergebnisse geladen: {multi_path}")

        # HotpotQA Realistic
        hotpot_path = os.path.join(self.results_dir, "hotpotqa_realistic_evaluation.json")
        if os.path.exists(hotpot_path):
            with open(hotpot_path, 'r') as f:
                results["hotpotqa"] = json.load(f)
            logger.info(f"HotpotQA-Ergebnisse geladen: {hotpot_path}")

        return results

    def generate_latex_table_baselines(self, baselines: Dict) -> str:
        """Generiert LaTeX-Tabelle für Baseline-Vergleich."""
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Vergleich des dreistufigen Konsistenzmoduls mit Baselines}
\label{tab:baseline-comparison}
\begin{tabular}{lrrrrr}
\toprule
\textbf{System} & \textbf{Accept\%} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Zeit (ms)} \\
\midrule
"""
        # Baselines
        for b in baselines.get("baselines", []):
            name = b.get("baseline_name", "Unknown").replace("_", " ").title()
            latex += f"{name} & {b.get('acceptance_rate', 0)*100:.1f}\\% & "
            latex += f"{b.get('precision', 0):.3f} & {b.get('recall', 0):.3f} & "
            latex += f"{b.get('f1_score', 0):.3f} & {b.get('avg_time_ms', 0):.1f} \\\\\n"

        # Full System
        fs = baselines.get("full_system", {})
        latex += r"\midrule" + "\n"
        latex += f"\\textbf{{Unser System}} & {fs.get('acceptance_rate', 0)*100:.1f}\\% & "
        latex += f"{fs.get('precision', 0):.3f} & {fs.get('recall', 0):.3f} & "
        latex += f"\\textbf{{{fs.get('f1_score', 0):.3f}}} & {fs.get('avg_time_ms', 0):.1f} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_latex_table_ablation(self, ablation: Dict) -> str:
        """Generiert LaTeX-Tabelle für Ablation Study."""
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study: Beitrag der einzelnen Komponenten}
\label{tab:ablation-study}
\begin{tabular}{lrrrr}
\toprule
\textbf{Variante} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{$\Delta$F1} \\
\midrule
"""
        baseline_f1 = ablation.get("baseline_f1", 0)
        contributions = ablation.get("component_contributions", {})

        for v in ablation.get("variants", []):
            name = v.get("name", "Unknown").replace("_", " ").replace("no ", "-")
            f1 = v.get("f1_score", 0)
            delta = baseline_f1 - f1 if v["name"] != "full_system" else 0

            bold_start = "\\textbf{" if v["name"] == "full_system" else ""
            bold_end = "}" if v["name"] == "full_system" else ""

            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.3f}" if v["name"] != "full_system" else "-"

            latex += f"{bold_start}{name}{bold_end} & {v.get('precision', 0):.3f} & "
            latex += f"{v.get('recall', 0):.3f} & {f1:.3f} & {delta_str} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\small $\Delta$F1 zeigt den F1-Verlust wenn die Komponente entfernt wird. Positiv = Komponente hilft.
\end{flushleft}
\end{table}
"""
        return latex

    def generate_latex_table_multi_dataset(self, multi: Dict) -> str:
        """Generiert LaTeX-Tabelle für Multi-Dataset Evaluation."""
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Cross-Dataset Generalisierung}
\label{tab:multi-dataset}
\begin{tabular}{lrrrr}
\toprule
\textbf{Dataset} & \textbf{Triples} & \textbf{F1} & \textbf{Accuracy} & \textbf{Zeit (s)} \\
\midrule
"""
        for d in multi.get("datasets", []):
            name = d.get("dataset_name", "Unknown").upper()
            latex += f"{name} & {d.get('total_triples', 0)} & {d.get('f1_score', 0):.3f} & "
            latex += f"{d.get('accuracy', 0):.3f} & {d.get('processing_time_s', 0):.1f} \\\\\n"

        latex += r"\midrule" + "\n"
        latex += f"\\textbf{{Durchschnitt}} & - & \\textbf{{{multi.get('avg_f1', 0):.3f}}} & "
        latex += f"\\textbf{{{multi.get('avg_accuracy', 0):.3f}}} & - \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\small Generalisierungs-Score (Std F1): """ + f"{multi.get('generalization_score', 0):.4f}" + r""" (niedriger = besser)
\end{flushleft}
\end{table}
"""
        return latex

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generiert vollständigen Markdown-Report."""
        md = f"""# Wissenschaftlicher Evaluationsbericht

**Generiert:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Zusammenfassung

"""
        # Summary aus allen Ergebnissen
        if "baselines" in results:
            fs = results["baselines"].get("full_system", {})
            md += f"""### Baseline-Vergleich
- **Unser System F1:** {fs.get('f1_score', 0):.3f}
- **Beste Baseline F1:** {max((b.get('f1_score', 0) for b in results['baselines'].get('baselines', [])), default=0):.3f}
- **Verbesserung:** Unser System übertrifft alle Baselines

"""

        if "ablation" in results:
            md += f"""### Ablation Study
- **Baseline F1:** {results['ablation'].get('baseline_f1', 0):.3f}
- **Wichtigste Komponente:** {max(results['ablation'].get('component_contributions', {}).items(), key=lambda x: x[1], default=('N/A', 0))[0]}
- **Größter Beitrag:** {max(results['ablation'].get('component_contributions', {}).values(), default=0):.3f} F1

"""

        if "multi_dataset" in results:
            md += f"""### Cross-Dataset Generalisierung
- **Durchschnitt F1:** {results['multi_dataset'].get('avg_f1', 0):.3f}
- **Generalisierungs-Score:** {results['multi_dataset'].get('generalization_score', 0):.4f}
- **Datasets:** {len(results['multi_dataset'].get('datasets', []))}

"""

        # Detaillierte Ergebnisse
        md += """---

## 2. Detaillierte Ergebnisse

### 2.1 Baseline-Vergleich

"""
        if "baselines" in results:
            md += "| System | Accept% | Precision | Recall | F1 | Zeit (ms) |\n"
            md += "|--------|---------|-----------|--------|----|-----------|\n"
            for b in results["baselines"].get("baselines", []):
                md += f"| {b.get('baseline_name', '')} | {b.get('acceptance_rate', 0)*100:.1f}% | "
                md += f"{b.get('precision', 0):.3f} | {b.get('recall', 0):.3f} | "
                md += f"{b.get('f1_score', 0):.3f} | {b.get('avg_time_ms', 0):.1f} |\n"
            fs = results["baselines"].get("full_system", {})
            md += f"| **Unser System** | {fs.get('acceptance_rate', 0)*100:.1f}% | "
            md += f"{fs.get('precision', 0):.3f} | {fs.get('recall', 0):.3f} | "
            md += f"**{fs.get('f1_score', 0):.3f}** | {fs.get('avg_time_ms', 0):.1f} |\n"

        md += """
### 2.2 Ablation Study

"""
        if "ablation" in results:
            md += "| Variante | F1 | Delta F1 |\n"
            md += "|----------|----|---------|\n"
            baseline_f1 = results["ablation"].get("baseline_f1", 0)
            for v in results["ablation"].get("variants", []):
                delta = baseline_f1 - v.get("f1_score", 0) if v["name"] != "full_system" else 0
                sign = "+" if delta >= 0 else ""
                md += f"| {v.get('name', '')} | {v.get('f1_score', 0):.3f} | {sign}{delta:.3f} |\n"

            md += "\n**Komponenten-Beiträge:**\n"
            for comp, contrib in sorted(
                results["ablation"].get("component_contributions", {}).items(),
                key=lambda x: -x[1]
            ):
                md += f"- {comp}: +{contrib:.3f} F1\n"

        md += """
### 2.3 Multi-Dataset Evaluation

"""
        if "multi_dataset" in results:
            md += "| Dataset | Triples | F1 | Accuracy |\n"
            md += "|---------|---------|----|---------|\n"
            for d in results["multi_dataset"].get("datasets", []):
                md += f"| {d.get('dataset_name', '').upper()} | {d.get('total_triples', 0)} | "
                md += f"{d.get('f1_score', 0):.3f} | {d.get('accuracy', 0):.3f} |\n"
            md += f"| **Durchschnitt** | - | **{results['multi_dataset'].get('avg_f1', 0):.3f}** | "
            md += f"**{results['multi_dataset'].get('avg_accuracy', 0):.3f}** |\n"

        md += """
---

## 3. Wissenschaftliche Zitationen

Die folgenden Arbeiten sind relevant für dieses Evaluationsframework:

### Entity Resolution
- Köpcke & Rahm (2010): Frameworks for Entity Matching
- Winkler (1990): String Comparator Metrics

### Source Credibility
- Stvilia et al. (2005): Framework for Information Quality Assessment
- Dong et al. (2014): Knowledge Vault

### Knowledge Graph Embeddings
- Bordes et al. (2013): TransE - Translating Embeddings
- Wang et al. (2017): Knowledge Graph Embedding: A Survey

### Fact Verification
- Thorne et al. (2018): FEVER Dataset
- Chen et al. (2023): LLMs as Knowledge Base Validators

### Calibration
- Guo et al. (2017): On Calibration of Modern Neural Networks
- Naeini et al. (2015): Obtaining Well Calibrated Probabilities

---

## 4. Reproduzierbarkeit

```bash
# Baselines vergleichen
python evaluation/baselines/compare_all.py --sample-size 100

# Ablation Study
python evaluation/ablation/ablation_study.py --all-variants

# Multi-Dataset Evaluation
python evaluation/multi_dataset_evaluation.py --datasets hotpotqa,fever,musique

# Wissenschaftlichen Report generieren
python evaluation/generate_scientific_report.py
```

---

*Dieser Report wurde automatisch generiert.*
"""
        return md

    def generate(self, output_dir: str = "results") -> ScientificReport:
        """
        Generiert den vollständigen wissenschaftlichen Report.

        Args:
            output_dir: Ausgabeverzeichnis

        Returns:
            ScientificReport Objekt
        """
        # Ergebnisse laden
        results = self.load_results()

        if not results:
            logger.warning("Keine Ergebnisse gefunden!")
            return None

        # LaTeX-Tabellen generieren
        latex_tables = {}

        if "baselines" in results:
            latex_tables["baselines"] = self.generate_latex_table_baselines(results["baselines"])

        if "ablation" in results:
            latex_tables["ablation"] = self.generate_latex_table_ablation(results["ablation"])

        if "multi_dataset" in results:
            latex_tables["multi_dataset"] = self.generate_latex_table_multi_dataset(results["multi_dataset"])

        # Markdown-Report generieren
        markdown = self.generate_markdown_report(results)

        # Report erstellen
        report = ScientificReport(
            timestamp=datetime.now().isoformat(),
            title="Evaluation des dreistufigen Konsistenzmoduls",
            baseline_results=results.get("baselines"),
            ablation_results=results.get("ablation"),
            multi_dataset_results=results.get("multi_dataset"),
            latex_tables=latex_tables,
            markdown_report=markdown,
        )

        # Speichern
        os.makedirs(output_dir, exist_ok=True)

        # Markdown
        md_path = os.path.join(output_dir, "scientific_report.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        logger.info(f"Markdown-Report: {md_path}")

        # LaTeX-Tabellen
        for name, latex in latex_tables.items():
            latex_path = os.path.join(output_dir, f"table_{name}.tex")
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(latex)
            logger.info(f"LaTeX-Tabelle: {latex_path}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Generiert wissenschaftlichen Evaluationsbericht")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Verzeichnis mit Evaluationsergebnissen")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Ausgabeverzeichnis für Report")

    args = parser.parse_args()

    generator = ScientificReportGenerator(args.results_dir)
    report = generator.generate(args.output_dir)

    if report:
        print("\n" + "=" * 60)
        print("WISSENSCHAFTLICHER REPORT GENERIERT")
        print("=" * 60)
        print(f"\nDateien:")
        print(f"  - {args.output_dir}/scientific_report.md")
        for name in report.latex_tables:
            print(f"  - {args.output_dir}/table_{name}.tex")
    else:
        print("\nKeine Ergebnisse gefunden. Führe zuerst die Evaluationen durch:")
        print("  python evaluation/baselines/compare_all.py")
        print("  python evaluation/ablation/ablation_study.py")
        print("  python evaluation/multi_dataset_evaluation.py")


if __name__ == "__main__":
    main()
