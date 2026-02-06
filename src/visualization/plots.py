# src/visualization/plots.py
"""
Visualisierungsfunktionen für Knowledge Graph Evaluation.

Generiert alle Visualisierungen für die Masterarbeit:
- Knowledge Graph als Netzwerk
- Validierungsmetriken als Balkendiagramme
- Konfidenz-Verteilungen als Histogramme
- Verarbeitungszeiten als Boxplots
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Farbpalette für Entitätstypen
ENTITY_COLORS = {
    "Person": "#FF6B6B",      # Rot
    "Organisation": "#4ECDC4", # Türkis
    "Ort": "#45B7D1",         # Blau
    "Ereignis": "#96CEB4",    # Grün
    "Dokument": "#FFEAA7",    # Gelb
    "Konzept": "#DDA0DD",     # Pflaume
    "Unknown": "#95A5A6"      # Grau
}

# Matplotlib-Stil für wissenschaftliche Publikationen
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150


def plot_knowledge_graph(
    graph_repo,
    output_path: str = "results/knowledge_graph.png",
    max_nodes: int = 100,
    figsize: tuple = (16, 12),
    show_labels: bool = True,
    title: str = "Knowledge Graph Visualisierung"
) -> str:
    """
    Visualisiert den Knowledge Graph als Netzwerk.

    Args:
        graph_repo: InMemoryGraphRepository oder Neo4jRepository
        output_path: Pfad für die Ausgabedatei
        max_nodes: Maximale Anzahl anzuzeigender Knoten
        figsize: Größe der Abbildung
        show_labels: Zeige Knotenbeschriftungen
        title: Titel der Abbildung

    Returns:
        Pfad zur gespeicherten Datei
    """
    # Erstelle NetworkX Graph
    G = nx.DiGraph()

    # Hole alle Entitäten
    try:
        entities = graph_repo.find_all_entities()
    except AttributeError:
        entities = []
        logger.warning("Repository hat keine find_all_entities Methode")

    if not entities:
        logger.warning("Keine Entitäten im Graph gefunden")
        # Erstelle leeren Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Keine Daten verfügbar",
                ha='center', va='center', fontsize=20, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        return output_path

    # Beschränke auf max_nodes
    if len(entities) > max_nodes:
        entities = entities[:max_nodes]
        logger.info(f"Graph auf {max_nodes} Knoten beschränkt")

    # Füge Knoten hinzu
    for entity in entities:
        entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        G.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity_type,
            color=ENTITY_COLORS.get(entity_type, ENTITY_COLORS["Unknown"])
        )

    # Füge Kanten hinzu
    entity_ids = set(e.id for e in entities)
    try:
        for entity in entities:
            relations = graph_repo.find_relations(source_id=entity.id)
            for rel in relations:
                target_id = rel.get("target", {}).get("id") if isinstance(rel.get("target"), dict) else None
                if target_id and target_id in entity_ids:
                    rel_type = rel.get("rel_type", "RELATED")
                    G.add_edge(entity.id, target_id, relation=rel_type)
    except Exception as e:
        logger.warning(f"Fehler beim Laden der Relationen: {e}")

    # Erstelle Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Layout berechnen
    if len(G.nodes()) > 50:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)

    # Knotenfarben und -größen
    node_colors = [G.nodes[n].get('color', ENTITY_COLORS["Unknown"]) for n in G.nodes()]
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]

    # Zeichne Kanten
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#CCCCCC',
        arrows=True,
        arrowsize=10,
        alpha=0.6,
        connectionstyle="arc3,rad=0.1"
    )

    # Zeichne Knoten
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    # Zeichne Labels
    if show_labels and len(G.nodes()) <= 50:
        labels = {n: G.nodes[n].get('name', n)[:20] for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=8,
            font_weight='bold'
        )

    # Legende
    legend_patches = [
        mpatches.Patch(color=color, label=entity_type)
        for entity_type, color in ENTITY_COLORS.items()
        if any(G.nodes[n].get('entity_type') == entity_type for n in G.nodes())
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    # Statistiken als Text
    stats_text = f"Knoten: {G.number_of_nodes()} | Kanten: {G.number_of_edges()}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Speichern
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    logger.info(f"Knowledge Graph gespeichert: {output_path}")
    return output_path


def plot_validation_metrics(
    metrics: Dict[str, Any],
    output_path: str = "results/validation_metrics.png",
    figsize: tuple = (12, 6),
    title: str = "Konsistenzmodul - Validierungsmetriken"
) -> str:
    """
    Zeigt Precision/Recall/F1 pro Stufe als Balkendiagramm.

    Args:
        metrics: Dictionary mit Metriken aus ConsistencyOrchestrator
        output_path: Pfad für die Ausgabedatei
        figsize: Größe der Abbildung
        title: Titel der Abbildung

    Returns:
        Pfad zur gespeicherten Datei
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # =========================================================================
    # Links: Validierungsergebnisse (Akzeptiert/Abgelehnt/Review)
    # =========================================================================
    ax1 = axes[0]

    summary = metrics.get("summary", {})
    if not summary:
        # Fallback für einfache Statistik-Dicts
        summary = metrics

    categories = ['Akzeptiert', 'Abgelehnt', 'Review']
    values = [
        summary.get("accepted", 0),
        summary.get("rejected", 0),
        summary.get("needs_review", 0)
    ]
    colors = ['#27AE60', '#E74C3C', '#F39C12']

    bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)

    # Werte auf Balken anzeigen
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.annotate(f'{val}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Anzahl Triples', fontsize=12)
    ax1.set_title('Validierungsergebnisse', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.2 if values else 10)

    # =========================================================================
    # Rechts: Pass-Rate pro Stufe
    # =========================================================================
    ax2 = axes[1]

    stage_breakdown = metrics.get("stage_breakdown", {})

    if stage_breakdown:
        stages = ['Regelbasiert', 'Embedding', 'LLM']
        pass_rates = []

        for stage_key in ["stage1_rule_based", "stage2_embedding", "stage3_llm"]:
            stage_data = stage_breakdown.get(stage_key, {})
            rate_str = stage_data.get("pass_rate", "0%")
            # Parse percentage string
            if isinstance(rate_str, str):
                rate = float(rate_str.replace("%", "")) / 100
            else:
                rate = float(rate_str)
            pass_rates.append(rate)

        colors2 = ['#3498DB', '#9B59B6', '#1ABC9C']
        bars2 = ax2.bar(stages, pass_rates, color=colors2, edgecolor='black', linewidth=1.2)

        # Prozente auf Balken anzeigen
        for bar, rate in zip(bars2, pass_rates):
            height = bar.get_height()
            ax2.annotate(f'{rate:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax2.set_ylabel('Pass-Rate', fontsize=12)
        ax2.set_title('Pass-Rate pro Validierungsstufe', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.15)
        ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Schwelle (70%)')
        ax2.legend(loc='lower right')
    else:
        ax2.text(0.5, 0.5, "Keine Stufendaten\nverfügbar",
                ha='center', va='center', fontsize=14, color='gray')
        ax2.axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Speichern
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    logger.info(f"Validierungsmetriken gespeichert: {output_path}")
    return output_path


def plot_confidence_distribution(
    metrics: Dict[str, Any],
    output_path: str = "results/confidence_distribution.png",
    figsize: tuple = (10, 6),
    title: str = "Konfidenz-Verteilung pro Stufe"
) -> str:
    """
    Histogramm der Konfidenzwerte pro Validierungsstufe.

    Args:
        metrics: Dictionary mit confidence_histograms
        output_path: Pfad für die Ausgabedatei
        figsize: Größe der Abbildung
        title: Titel der Abbildung

    Returns:
        Pfad zur gespeicherten Datei
    """
    fig, ax = plt.subplots(figsize=figsize)

    histograms = metrics.get("confidence_histograms", {})

    if histograms:
        # Bins für Histogramm
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = 0.08

        colors = ['#3498DB', '#9B59B6', '#1ABC9C']
        labels = ['Regelbasiert', 'Embedding', 'LLM']
        stage_keys = ['stage1', 'stage2', 'stage3']

        for i, (stage_key, label, color) in enumerate(zip(stage_keys, labels, colors)):
            hist_data = histograms.get(stage_key, [])
            if hist_data:
                offset = (i - 1) * width
                ax.bar(bin_centers + offset, hist_data, width=width,
                       label=label, color=color, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Konfidenz', fontsize=12)
        ax.set_ylabel('Anzahl', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.set_xlim(-0.05, 1.05)

        # Schwellenwerte markieren
        ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5, label='High (0.9)')
        ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Medium (0.7)')
    else:
        # Fallback: Zeige Platzhalter
        ax.text(0.5, 0.5, "Keine Konfidenz-Daten\nverfügbar",
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    # Speichern
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    logger.info(f"Konfidenz-Verteilung gespeichert: {output_path}")
    return output_path


def plot_processing_times(
    metrics: Dict[str, Any],
    output_path: str = "results/processing_times.png",
    figsize: tuple = (10, 6),
    title: str = "Verarbeitungszeit pro Stufe"
) -> str:
    """
    Boxplot der Verarbeitungszeiten pro Validierungsstufe.

    Args:
        metrics: Dictionary mit timing-Daten
        output_path: Pfad für die Ausgabedatei
        figsize: Größe der Abbildung
        title: Titel der Abbildung

    Returns:
        Pfad zur gespeicherten Datei
    """
    fig, ax = plt.subplots(figsize=figsize)

    stage_breakdown = metrics.get("stage_breakdown", {})

    if stage_breakdown:
        stages = ['Regelbasiert', 'Embedding', 'LLM']
        avg_times = []

        for stage_key in ["stage1_rule_based", "stage2_embedding", "stage3_llm"]:
            stage_data = stage_breakdown.get(stage_key, {})
            time_str = stage_data.get("avg_time_ms", "0")
            # Parse time string
            if isinstance(time_str, str):
                time_val = float(time_str.replace("ms", "").strip())
            else:
                time_val = float(time_str)
            avg_times.append(time_val)

        colors = ['#3498DB', '#9B59B6', '#1ABC9C']
        bars = ax.bar(stages, avg_times, color=colors, edgecolor='black', linewidth=1.2)

        # Werte auf Balken anzeigen
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax.annotate(f'{time_val:.1f}ms',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Durchschnittliche Zeit (ms)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Log-Skala wenn große Unterschiede
        if max(avg_times) > 10 * min(avg_times) if avg_times and min(avg_times) > 0 else False:
            ax.set_yscale('log')
            ax.set_ylabel('Durchschnittliche Zeit (ms, log)', fontsize=12)
    else:
        ax.text(0.5, 0.5, "Keine Timing-Daten\nverfügbar",
                ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')

    plt.tight_layout()

    # Speichern
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    logger.info(f"Verarbeitungszeiten gespeichert: {output_path}")
    return output_path


def plot_entity_type_distribution(
    graph_repo,
    output_path: str = "results/entity_types.png",
    figsize: tuple = (10, 6),
    title: str = "Verteilung der Entitätstypen"
) -> str:
    """
    Kreisdiagramm der Entitätstypen im Graph.

    Args:
        graph_repo: Graph Repository
        output_path: Pfad für die Ausgabedatei
        figsize: Größe der Abbildung
        title: Titel der Abbildung

    Returns:
        Pfad zur gespeicherten Datei
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Zähle Entitätstypen
    type_counts = {}
    try:
        entities = graph_repo.find_all_entities()
        for entity in entities:
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    except Exception as e:
        logger.warning(f"Fehler beim Zählen der Entitätstypen: {e}")

    if type_counts:
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = [ENTITY_COLORS.get(t, ENTITY_COLORS["Unknown"]) for t in labels]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            explode=[0.02] * len(sizes)
        )

        # Formatierung
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title(title, fontsize=14, fontweight='bold')

        # Statistik
        total = sum(sizes)
        ax.text(0.5, -0.1, f'Gesamt: {total} Entitäten',
                transform=ax.transAxes, ha='center', fontsize=11)
    else:
        ax.text(0.5, 0.5, "Keine Entitäten\nverfügbar",
                ha='center', va='center', fontsize=14, color='gray')

    plt.tight_layout()

    # Speichern
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    logger.info(f"Entitätstypen-Verteilung gespeichert: {output_path}")
    return output_path


def generate_all_visualizations(
    graph_repo,
    metrics: Dict[str, Any],
    output_dir: str = "results/"
) -> Dict[str, str]:
    """
    Generiert alle Visualisierungen.

    Args:
        graph_repo: Graph Repository (InMemory oder Neo4j)
        metrics: Metriken aus dem ConsistencyOrchestrator
        output_dir: Ausgabeverzeichnis

    Returns:
        Dictionary mit Pfaden zu allen generierten Dateien
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = {}

    logger.info(f"Generiere Visualisierungen in {output_dir}...")

    # 1. Knowledge Graph
    try:
        path = plot_knowledge_graph(
            graph_repo,
            output_path=str(output_path / "knowledge_graph.png")
        )
        generated_files["knowledge_graph"] = path
    except Exception as e:
        logger.error(f"Fehler bei Knowledge Graph Visualisierung: {e}")

    # 2. Validierungsmetriken
    try:
        path = plot_validation_metrics(
            metrics,
            output_path=str(output_path / "validation_metrics.png")
        )
        generated_files["validation_metrics"] = path
    except Exception as e:
        logger.error(f"Fehler bei Validierungsmetriken: {e}")

    # 3. Konfidenz-Verteilung
    try:
        path = plot_confidence_distribution(
            metrics,
            output_path=str(output_path / "confidence_distribution.png")
        )
        generated_files["confidence_distribution"] = path
    except Exception as e:
        logger.error(f"Fehler bei Konfidenz-Verteilung: {e}")

    # 4. Verarbeitungszeiten
    try:
        path = plot_processing_times(
            metrics,
            output_path=str(output_path / "processing_times.png")
        )
        generated_files["processing_times"] = path
    except Exception as e:
        logger.error(f"Fehler bei Verarbeitungszeiten: {e}")

    # 5. Entitätstypen
    try:
        path = plot_entity_type_distribution(
            graph_repo,
            output_path=str(output_path / "entity_types.png")
        )
        generated_files["entity_types"] = path
    except Exception as e:
        logger.error(f"Fehler bei Entitätstypen: {e}")

    logger.info(f"Visualisierungen generiert: {len(generated_files)} Dateien")

    return generated_files


if __name__ == "__main__":
    # Test-Ausführung
    logging.basicConfig(level=logging.INFO)

    print("\n=== Visualisierung Test ===\n")

    # Erstelle Test-Daten
    from src.graph.memory_repository import InMemoryGraphRepository
    from src.models.entities import Entity, EntityType, Relation

    repo = InMemoryGraphRepository()

    # Füge Test-Entitäten hinzu
    e1 = Entity(name="Albert Einstein", entity_type=EntityType.PERSON)
    e2 = Entity(name="Relativitätstheorie", entity_type=EntityType.CONCEPT)
    e3 = Entity(name="Physik-Nobelpreis", entity_type=EntityType.EVENT)
    e4 = Entity(name="ETH Zürich", entity_type=EntityType.ORGANIZATION)

    for e in [e1, e2, e3, e4]:
        repo.create_entity(e)

    # Füge Relationen hinzu
    r1 = Relation(source_id=e1.id, target_id=e2.id, relation_type="ENTWICKELTE")
    r2 = Relation(source_id=e1.id, target_id=e3.id, relation_type="ERHIELT")
    r3 = Relation(source_id=e1.id, target_id=e4.id, relation_type="STUDIERTE_AN")

    for r in [r1, r2, r3]:
        repo.create_relation(r)

    # Test-Metriken
    test_metrics = {
        "summary": {
            "total_processed": 100,
            "accepted": 75,
            "rejected": 15,
            "needs_review": 10,
            "acceptance_rate": "75.0%"
        },
        "stage_breakdown": {
            "stage1_rule_based": {"pass_rate": "95.0%", "avg_time_ms": "2.5"},
            "stage2_embedding": {"pass_rate": "80.0%", "avg_time_ms": "15.3"},
            "stage3_llm": {"pass_rate": "70.0%", "avg_time_ms": "850.0"}
        }
    }

    # Generiere alle Visualisierungen
    output_dir = "/private/tmp/claude/-Users-furkansaygin-masterarbeit-graphrag/e3978fe4-b0e1-4d80-99dd-9a39ddfb7468/scratchpad/viz_test"
    files = generate_all_visualizations(repo, test_metrics, output_dir)

    print(f"\nGenerierte Dateien:")
    for name, path in files.items():
        print(f"  - {name}: {path}")

    print("\n=== Test abgeschlossen ===")
