"""
Funktionen zum Speichern von Ergebnissen und Metriken.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

def save_results(
    results: Dict[str, Any],
    output_dir: str,
    filename: str
):
    """Speichert Ergebnisse als JSON-Datei."""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_similarity_matrix(
    matrix: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str
):
    """Speichert eine Ähnlichkeitsmatrix als CSV und visualisiert sie."""
    # Speichere als CSV
    df = pd.DataFrame(matrix)
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    
    # Erstelle Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlOrRd", vmin=0, vmax=1)
    plt.title("Ähnlichkeitsmatrix der Textvarianten")
    plt.tight_layout()
    
    # Speichere Plot
    plot_path = output_path.with_suffix(".png")
    plt.savefig(plot_path)
    plt.close()

def save_summary_comparison(
    summaries: Dict[str, str],
    output_dir: str,
    filename: str
):
    """Speichert einen Vergleich verschiedener Zusammenfassungen."""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for variant, summary in summaries.items():
            f.write(f"=== {variant} ===\n")
            f.write(f"{summary}\n\n") 