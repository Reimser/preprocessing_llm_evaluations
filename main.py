"""
Hauptsteuerung der Preprocessing-Vergleichspipeline.
"""

import os
from pathlib import Path
from typing import Dict, List
import argparse

from preprocessing import (
    clean,
    stopwords,
    lemmatize,
    truncate,
    sentence_filter,
    tfidf
)
from generate.generate_summaries import process_text_variants
from evaluate.compare_embeddings import EmbeddingComparator
from utils.loader import (
    load_cnn_dailymail,
    load_processed_variants,
    load_goldstandard_summaries
)
from utils.writer import (
    save_results,
    save_similarity_matrix,
    save_summary_comparison
)

def process_text(text: str, tfidf_analyzer: tfidf.TFIDFAnalyzer) -> Dict[str, str]:
    """Verarbeitet einen Text mit verschiedenen Preprocessing-Strategien."""
    variants = {
        "original": text,
        "cleaned": clean.normalize_text(text),
        "no_stopwords": stopwords.remove_stopwords(text),
        "lemmatized": lemmatize.lemmatize_text(text),
        "truncated": truncate.truncate_by_tokens(text, max_tokens=100),
        "tfidf_filtered": tfidf_analyzer.filter_by_importance(text)
    }
    return variants

def main():
    parser = argparse.ArgumentParser(description="Preprocessing-Vergleichspipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Verzeichnis mit den Daten")
    parser.add_argument("--output_dir", type=str, default="results", help="Ausgabeverzeichnis")
    args = parser.parse_args()
    
    # Lade Daten
    print("Lade Daten...")
    data = load_cnn_dailymail(args.data_dir)
    goldstandard = load_goldstandard_summaries(args.data_dir)
    
    # Initialisiere Analyzer
    comparator = EmbeddingComparator()
    tfidf_analyzer = tfidf.TFIDFAnalyzer()
    
    # Trainiere TF-IDF auf allen Texten
    all_texts = [item["text"] for item in data]
    tfidf_analyzer.fit_transform(all_texts)
    
    # Verarbeite Texte
    print("Verarbeite Texte...")
    all_variants = {}
    all_summaries = {}
    
    for i, item in enumerate(data[:10]):  # Verarbeite nur die ersten 10 Beispiele
        print(f"Verarbeite Beispiel {i+1}/10...")
        
        # Preprocessing
        variants = process_text(item["text"], tfidf_analyzer)
        all_variants.update(variants)
        
        # Generiere Zusammenfassungen
        summaries = process_text_variants(variants)
        all_summaries.update(summaries)
        
        # Vergleiche mit Goldstandard
        if i < len(goldstandard):
            similarities = comparator.compare_with_goldstandard(
                summaries,
                goldstandard[i]
            )
            
            # Speichere Ergebnisse
            save_results(
                similarities,
                args.output_dir,
                f"similarities_{i}.json"
            )
    
    # Vergleiche alle Varianten miteinander
    similarity_matrix = comparator.compare_variants(all_summaries)
    save_similarity_matrix(
        similarity_matrix,
        args.output_dir,
        "similarity_matrix.csv"
    )
    
    # Speichere Zusammenfassungsvergleich
    save_summary_comparison(
        all_summaries,
        args.output_dir,
        "summary_comparison.txt"
    )
    
    print("Pipeline abgeschlossen!")

if __name__ == "__main__":
    main() 