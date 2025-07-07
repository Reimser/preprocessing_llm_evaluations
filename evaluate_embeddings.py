import pandas as pd
from models.embedder import TextEmbedder

# 1. CSV-Datei laden
df = pd.read_csv("data/processed/eval_results_with_all_strategies.csv")

# 2. Spalten definieren
article_col = "article"
summary_cols = [
    "gpt_none",
    "gpt_remove_special_chars+normalize_text",
    "gpt_remove_stopwords",
    "gpt_lemmatize",
    "gpt_truncate_tokens",
    "gpt_filter_main_sentences"
]

# 3. Embedding-Modell laden
embedder = TextEmbedder()

# 4. Artikel-Embeddings berechnen
article_embeddings = embedder.encode(df[article_col].tolist())

# 5. Für jede Strategie: Summary-Embeddings & Similarity berechnen
for col in summary_cols:
    summary_embeddings = embedder.encode(df[col].tolist())
    similarities = embedder.compute_similarity(article_embeddings, summary_embeddings)

    # Cosine Similarity-Diagonale (1:1-Vergleich) extrahieren
    scores = [round(float(sim.item()), 4) for sim in similarities.diagonal()]
    df[f"{col}_similarity"] = scores

# 6. Ergebnis speichern
df.to_csv("results/evaluation_results.csv", index=False)
print("✅ Evaluation abgeschlossen – Ergebnisse in: output/evaluation_results.csv")
