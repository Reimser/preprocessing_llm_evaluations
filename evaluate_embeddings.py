import pandas as pd 
from models.embedder import TextEmbedder  

df = pd.read_csv("data/processed/eval_results_with_all_strategies.csv")  # Ergebnisdaten laden

article_col = "article"  # Spalte mit Artikeln
summary_cols = [  # Spalten mit GPT-4 Zusammenfassungen
    "gpt_none",
    "gpt_remove_special_chars+normalize_text",
    "gpt_remove_stopwords",
    "gpt_lemmatize",
    "gpt_truncate_tokens",
    "gpt_filter_main_sentences"
]

embedder = TextEmbedder()  # Sentence-BERT Modell initialisieren

article_embeddings = embedder.encode(df[article_col].tolist())  # Embeddings für Artikel berechnen

for col in summary_cols:  # Für jede Strategie:
    summary_embeddings = embedder.encode(df[col].tolist())  # Embeddings für Zusammenfassungen
    similarities = embedder.compute_similarity(article_embeddings, summary_embeddings)  # Similarity-Matrix

    scores = [round(float(sim.item()), 4) for sim in similarities.diagonal()]  # 1:1 Vergleich Scores extrahieren
    df[f"{col}_similarity"] = scores  # Scores als neue Spalte hinzufügen

df.to_csv("results/evaluation_results.csv", index=False)  # Ergebnisse speichern
print("Evaluation abgeschlossen – Ergebnisse in: output/evaluation_results.csv")  # Abschlussmeldung
