# main.py

import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from generate.generate_summaries import generate_summary, load_prompt # aus generate/generate_summaries.py
from preprocessing import apply_preprocessing # aus preprocessing/__init__.py

load_dotenv()  # .env-Datei laden
df = pd.read_csv("data/eval/eval_subset.csv")  # Testdaten laden
prompt_template = load_prompt("prompts/default_prompt.txt")  # Prompt-Vorlage laden

strategien = [  # Definierte Preprocessing-Strategien
    [],  # Baseline: keine Vorverarbeitung
    ["remove_special_chars", "normalize_text"],
    ["remove_stopwords"],
    ["lemmatize"],
    ["truncate_tokens"],
    ["filter_main_sentences"]
]

for steps in strategien:  # Iteration über alle Strategien
    name = "none" if not steps else "+".join(steps)  # Name generieren
    print(f"Verarbeite Strategie: {name}")
    summaries = []  # Liste für Ergebnisse

    for i in tqdm(range(len(df))):  # Fortschrittsanzeige für Artikel
        article = df.at[i, "article"]  # Einzelnen Artikel holen
        processed_text = apply_preprocessing(article, steps)  # Preprocessing anwenden
        summary = generate_summary(processed_text, prompt_template)  # GPT-4 API-Aufruf
        summaries.append(summary)
        time.sleep(1.5)  # API-Rate-Limit berücksichtigen

    df[f"gpt_{name}"] = summaries  # Ergebnisse in neuer Spalte speichern

df.to_csv("data/processed/eval_results_with_all_strategies.csv", index=False)  # Gesamtergebnis speichern
print("Fertig! Ergebnisse gespeichert unter data/processed/eval_results_with_all_strategies.csv")
