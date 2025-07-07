# main.py

import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from generate.generate_summaries import generate_summary, load_prompt
from preprocessing import apply_preprocessing

# Lade Umgebungsvariablen
load_dotenv()

# Lade Daten
df = pd.read_csv("data/eval/eval_subset.csv")

# Prompt-Vorlage
prompt_template = load_prompt("prompts/default_prompt.txt")

# Liste der Strategien, die du vergleichen willst
strategien = [
    [],  # keine Vorverarbeitung – baseline
    ["remove_special_chars", "normalize_text"],
    ["remove_stopwords"],
    ["lemmatize"],
    ["truncate_tokens"],
    ["filter_main_sentences"]
]

# Durchlaufe jede Strategie
for steps in strategien:
    name = "none" if not steps else "+".join(steps)
    print(f"Verarbeite Strategie: {name}")

    summaries = []

    for i in tqdm(range(len(df))):
        article = df.at[i, "article"]
        processed_text = apply_preprocessing(article, steps)
        summary = generate_summary(processed_text, prompt_template)
        summaries.append(summary)
        time.sleep(1.5)  # vermeidet 429-Fehler (API-Limit)

    df[f"gpt_{name}"] = summaries

# Speichere Ergebnis
df.to_csv("data/processed/eval_results_with_all_strategies.csv", index=False)
print("Fertig! Ergebnisse gespeichert unter data/processed/eval_results_with_all_strategies.csv")
