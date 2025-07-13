import os
import pandas as pd
from transformers import GPT2TokenizerFast

# GPT2-Tokenizer zum Tokenzählen
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Parameter für Eval-Subset
MIN_TOKENS = 50
MAX_TOKENS = 800
SAMPLE_SIZE = 3
RAW_PATH = "../data/raw/cnn.csv"
SAVE_PATH = "../data/eval/eval_subset.csv"

def estimate_tokens(text):
    return len(tokenizer.encode(text))

def create_eval_subset():
    print("Lade lokale Datei:", RAW_PATH)
    df = pd.read_csv(RAW_PATH)

    print("Filtere nach Tokenlänge...")
    df["token_count"] = df["article"].apply(estimate_tokens)
    filtered_df = df[
        (df["token_count"] >= MIN_TOKENS) &
        (df["token_count"] <= MAX_TOKENS)
    ]

    if len(filtered_df) < SAMPLE_SIZE:
        print(f"Nur {len(filtered_df)} passende Artikel gefunden. SAMPLE_SIZE wird angepasst.")
        sample_size = len(filtered_df)
    else:
        sample_size = SAMPLE_SIZE

    sampled_df = filtered_df.sample(n=sample_size, random_state=42).drop(columns=["token_count"])
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    sampled_df.to_csv(SAVE_PATH, index=False)

    print(f"Eval-Subset ({sample_size} Artikel) gespeichert unter: {SAVE_PATH}")

if __name__ == "__main__":
    create_eval_subset()
