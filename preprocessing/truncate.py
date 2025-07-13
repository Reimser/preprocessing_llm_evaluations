"""
Funktionen zum Kürzen von Texten auf eine bestimmte Token- oder Satzanzahl – ohne NLTK.
Verwendet den GPT2-Tokenizer von HuggingFace für tokenbasiertes Trunkieren.
"""
import os
os.environ["USE_TF"] = "0"  # Deaktiviert TensorFlow-Nutzung in transformers

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # GPT2-Tokenizer laden

def truncate_by_tokens(text: str, max_tokens: int = 512) -> str:
    """Kürzt den Text auf max. Anzahl GPT2-Tokens."""
    tokens = tokenizer.encode(text)  # Text tokenisieren
    if len(tokens) <= max_tokens:
        return text  # Wenn kurz genug: Rückgabe
    truncated_ids = tokens[:max_tokens]  # Token-Limit anwenden
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)  # Rückübersetzung
