"""
Funktionen zum Kürzen von Texten auf eine bestimmte Token- oder Satzanzahl – ohne NLTK.
Verwendet den GPT2-Tokenizer von HuggingFace für tokenbasiertes Trunkieren.
"""
import os
os.environ["USE_TF"] = "0"

from transformers import GPT2TokenizerFast

# Initialisiere GPT2-Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def truncate_by_tokens(text: str, max_tokens: int = 512) -> str:
    """
    Kürzt den Text auf eine maximale Anzahl von GPT2-Tokens.

    Args:
        text (str): Eingabetext.
        max_tokens (int): Maximale Anzahl an Tokens.

    Returns:
        str: Gekürzter Text.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_ids = tokens[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def truncate_by_sentences(text: str, max_sentences: int) -> str:
    """
    Kürzt den Text auf eine maximale Anzahl an Sätzen – optional, nur wenn du sentencebasiertes Trunkieren brauchst.

    Args:
        text (str): Eingabetext.
        max_sentences (int): Maximale Anzahl an Sätzen.

    Returns:
        str: Gekürzter Text.
    """
    sentences = text.split(". ")  # primitive Satztrennung, ohne NLTK
    if len(sentences) <= max_sentences:
        return text

    return ". ".join(sentences[:max_sentences]) + "."
