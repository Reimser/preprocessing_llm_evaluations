"""
Funktionen zum Kürzen von Texten auf eine bestimmte Token-Anzahl.
"""

from typing import List
import nltk
from nltk.tokenize import word_tokenize

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Kürzt den Text auf eine maximale Anzahl von Tokens."""
    tokens = word_tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return ' '.join(truncated_tokens)

def truncate_by_sentences(text: str, max_sentences: int) -> str:
    """Kürzt den Text auf eine maximale Anzahl von Sätzen."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    
    truncated_sentences = sentences[:max_sentences]
    return ' '.join(truncated_sentences) 