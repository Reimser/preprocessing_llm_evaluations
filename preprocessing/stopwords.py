"""
Funktionen zum Entfernen von Stoppwörtern.
"""

from typing import List, Set
import nltk
from nltk.corpus import stopwords

def get_stopwords(language: str = 'english') -> Set[str]:
    """Lädt die Stoppwörter für die angegebene Sprache."""
    try:
        return set(stopwords.words(language))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words(language))

def remove_stopwords(text: str, language: str = 'english') -> str:
    """Entfernt Stoppwörter aus dem Text."""
    stop_words = get_stopwords(language)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words) 