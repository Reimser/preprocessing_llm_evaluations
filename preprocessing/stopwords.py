"""
Funktionen zum Entfernen von Stoppwörtern.
"""

from typing import Set
import nltk
from nltk.corpus import stopwords

# Cache für Stopwords, um mehrfaches Laden zu vermeiden
_stopword_cache = {}

def get_stopwords(language: str = 'english') -> Set[str]:
    """Lädt und cached die Stoppwörter für die angegebene Sprache."""
    if language in _stopword_cache:
        return _stopword_cache[language]
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    stop_words = set(stopwords.words(language))
    _stopword_cache[language] = stop_words
    return stop_words

def remove_stopwords(text: str, language: str = 'english') -> str:
    """Entfernt Stoppwörter aus dem Text."""
    stop_words = get_stopwords(language)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
