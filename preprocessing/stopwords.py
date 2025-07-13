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
        return _stopword_cache[language]  # Aus Cache zurückgeben
    try:
        nltk.data.find('corpora/stopwords')  # Prüfen ob Stopwords-Korpus vorhanden
    except LookupError:
        nltk.download('stopwords')  # Download falls nicht vorhanden
    stop_words = set(stopwords.words(language))  # Stopwords laden
    _stopword_cache[language] = stop_words  # Cache aktualisieren
    return stop_words

def remove_stopwords(text: str, language: str = 'english') -> str:
    """Entfernt Stoppwörter aus dem Text."""
    stop_words = get_stopwords(language)  # Stopword-Set holen
    words = text.split()  # Text tokenisieren (einfach mit split)
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Filter anwenden
    return ' '.join(filtered_words)  # Ergebnis zurückgeben
