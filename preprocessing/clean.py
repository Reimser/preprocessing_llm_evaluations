"""
Textbereinigungsfunktionen für die Vorverarbeitung.
"""

import re
import unicodedata

def remove_special_chars(text: str) -> str:
    """Entfernt Sonderzeichen und normalisiert den Text."""
    text = unicodedata.normalize('NFKD', text)  # Unicode-Normalisierung
    text = re.sub(r'[^\w\s]', ' ', text)  # Ersetzt Sonderzeichen durch Leerzeichen
    text = re.sub(r'\s+', ' ', text)  # Mehrfache Leerzeichen reduzieren
    return text.strip()  # Führende/trailende Leerzeichen entfernen

def normalize_text(text: str) -> str:
    """Normalisiert den Text (Kleinbuchstaben, etc.)."""
    text = text.lower()  # Alles in Kleinbuchstaben umwandeln
    text = re.sub(r'\s+', ' ', text)  # Mehrfache Leerzeichen reduzieren
    return text.strip()  # Trim whitespace

def clean_text(text: str) -> str:
    """Führt Standard-Textbereinigung aus: Normalisieren und Sonderzeichen entfernen."""
    text = normalize_text(text)  # Erst Kleinbuchstaben + Leerzeichen
    text = remove_special_chars(text)  # Dann Sonderzeichen entfernen
    return text  # Ergebnis zurückgeben
