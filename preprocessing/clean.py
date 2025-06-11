"""
Textbereinigungsfunktionen für die Vorverarbeitung.
"""

import re
import unicodedata

def remove_special_chars(text: str) -> str:
    """Entfernt Sonderzeichen und normalisiert den Text."""
    # Normalisiere Unicode-Zeichen
    text = unicodedata.normalize('NFKD', text)
    
    # Ersetze Sonderzeichen durch Leerzeichen
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Entferne mehrfache Leerzeichen
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def normalize_text(text: str) -> str:
    """Normalisiert den Text (Kleinbuchstaben, etc.)."""
    # Konvertiere zu Kleinbuchstaben
    text = text.lower()
    
    # Entferne mehrfache Leerzeichen
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip() 