"""
Lemmatisierungsfunktionen mit spaCy.
"""

import spacy
from typing import Optional

def load_spacy_model(model_name: str = 'en_core_web_sm') -> Optional[spacy.language.Language]:
    """Lädt das spaCy-Modell."""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Modell {model_name} nicht gefunden. Versuche es zu installieren...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def lemmatize_text(text: str, model_name: str = 'en_core_web_sm') -> str:
    """Lemmatisiert den Text mit spaCy."""
    nlp = load_spacy_model(model_name)
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized) 