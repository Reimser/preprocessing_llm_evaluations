"""
Lemmatisierungsfunktionen mit spaCy.
"""

import spacy

_spacy_model = None

def load_spacy_model(model_name: str = 'en_core_web_sm'):
    """Lädt das spaCy-Modell nur einmal."""
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load(model_name)
        except OSError:
            print(f"Modell {model_name} nicht gefunden. Versuche es zu installieren...")
            spacy.cli.download(model_name)
            _spacy_model = spacy.load(model_name)
    return _spacy_model

def lemmatize_text(text: str) -> str:
    """Lemmatisiert den Text mit spaCy."""
    nlp = load_spacy_model()
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])
