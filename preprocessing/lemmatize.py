import spacy  # spaCy für Lemmatisierung

_spacy_model = None  # Cache für das geladene spaCy-Modell

def load_spacy_model(model_name: str = 'en_core_web_sm'):
    """Lädt das spaCy-Modell nur einmal."""
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load(model_name)  # Versucht Modell zu laden
        except OSError:
            print(f"Modell {model_name} nicht gefunden. Versuche es zu installieren...")
            spacy.cli.download(model_name)  # Automatische Installation
            _spacy_model = spacy.load(model_name)  # Laden nach Download
    return _spacy_model  # Gibt geladenes Modell zurück

def lemmatize_text(text: str) -> str:
    """Lemmatisiert den Text mit spaCy."""
    nlp = load_spacy_model()  # Modell laden (mit Cache)
    doc = nlp(text)  # Text verarbeiten
    return ' '.join([token.lemma_ for token in doc])  # Lemmata extrahieren und zurückgeben
