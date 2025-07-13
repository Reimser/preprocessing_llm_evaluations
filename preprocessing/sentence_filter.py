"""
Funktionen zum Filtern von Sätzen basierend auf verschiedenen Kriterien.
"""

import spacy
from typing import List, Set
import nltk
from nltk.tokenize import sent_tokenize

# Initialisierung für spaCy
_spacy_model = None  # Cache für spaCy-Modell

def load_spacy_model(model_name: str = 'en_core_web_sm'):
    """Lädt das spaCy-Modell nur einmal."""
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load(model_name)  # Modell laden
        except OSError:
            print(f"Modell {model_name} nicht gefunden. Versuche es zu installieren...")
            spacy.cli.download(model_name)  # Modell herunterladen
            _spacy_model = spacy.load(model_name)  # Nach Download laden
    return _spacy_model

def filter_main_sentences(text: str) -> str:
    """Filtert Hauptsätze basierend auf Subjekt und Verb."""
    nlp = load_spacy_model()
    doc = nlp(text)
    main_sentences = []
    for sent in doc.sents:
        has_subject = any(token.dep_ == "nsubj" for token in sent)  # Subjektprüfung
        has_verb = any(token.pos_ == "VERB" for token in sent)  # Verbprüfung
        if has_subject and has_verb:
            main_sentences.append(sent.text)
    return ' '.join(main_sentences)  # Hauptsätze zusammenführen
