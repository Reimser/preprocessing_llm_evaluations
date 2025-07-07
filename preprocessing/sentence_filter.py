"""
Funktionen zum Filtern von Sätzen basierend auf verschiedenen Kriterien.
"""

import spacy
from typing import List, Set
import nltk
from nltk.tokenize import sent_tokenize

# Initialisierung für spaCy
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

def filter_by_keywords(text: str, keywords: Set[str]) -> str:
    """Filtert Sätze, die bestimmte Keywords enthalten."""
    sentences = sent_tokenize(text)
    filtered = [
        sent for sent in sentences
        if any(keyword.lower() in sent.lower() for keyword in keywords)
    ]
    return ' '.join(filtered)

def filter_main_sentences(text: str) -> str:
    """Filtert Hauptsätze basierend auf Subjekt und Verb."""
    nlp = load_spacy_model()
    doc = nlp(text)
    main_sentences = []
    
    for sent in doc.sents:
        has_subject = any(token.dep_ == "nsubj" for token in sent)
        has_verb = any(token.pos_ == "VERB" for token in sent)
        if has_subject and has_verb:
            main_sentences.append(sent.text)
    
    return ' '.join(main_sentences)

def filter_by_ner(text: str, entity_types: Set[str]) -> str:
    """Filtert Sätze, die bestimmte Named Entities enthalten."""
    nlp = load_spacy_model()
    doc = nlp(text)
    filtered_sentences = []
    
    for sent in doc.sents:
        entities = [ent.label_ for ent in sent.ents]
        if any(entity in entity_types for entity in entities):
            filtered_sentences.append(sent.text)
    
    return ' '.join(filtered_sentences)
