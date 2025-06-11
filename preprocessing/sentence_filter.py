"""
Funktionen zum Filtern von Sätzen basierend auf verschiedenen Kriterien.
"""

import spacy
from typing import List, Set
import nltk
from nltk.tokenize import sent_tokenize

def filter_by_keywords(text: str, keywords: Set[str]) -> str:
    """Filtert Sätze, die bestimmte Keywords enthalten."""
    sentences = sent_tokenize(text)
    filtered_sentences = [
        sent for sent in sentences
        if any(keyword.lower() in sent.lower() for keyword in keywords)
    ]
    return ' '.join(filtered_sentences)

def filter_main_sentences(text: str, nlp) -> str:
    """Filtert Hauptsätze basierend auf syntaktischer Komplexität."""
    doc = nlp(text)
    main_sentences = []
    
    for sent in doc.sents:
        # Einfache Heuristik: Sätze mit Subjekt und Verb
        has_subject = any(token.dep_ == "nsubj" for token in sent)
        has_verb = any(token.pos_ == "VERB" for token in sent)
        
        if has_subject and has_verb:
            main_sentences.append(sent.text)
    
    return ' '.join(main_sentences)

def filter_by_ner(text: str, nlp, entity_types: Set[str]) -> str:
    """Filtert Sätze basierend auf Named Entity Recognition."""
    doc = nlp(text)
    filtered_sentences = []
    
    for sent in doc.sents:
        entities = [ent.label_ for ent in sent.ents]
        if any(entity_type in entities for entity_type in entity_types):
            filtered_sentences.append(sent.text)
    
    return ' '.join(filtered_sentences) 