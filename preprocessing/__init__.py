from preprocessing.clean import remove_special_chars, normalize_text
from preprocessing.stopwords import remove_stopwords
from preprocessing.truncate import truncate_by_tokens, truncate_by_sentences
from preprocessing.lemmatize import lemmatize_text
from preprocessing.sentence_filter import (
    filter_by_keywords,
    filter_main_sentences,
    filter_by_ner
)
import nltk
nltk.download('punkt')

# Wrapper für truncate_by_tokens mit Standardwert
def truncate_tokens_default(text: str) -> str:
    return truncate_by_tokens(text, max_tokens=512)



STRATEGIES = {
    "remove_special_chars": remove_special_chars,
    "normalize_text": normalize_text,
    "remove_stopwords": remove_stopwords,
    "lemmatize": lemmatize_text,
    "truncate_tokens": truncate_tokens_default,
    "truncate_sentences": truncate_by_sentences,
    "filter_main_sentences": filter_main_sentences,
}

def apply_preprocessing(text: str, steps: list) -> str:
    for step in steps:
        func = STRATEGIES.get(step)
        if func:
            text = func(text)
    return text
