"""
TF-IDF basierte Textanalyse und -verarbeitung.
"""

from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFAnalyzer:
    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """Initialisiert den TF-IDF Analyzer."""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        self.feature_names = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Berechnet TF-IDF Matrix für die Texte."""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return tfidf_matrix
    
    def get_important_terms(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """Extrahiert die wichtigsten Begriffe aus einem Text."""
        tfidf_matrix = self.vectorizer.transform([text])
        feature_index = tfidf_matrix[0, :].nonzero()[1]
        tfidf_scores = zip(
            feature_index,
            [tfidf_matrix[0, x] for x in feature_index]
        )
        
        # Sortiere nach TF-IDF Score
        sorted_scores = sorted(
            tfidf_scores,
            key=lambda x: x[1],
            reverse=True
        )
        
        # Extrahiere die Top-N Begriffe
        return [
            (self.feature_names[i], score)
            for i, score in sorted_scores[:top_n]
        ]
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten."""
        tfidf_matrix = self.vectorizer.transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    def filter_by_importance(
        self,
        text: str,
        threshold: float = 0.1
    ) -> str:
        """Filtert den Text basierend auf TF-IDF Scores."""
        words = text.split()
        tfidf_matrix = self.vectorizer.transform([text])
        
        # Berechne durchschnittlichen TF-IDF Score pro Wort
        word_scores = {}
        for word in words:
            if word in self.vectorizer.vocabulary_:
                idx = self.vectorizer.vocabulary_[word]
                score = tfidf_matrix[0, idx]
                word_scores[word] = score
        
        # Filtere Wörter mit niedrigem Score
        filtered_words = [
            word for word in words
            if word_scores.get(word, 0) > threshold
        ]
        
        return ' '.join(filtered_words) 