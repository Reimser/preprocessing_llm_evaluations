"""
Vergleicht Embeddings von verschiedenen Textvarianten und Goldstandard-Zusammenfassungen.
"""

import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingComparator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialisiert den Embedding-Comparator mit einem SentenceTransformer-Modell."""
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generiert Embeddings für einen Text."""
        return self.model.encode(text)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Texten."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def compare_with_goldstandard(
        self,
        summaries: Dict[str, str],
        goldstandard: str
    ) -> Dict[str, float]:
        """Vergleicht verschiedene Zusammenfassungen mit dem Goldstandard."""
        similarities = {}
        for variant_name, summary in summaries.items():
            similarity = self.compute_similarity(summary, goldstandard)
            similarities[variant_name] = similarity
        return similarities
    
    def compare_variants(
        self,
        summaries: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """Vergleicht alle Varianten miteinander."""
        variants = list(summaries.keys())
        similarity_matrix = {}
        
        for var1 in variants:
            similarity_matrix[var1] = {}
            for var2 in variants:
                if var1 != var2:
                    similarity = self.compute_similarity(
                        summaries[var1],
                        summaries[var2]
                    )
                    similarity_matrix[var1][var2] = similarity
        
        return similarity_matrix 