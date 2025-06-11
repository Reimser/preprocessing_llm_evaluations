"""
Konfiguration und Wrapper für SentenceTransformer-Modelle.
"""

from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class TextEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """Initialisiert den TextEmbedder mit einem SentenceTransformer-Modell."""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True
    ) -> torch.Tensor:
        """Generiert Embeddings für eine Liste von Texten."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True
        )
    
    def compute_similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Berechnet die Ähnlichkeit zwischen zwei Embedding-Matrizen."""
        return cos_sim(embeddings1, embeddings2)
    
    def save_embeddings(
        self,
        embeddings: torch.Tensor,
        filepath: str
    ):
        """Speichert Embeddings in einer Datei."""
        torch.save(embeddings, filepath)
    
    def load_embeddings(
        self,
        filepath: str
    ) -> torch.Tensor:
        """Lädt Embeddings aus einer Datei."""
        return torch.load(filepath) 