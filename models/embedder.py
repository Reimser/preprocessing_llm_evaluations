"""
Konfiguration und Wrapper für SentenceTransformer-Modelle.
"""

class TextEmbedder:  # Wrapper-Klasse für Sentence-BERT
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatische Geräteerkennung
        else:
            self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)  # Modell laden

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> torch.Tensor:
        return self.model.encode(  # Embeddings für Textliste berechnen
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True
        )

    def compute_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        return cos_sim(embeddings1, embeddings2)  # Kosinus-Ähnlichkeit berechnen

    def save_embeddings(self, embeddings: torch.Tensor, filepath: str):
        torch.save(embeddings, filepath)  # Embeddings in Datei speichern

    def load_embeddings(self, filepath: str) -> torch.Tensor:
        return torch.load(filepath)  # Embeddings aus Datei laden