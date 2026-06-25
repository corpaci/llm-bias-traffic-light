import torch
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        return self._model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
