from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from typing import List


class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu", cache_folder: str = None):
        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=cache_folder
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, batch_size=8, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True
        )
        return embedding.tolist()

