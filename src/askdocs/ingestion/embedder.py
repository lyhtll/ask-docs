from typing import List

from sentence_transformers import SentenceTransformer

from src.askdocs.core.config import settings


class Embedder:

    def __init__(self):
        self._model = SentenceTransformer(settings.embedding_model)
        # "BAAI/bge-m3" → 처음 실행 시 모델 자동 다운로드

    async def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=settings.show_progress_bar,
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()