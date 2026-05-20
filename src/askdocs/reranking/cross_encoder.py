from typing import List

from sentence_transformers import CrossEncoder

from src.askdocs.core.config import settings
from src.askdocs.models.chunk import Chunk


class Reranker:

    def __init__(self):
        self._model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = None,
    ) -> List[Chunk]:
        top_k = top_k or settings.rerank_top_k

        if not chunks:
            return []

        # 쿼리-청크 쌍 만들기
        pairs = [(query, chunk.content) for chunk in chunks]

        # 점수 계산
        scores = self._model.predict(pairs)

        # 점수 높은 순으로 청크 정렬
        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        # 상위 top_k개만 반환
        return [chunk for _, chunk in ranked[:top_k]]