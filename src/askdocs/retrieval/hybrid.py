from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.core.config import settings
from src.askdocs.models.chunk import Chunk
from src.askdocs.retrieval.bm25_store import BM25Store
from src.askdocs.retrieval.vector_store import VectorStore


class HybridSearch:

    def __init__(self):
        self._vector_store = VectorStore()
        self._bm25_store   = BM25Store()

    async def search(
        self,
        query: str,
        query_vector: List[float],
        db: AsyncSession,
        top_k: int = None,
    ) -> List[Chunk]:
        top_k = top_k or settings.top_k

        # 1. 각각 검색
        vector_results = await self._vector_store.search(query_vector, db, top_k)
        bm25_results   = self._bm25_store.search(query, top_k)

        # 2. RRF로 결합
        return self._rrf(vector_results, bm25_results, top_k)

    def _rrf(
        self,
        vector_results: List[Chunk],
        bm25_results: List[str],
        top_k: int,
        k: int = 60,        # RRF 상수
    ) -> List[Chunk]:

        scores = {}

        # vector 결과 점수 계산
        for rank, chunk in enumerate(vector_results):
            chunk_id = str(chunk.id)
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # bm25 결과 점수 계산
        bm25_contents = {chunk.content: chunk for chunk in vector_results}
        for rank, content in enumerate(bm25_results):
            if content in bm25_contents:
                chunk_id = str(bm25_contents[content].id)
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # 점수 높은 순으로 정렬
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        # Chunk 객체로 변환해서 반환
        id_to_chunk = {str(chunk.id): chunk for chunk in vector_results}
        return [
            id_to_chunk[chunk_id]
            for chunk_id in sorted_ids[:top_k]
            if chunk_id in id_to_chunk
        ]