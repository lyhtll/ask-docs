from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.core.config import settings
from src.askdocs.models.chunk import Chunk
from src.askdocs.retrieval.es_store import ESStore
from src.askdocs.retrieval.vector_store import VectorStore


class HybridSearch:

    def __init__(self):
        self._vector_store = VectorStore()
        self._es_store     = ESStore()

    async def search(
        self,
        query: str,
        query_vector: List[float],
        db: AsyncSession,
        top_k: int = None,
    ) -> List[Chunk]:
        top_k = top_k or settings.top_k

        vector_results = await self._vector_store.search(query_vector, db, top_k)
        es_chunk_ids   = await self._es_store.search(query, top_k)

        return await self._rrf(vector_results, es_chunk_ids, db, top_k)

    async def _rrf(
        self,
        vector_results: List[Chunk],
        es_chunk_ids: List[str],
        db: AsyncSession,
        top_k: int,
        k: int = 60,
    ) -> List[Chunk]:
        scores: dict[str, float] = {}

        for rank, chunk in enumerate(vector_results):
            cid = str(chunk.id)
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)

        for rank, cid in enumerate(es_chunk_ids):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)

        sorted_ids  = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        id_to_chunk = {str(chunk.id): chunk for chunk in vector_results}

        missing_ids = [cid for cid in sorted_ids if cid not in id_to_chunk]
        if missing_ids:
            result = await db.execute(
                select(Chunk).where(Chunk.id.in_(missing_ids))
            )
            for chunk in result.scalars().all():
                id_to_chunk[str(chunk.id)] = chunk

        return [id_to_chunk[cid] for cid in sorted_ids if cid in id_to_chunk]