from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.askdocs.core.config import settings
from src.askdocs.models.chunk import Chunk


class VectorStore:

    async def search(
        self,
        query_vector: List[float],
        db: AsyncSession,
        top_k: int = None,
    ) -> List[Chunk]:
        top_k = top_k or settings.top_k

        result = await db.execute(
            select(Chunk)
            .where(Chunk.embedding.is_not(None))          # parent 청크(embedding=None) 제외
            .order_by(Chunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
            .options(joinedload(Chunk.parent))             # parent 즉시 로드 (N+1 방지)
        )
        return result.scalars().all()

    async def search_by_doc(
        self,
        query_vector: List[float],
        doc_id: UUID,
        db: AsyncSession,
        top_k: int = None,
    ) -> List[Chunk]:
        top_k = top_k or settings.top_k

        result = await db.execute(
            select(Chunk)
            .where(Chunk.doc_id == doc_id, Chunk.embedding.is_not(None))
            .order_by(Chunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
            .options(joinedload(Chunk.parent))
        )
        return result.scalars().all()

    async def delete_by_doc(
        self,
        doc_id: UUID,
        db: AsyncSession,
    ) -> None:
        chunks = await db.execute(
            select(Chunk).where(Chunk.doc_id == doc_id)
        )
        for chunk in chunks.scalars().all():
            await db.delete(chunk)
        await db.commit()