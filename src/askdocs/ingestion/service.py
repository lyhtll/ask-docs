from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.core.config import settings
from src.askdocs.ingestion.loader import DocumentLoader
from src.askdocs.ingestion.chunker import Chunker
from src.askdocs.ingestion.embedder import Embedder
from src.askdocs.models.document import Document
from src.askdocs.models.chunk import Chunk


class IngestionService:

    def __init__(self):
        self._loader   = DocumentLoader()
        self._chunker  = Chunker()
        self._embedder = Embedder()

    async def ingest(self, file_path: str, db: AsyncSession) -> Document:
        text = await self._loader.load(file_path)

        doc = Document(
            filename=Path(file_path).name,
            content=text,
            source=file_path,
        )
        db.add(doc)
        await db.flush()

        if settings.chunk_strategy == "parent_child":
            await self._ingest_parent_child(text, file_path, doc.id, db)
        else:
            await self._ingest_flat(text, file_path, doc.id, db)

        await db.commit()
        return doc

    async def _ingest_flat(self, text: str, source: str, doc_id, db: AsyncSession) -> None:
        chunks = self._chunker.chunk(text, strategy=settings.chunk_strategy)
        vectors = await self._embedder.embed(chunks)

        db.add_all([
            Chunk(
                content=chunk_text,
                embedding=vector,
                chunk_index=idx,
                source=source,
                doc_id=doc_id,
            )
            for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors))
        ])

    async def _ingest_parent_child(self, text: str, source: str, doc_id, db: AsyncSession) -> None:
        pairs = self._chunker.chunk_parent_child(text)  # [(parent_text, [child_texts]), ...]

        child_index = 0
        for parent_idx, (parent_text, child_texts) in enumerate(pairs):
            # 1. Parent 청크 저장 (embedding 없음 — 검색 대상 제외, 문맥 제공 전용)
            parent_chunk = Chunk(
                content=parent_text,
                embedding=None,
                chunk_index=parent_idx,
                source=source,
                doc_id=doc_id,
                parent_id=None,
            )
            db.add(parent_chunk)
            await db.flush()  # parent.id 확보

            # 2. Child 청크 임베딩 후 저장
            if not child_texts:
                continue

            vectors = await self._embedder.embed(child_texts)
            db.add_all([
                Chunk(
                    content=child_text,
                    embedding=vector,
                    chunk_index=child_index + i,
                    source=source,
                    doc_id=doc_id,
                    parent_id=parent_chunk.id,
                )
                for i, (child_text, vector) in enumerate(zip(child_texts, vectors))
            ])
            child_index += len(child_texts)