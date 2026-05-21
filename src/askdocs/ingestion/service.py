from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.core.config import settings
from src.askdocs.ingestion.loader import DocumentLoader
from src.askdocs.ingestion.chunker import Chunker
from src.askdocs.ingestion.embedder import Embedder
from src.askdocs.models.document import Document
from src.askdocs.models.chunk import Chunk
from src.askdocs.retrieval.es_store import ESStore


class IngestionService:

    def __init__(self):
        self._loader   = DocumentLoader()
        self._chunker  = Chunker()
        self._embedder = Embedder()
        self._es_store = ESStore()

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

    async def delete(self, doc_id: str, db: AsyncSession) -> None:
        await self._es_store.delete_by_doc(doc_id)
        doc = await db.get(Document, doc_id)
        if doc:
            await db.delete(doc)
            await db.commit()

    async def _ingest_flat(self, text: str, source: str, doc_id, db: AsyncSession) -> None:
        chunks = self._chunker.chunk(text, strategy=settings.chunk_strategy)
        vectors = await self._embedder.embed(chunks)

        chunk_objs = [
            Chunk(
                content=chunk_text,
                embedding=vector,
                chunk_index=idx,
                source=source,
                doc_id=doc_id,
            )
            for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors))
        ]
        db.add_all(chunk_objs)
        await db.flush()

        for chunk in chunk_objs:
            await self._es_store.index_chunk(str(chunk.id), str(doc_id), chunk.content)

    async def _ingest_parent_child(self, text: str, source: str, doc_id, db: AsyncSession) -> None:
        pairs = self._chunker.chunk_parent_child(text)

        child_index = 0
        for parent_idx, (parent_text, child_texts) in enumerate(pairs):
            parent_chunk = Chunk(
                content=parent_text,
                embedding=None,
                chunk_index=parent_idx,
                source=source,
                doc_id=doc_id,
                parent_id=None,
            )
            db.add(parent_chunk)
            await db.flush()

            if not child_texts:
                continue

            vectors = await self._embedder.embed(child_texts)
            child_objs = [
                Chunk(
                    content=child_text,
                    embedding=vector,
                    chunk_index=child_index + i,
                    source=source,
                    doc_id=doc_id,
                    parent_id=parent_chunk.id,
                )
                for i, (child_text, vector) in enumerate(zip(child_texts, vectors))
            ]
            db.add_all(child_objs)
            await db.flush()

            for chunk in child_objs:
                await self._es_store.index_chunk(str(chunk.id), str(doc_id), chunk.content)

            child_index += len(child_texts)