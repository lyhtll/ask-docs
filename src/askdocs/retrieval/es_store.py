from typing import List

from elasticsearch import AsyncElasticsearch

from src.askdocs.core.config import settings

_INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "korean": {
                "type": "nori",
                "decompound_mode": "mixed",
            }
        }
    }
}

_INDEX_MAPPINGS = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "doc_id":   {"type": "keyword"},
        "content":  {"type": "text", "analyzer": "korean"},
    }
}


class ESStore:

    def __init__(self):
        self._es    = AsyncElasticsearch(settings.es_url)
        self._index = settings.es_index

    async def ensure_index(self) -> None:
        exists = await self._es.indices.exists(index=self._index)
        if not exists:
            await self._es.indices.create(
                index=self._index,
                settings=_INDEX_SETTINGS,
                mappings=_INDEX_MAPPINGS,
            )

    async def index_chunk(self, chunk_id: str, doc_id: str, content: str) -> None:
        await self._es.index(
            index=self._index,
            id=chunk_id,
            document={"chunk_id": chunk_id, "doc_id": doc_id, "content": content},
        )

    async def search(self, query: str, top_k: int = None) -> List[str]:
        top_k = top_k or settings.top_k
        resp = await self._es.search(
            index=self._index,
            query={"match": {"content": {"query": query, "analyzer": "korean"}}},
            size=top_k,
        )
        return [hit["_id"] for hit in resp["hits"]["hits"]]

    async def delete_by_doc(self, doc_id: str) -> None:
        await self._es.delete_by_query(
            index=self._index,
            query={"term": {"doc_id": doc_id}},
        )

    async def close(self) -> None:
        await self._es.close()