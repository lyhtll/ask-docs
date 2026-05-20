from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.api.dependencies.db import get_db
from src.askdocs.api.dependencies.pipeline import get_pipeline


router = APIRouter()


class ChatRequest(BaseModel):
    query: str


@router.post("/")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    pipeline = get_pipeline(db)

    result = await pipeline.ainvoke({
        "query": request.query,
        "expanded_query": None,
        "query_vector": None,
        "chunks": None,
        "reranked_chunks": None,
        "answer": None,
        "is_complex": None,
    })

    return {"answer": result["answer"]}


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    async def generate():
        from src.askdocs.pipeline.nodes.generate import generate_stream_node
        from src.askdocs.pipeline.nodes.retrieve import embed_query_node, retrieve_node
        from src.askdocs.pipeline.nodes.rerank import rerank_node
        from src.askdocs.pipeline.nodes.route import router_node

        # 상태 초기화
        state = {
            "query": request.query,
            "expanded_query": None,
            "query_vector": None,
            "chunks": None,
            "reranked_chunks": None,
            "answer": None,
            "is_complex": None,
        }

        # 노드 순차 실행
        state = await router_node(state)
        state = await embed_query_node(state)
        state = await retrieve_node(state, db)
        state = await rerank_node(state)

        # 스트리밍 생성
        async for token in generate_stream_node(state):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")