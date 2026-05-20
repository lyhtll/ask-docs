from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.api.dependencies.db import get_db
from src.askdocs.api.dependencies.pipeline import get_pipeline
from src.askdocs.api.schemas import ChatRequest, ChatResponse


router = APIRouter()


@router.post(
    "/",
    response_model=ChatResponse,
    summary="질문 답변",
    description=(
        "업로드된 문서를 기반으로 질문에 답합니다. "
        "내부적으로 **Router → (HyDE) → Hybrid Search → CrossEncoder Reranking → LLM** 순으로 실행됩니다. "
        "단순 질문은 HyDE 없이 바로 검색, 복잡한 질문은 HyDE로 쿼리를 확장한 뒤 검색합니다."
    ),
)
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


@router.post(
    "/stream",
    summary="질문 답변 (스트리밍)",
    description=(
        "응답을 토큰 단위로 스트리밍합니다. "
        "`Content-Type: text/plain`으로 반환되며, "
        "클라이언트는 청크 단위로 수신해 실시간으로 표시할 수 있습니다."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/plain": {"example": "연차는 최소 3일 전까지 신청해야 합니다."}},
            "description": "스트리밍 텍스트 응답",
        }
    },
)
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    async def generate():
        from src.askdocs.pipeline.nodes.generate import generate_stream_node
        from src.askdocs.pipeline.nodes.retrieve import embed_query_node, retrieve_node
        from src.askdocs.pipeline.nodes.rerank import rerank_node
        from src.askdocs.pipeline.nodes.route import router_node

        state = {
            "query": request.query,
            "expanded_query": None,
            "query_vector": None,
            "chunks": None,
            "reranked_chunks": None,
            "answer": None,
            "is_complex": None,
        }

        state = await router_node(state)
        state = await embed_query_node(state)
        state = await retrieve_node(state, db)
        state = await rerank_node(state)

        async for token in generate_stream_node(state):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")