from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.askdocs.api.routers import documents, chat
from src.askdocs.api.schemas import HealthResponse
from src.askdocs.core.database import engine, Base
from src.askdocs.retrieval.es_store import ESStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    es_store = ESStore()
    await es_store.ensure_index()

    yield

    await es_store.close()
    await engine.dispose()


app = FastAPI(
    title="askdocs",
    description=(
        "## RAG 기반 문서 Q&A 시스템\n\n"
        "문서를 업로드하면 자연어로 질문할 수 있습니다.\n\n"
        "### 파이프라인\n"
        "```\n"
        "문서 업로드 → Parent-Child 청킹 → BGE-M3 임베딩 → pgvector 저장\n"
        "질문 → Router → (HyDE) → Hybrid Search (ES BM25+KNN, RRF)\n"
        "     → CrossEncoder Reranking → LLM (Ollama) → 답변\n"
        "```\n\n"
        "### 지원 파일 형식\n"
        "- PDF (`.pdf`)\n"
        "- 텍스트 (`.txt`, `.md`)\n"
    ),
    version="0.1.0",
    contact={"name": "askdocs", "url": "https://github.com/lyhtll/ask-docs"},
    lifespan=lifespan,
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router,      prefix="/chat",      tags=["chat"])


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스 체크",
    tags=["health"],
)
async def health():
    return {"status": "ok"}