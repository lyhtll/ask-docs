from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.askdocs.api.routers import documents, chat
from src.askdocs.core.database import engine, Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # 테이블 자동 생성
    yield
    # 앱 종료 시
    await engine.dispose()  # 커넥션 풀 정리


app = FastAPI(
    title="askdocs",
    description="RAG 기반 문서 Q&A 시스템",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router,      prefix="/chat",      tags=["chat"])


@app.get("/health")
async def health():
    return {"status": "ok"}