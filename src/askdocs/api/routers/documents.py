import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.api.dependencies.db import get_db
from src.askdocs.api.schemas import DeleteResponse, DocumentItem, UploadResponse
from src.askdocs.ingestion.service import IngestionService
from src.askdocs.models.document import Document


router = APIRouter()
ingestion = IngestionService()


@router.post(
    "/",
    response_model=UploadResponse,
    summary="문서 업로드",
    description=(
        "PDF, txt, md 파일을 업로드합니다. "
        "파일 저장 후 **백그라운드에서 청킹 → 임베딩 → DB 저장**이 진행됩니다. "
        "인덱싱이 완료되기 전에 질문하면 해당 문서는 검색되지 않습니다."
    ),
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    upload_dir = Path("documents")
    upload_dir.mkdir(exist_ok=True)
    save_path = upload_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(ingestion.ingest, str(save_path), db)
    return {"message": f"{file.filename} 업로드 완료. 인덱싱 중..."}


@router.get(
    "/",
    response_model=List[DocumentItem],
    summary="문서 목록 조회",
    description="현재 인덱싱된 문서 목록을 반환합니다.",
)
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document))
    docs = result.scalars().all()
    return [{"id": doc.id, "filename": doc.filename} for doc in docs]


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="문서 삭제",
    description="문서와 연결된 모든 청크(임베딩 포함)를 삭제합니다.",
    responses={404: {"description": "문서를 찾을 수 없습니다."}},
)
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    filename = doc.filename
    await ingestion.delete(doc_id, db)
    return {"message": f"{filename} 삭제 완료"}