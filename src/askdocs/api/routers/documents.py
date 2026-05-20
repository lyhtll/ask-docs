from http.client import HTTPException

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.api.dependencies.db import get_db
from src.askdocs.ingestion.service import IngestionService
from src.askdocs.models.document import Document
from sqlalchemy import select
import shutil
from pathlib import Path


router = APIRouter()
ingestion = IngestionService()


@router.post("/")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    # 파일 저장
    save_path = f"documents/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 백그라운드에서 인덱싱
    background_tasks.add_task(ingestion.ingest, save_path, db)

    return {"message": f"{file.filename} 업로드 완료. 인덱싱 중..."}


@router.get("/")
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document))
    docs = result.scalars().all()
    return [{"id": str(doc.id), "filename": doc.filename} for doc in docs]


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    await db.delete(doc)
    await db.commit()
    return {"message": f"{doc.filename} 삭제 완료"}