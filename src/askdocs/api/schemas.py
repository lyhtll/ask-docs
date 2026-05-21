from uuid import UUID

from pydantic import BaseModel, Field


# ── Documents ────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    message: str = Field(..., examples=["report.pdf 업로드 완료. 인덱싱 중..."])


class DocumentItem(BaseModel):
    id: UUID = Field(..., examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"])
    filename: str = Field(..., examples=["report.pdf"])


class DeleteResponse(BaseModel):
    message: str = Field(..., examples=["report.pdf 삭제 완료"])


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        examples=["연차 신청 기한은 언제까지인가요?"],
    )


class SourceItem(BaseModel):
    index: int = Field(..., examples=[1])
    filename: str = Field(..., examples=["report.pdf"])
    content: str = Field(..., examples=["연차는 최소 3일 전까지 신청해야 합니다."])


class ChatResponse(BaseModel):
    answer: str = Field(..., examples=["연차는 최소 3일 전까지 신청해야 합니다."])
    sources: list[SourceItem] = Field(default_factory=list)


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])