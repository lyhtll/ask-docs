import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.askdocs.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename   = Column(String(255), nullable=False)
    content    = Column(Text, nullable=True)        # 원본 텍스트
    source     = Column(String(500), nullable=False) # 파일 경로
    created_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")