import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.askdocs.core.config import settings
from src.askdocs.core.database import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content     = Column(Text, nullable=False)
    embedding   = Column(Vector(settings.embedding_dim), nullable=True)  # parent 청크는 None
    chunk_index = Column(Integer, nullable=False)
    source      = Column(String(500), nullable=False)

    # FK
    doc_id    = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id"), nullable=True)

    # 관계
    document = relationship("Document", back_populates="chunks")
    parent   = relationship("Chunk", remote_side=[id])