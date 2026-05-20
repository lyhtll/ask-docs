from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from src.askdocs.core.config import settings

engine = create_async_engine(
    settings.db_url,
    echo=True,          # SQL 쿼리 로그 출력 (개발 중 디버깅용)
    pool_size=10,       # 커넥션 풀 크기
    max_overflow=20,    # 풀 초과 시 추가 허용 커넥션
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    pass