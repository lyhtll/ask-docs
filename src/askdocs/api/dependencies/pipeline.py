from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from src.askdocs.pipeline.graph import build_graph


def get_pipeline(db: AsyncSession):
    return build_graph(db)