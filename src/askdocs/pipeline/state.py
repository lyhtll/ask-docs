from typing import List, Optional
from typing_extensions import TypedDict

from src.askdocs.models.chunk import Chunk


class GraphState(TypedDict):
    # 입력
    query: str                          # 사용자 질문

    # Query Expansion
    expanded_query: Optional[str]       # HyDE로 확장된 쿼리

    # Retrieval
    query_vector: Optional[List[float]] # 쿼리 임베딩 벡터
    chunks: Optional[List[Chunk]]       # Hybrid Search 결과

    # Reranking
    reranked_chunks: Optional[List[Chunk]]  # 재랭킹 결과

    # Generation
    answer: Optional[str]               # 최종 답변

    # 메타
    is_complex: Optional[bool]          # 복잡한 질문 여부 (분기용)