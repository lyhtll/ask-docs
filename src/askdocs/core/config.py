from pydantic.v1 import BaseSettings, Field


class Settings(BaseSettings):
    # PostgreSQL
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="askdocs")
    db_user: str = Field(default="askdocs")
    db_password: str = Field(default="askdocs")

    # 임베딩
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dim: int = Field(default=1024)
    show_progress_bar: bool = Field(default=True)

    # 청킹
    chunk_strategy: str = Field(default="parent_child")  # recursive | document_structure | parent_child
    chunk_size: int = Field(default=500)        # parent 청크 크기
    chunk_overlap: int = Field(default=50)
    child_chunk_size: int = Field(default=100)  # child 청크 크기
    child_chunk_overlap: int = Field(default=20)

    # 검색
    top_k: int = Field(default=50)       # 1차 검색 (two-stage: recall 확보)
    rerank_top_k: int = Field(default=5) # 재랭킹 후 LLM에 전달

    # Hybrid Search 가중치
    bm25_weight: float = Field(default=0.4)
    knn_weight: float = Field(default=0.6)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="gemma3")

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()