import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from src.askdocs.core.config import settings


class BM25Store:

    def __init__(self):
        self._bm25 = None
        self._chunks: List[str] = []
        self._index_path = Path("storage/bm25/index.pkl")

    def build(self, chunks: List[str]) -> None:
        self._chunks = chunks
        tokenized = [chunk.split() for chunk in chunks] #띄어쓰기 기준으로 토크나이징
        self._bm25 = BM25Okapi(tokenized)
        self._save()

    def search(
        self,
        query: str,
        top_k: int = None,
    ) -> List[str]:
        top_k = top_k or settings.top_k

        if self._bm25 is None:
            self._load()

        tokenized_query = query.split()
        scores = self._bm25.get_scores(tokenized_query)

        # 점수 높은 순으로 인덱스 정렬
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [self._chunks[i] for i in top_indices]

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump((self._bm25, self._chunks), f)

    def _load(self) -> None:
        if not self._index_path.exists():
            raise FileNotFoundError("BM25 인덱스가 없습니다. 먼저 문서를 업로드하세요.")
        with open(self._index_path, "rb") as f:
            self._bm25, self._chunks = pickle.load(f)