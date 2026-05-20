from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.askdocs.core.config import settings


class Chunker:
    def __init__(self):
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._recursive = self._parent_splitter  # 하위 호환

    def chunk(self, text: str, strategy: str = "recursive") -> List[str]:
        if strategy == "recursive":
            return self._chunk_recursive(text)
        elif strategy == "document_structure":
            return self._chunk_document_structure(text)
        else:
            raise ValueError(f"지원하지 않는 청킹 전략입니다: {strategy}")

    def chunk_parent_child(self, text: str) -> List[Tuple[str, List[str]]]:
        """parent 청크 → child 청크 목록 쌍 반환.

        검색은 child(소형)로, LLM 컨텍스트는 parent(대형)로 전달하는 패턴.
        """
        parents = self._parent_splitter.split_text(text)
        return [
            (parent, self._child_splitter.split_text(parent))
            for parent in parents
        ]

    def _chunk_recursive(self, text: str) -> List[str]:
        return self._recursive.split_text(text)

    def _chunk_document_structure(self, text: str) -> List[str]:
        # 마크다운 헤더 기준으로 분리
        chunks = []
        current_chunk = []
        current_header = ""

        for line in text.split("\n"):
            if line.startswith("#"):  # 헤더 발견
                if current_chunk:  # 이전 청크 저장
                    chunks.append(
                        current_header + "\n" + "\n".join(current_chunk)
                    )
                current_header = line  # 새 헤더 저장
                current_chunk = []
            else:
                current_chunk.append(line)

        if current_chunk:  # 마지막 청크 저장
            chunks.append(
                current_header + "\n" + "\n".join(current_chunk)
            )

        return chunks if chunks else self._chunk_recursive(text)