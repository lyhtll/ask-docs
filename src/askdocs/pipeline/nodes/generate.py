from pathlib import Path

from ollama import AsyncClient

from src.askdocs.core.config import settings
from src.askdocs.pipeline.state import GraphState


ollama = AsyncClient(host=settings.ollama_base_url)


def _context_text(chunk) -> str:
    # parent_child 패턴: child로 검색, parent 전문을 LLM에 전달
    return chunk.parent.content if chunk.parent else chunk.content


def _build_sources(chunks: list) -> list[dict]:
    return [
        {
            "index": idx + 1,
            "filename": Path(chunk.source).name,
            "content": chunk.content,  # 검색에 매칭된 실제 구절
        }
        for idx, chunk in enumerate(chunks)
    ]


async def generate_node(state: GraphState) -> GraphState:
    chunks = state["reranked_chunks"]
    context = "\n\n".join([
        f"[{idx+1}] {_context_text(chunk)}"
        for idx, chunk in enumerate(chunks)
    ])

    prompt = f"""다음 문서를 참고해서 질문에 답하세요.
문서를 벗어난 내용은 답하지 마세요.

문서:
{context}

질문: {state["query"]}
답변:"""

    response = await ollama.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
    )

    return {**state, "answer": response["message"]["content"], "sources": _build_sources(chunks)}


async def generate_stream_node(state: GraphState):
    context = "\n\n".join([
        f"[{idx+1}] {_context_text(chunk)}"
        for idx, chunk in enumerate(state["reranked_chunks"])
    ])

    prompt = f"""다음 문서를 참고해서 질문에 답하세요.

문서:
{context}

질문: {state["query"]}
답변:"""

    # 스트리밍
    async for part in await ollama.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        yield part["message"]["content"]