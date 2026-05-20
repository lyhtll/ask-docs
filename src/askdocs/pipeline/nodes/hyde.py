from ollama import AsyncClient

from src.askdocs.core.config import settings
from src.askdocs.pipeline.state import GraphState


ollama = AsyncClient(host=settings.ollama_base_url)


async def hyde_node(state: GraphState) -> GraphState:
    query = state["query"]

    prompt = f"""다음 질문에 대한 답변이 담긴 문서를 작성해줘.
실제 문서처럼 구체적으로 작성해줘.

질문: {query}

문서:"""

    response = await ollama.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
    )

    # 가상 문서를 expanded_query로 저장
    expanded_query = response["message"]["content"]
    return {**state, "expanded_query": expanded_query}