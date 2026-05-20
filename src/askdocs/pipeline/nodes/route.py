from ollama import AsyncClient

from askdocs.core.config import settings
from askdocs.pipeline.state import GraphState


ollama = AsyncClient(host=settings.ollama_base_url)


async def router_node(state: GraphState) -> GraphState:
    query = state["query"]

    prompt = f"""다음 질문이 단순한지 복잡한지 판단해줘.

단순(simple): 단일 사실 조회
예시) "연차 신청 기한은?", "병가 규정 알려줘"

복잡(complex): 비교, 분석, 다단계 추론 필요
예시) "연차랑 병가 차이점은?", "휴가 제도를 분석해줘"

질문: {query}

simple 또는 complex 중 하나만 답해:"""

    response = await ollama.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
    )

    is_complex = "complex" in response["message"]["content"].lower()
    return {**state, "is_complex": is_complex}