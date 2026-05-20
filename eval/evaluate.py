"""
RAGAS 평가 스크립트

설치:
  pip install ragas datasets

실행:
  python eval/evaluate.py

ground_truth를 직접 작성하기 어려우면 generate_dataset()으로
LLM이 문서에서 QA 쌍을 자동 생성한다.
"""

import asyncio
from typing import Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# ── 수동 평가 데이터셋 ────────────────────────────────────────────────────────
# ground_truth: 정답 기준. 없으면 context_recall 계산 불가.
EVAL_DATA = [
    {
        "question": "연차 신청 기한은?",
        "answer": "최소 3일 전까지 신청해야 합니다.",
        "contexts": ["연차는 최소 3일 전까지 신청해야 합니다. 팀장 승인이 필요합니다."],
        "ground_truth": "최소 3일 전까지 신청해야 한다.",
    },
    # 평가 데이터 추가 ...
]


def run_evaluation(data: list[dict] | None = None) -> dict:
    dataset = Dataset.from_list(data or EVAL_DATA)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = {
        "faithfulness":       round(result["faithfulness"], 4),       # 환각률 역수 (높을수록 좋음)
        "answer_relevancy":   round(result["answer_relevancy"], 4),   # 질문-답변 관련성
        "context_recall":     round(result["context_recall"], 4),     # 필요한 문서를 잘 찾았는가
        "context_precision":  round(result["context_precision"], 4),  # 찾은 문서가 관련 있는가
    }
    return scores


# ── LLM으로 QA 데이터셋 자동 생성 ────────────────────────────────────────────
async def generate_dataset(documents: list[str], n: int = 10) -> list[dict]:
    """문서 목록에서 LLM으로 QA 쌍 자동 생성."""
    from ollama import AsyncClient
    from src.askdocs.core.config import settings

    ollama = AsyncClient(host=settings.ollama_base_url)
    pairs = []

    for doc in documents[:n]:
        prompt = f"""다음 문서에서 질문-답변 쌍 1개를 JSON 형식으로 생성하세요.

문서:
{doc}

출력 형식 (JSON만):
{{"question": "...", "ground_truth": "..."}}"""

        response = await ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        try:
            qa = json.loads(response["message"]["content"])
            pairs.append({
                "question":     qa["question"],
                "answer":       "",       # 실제 파이프라인 실행 후 채워야 함
                "contexts":     [doc],
                "ground_truth": qa["ground_truth"],
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return pairs


if __name__ == "__main__":
    scores = run_evaluation()
    print("\n── RAGAS 평가 결과 ──")
    for metric, score in scores.items():
        print(f"  {metric:<22} {score}")