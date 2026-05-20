from src.askdocs.pipeline.state import GraphState
from src.askdocs.reranking.cross_encoder import Reranker


reranker = Reranker()


async def rerank_node(state: GraphState) -> GraphState:
    reranked_chunks = reranker.rerank(
        query=state["query"],
        chunks=state["chunks"],
    )
    return {**state, "reranked_chunks": reranked_chunks}