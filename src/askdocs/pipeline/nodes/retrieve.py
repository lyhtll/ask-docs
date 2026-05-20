from src.askdocs.ingestion.embedder import Embedder
from src.askdocs.pipeline.state import GraphState
from src.askdocs.retrieval.hybrid import HybridSearch


embedder = Embedder()
hybrid_search = HybridSearch()


async def embed_query_node(state: GraphState) -> GraphState:
    # expanded_query 있으면 그걸 쓰고, 없으면 원래 query 사용
    query = state.get("expanded_query") or state["query"]

    query_vector = await embedder.embed_query(query)

    return {**state, "query_vector": query_vector}


async def retrieve_node(state: GraphState, db) -> GraphState:
    chunks = await hybrid_search.search(
        query=state["query"],
        query_vector=state["query_vector"],
        db=db,
    )
    return {**state, "chunks": chunks}