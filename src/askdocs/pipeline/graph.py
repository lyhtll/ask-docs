from langgraph.graph import StateGraph, END

from src.askdocs.pipeline.nodes.hyde import hyde_node
from src.askdocs.pipeline.state import GraphState
from src.askdocs.pipeline.nodes.route import router_node
from src.askdocs.pipeline.nodes.retrieve import embed_query_node, retrieve_node
from src.askdocs.pipeline.nodes.rerank import rerank_node
from src.askdocs.pipeline.nodes.generate import generate_node


def build_graph(db):

    async def _retrieve(s):
        return await retrieve_node(s, db)

    graph = StateGraph(GraphState)

    # ── 노드 등록 ──
    graph.add_node("분기", router_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("embed_query", embed_query_node)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    # ── 진입점 ──
    graph.set_entry_point("분기")

    # ── 조건부 엣지 ──
    graph.add_conditional_edges(
        "분기",
        lambda state: "complex" if state["is_complex"] else "simple",
        {
            "simple":  "embed_query",
            "complex": "hyde",
        }
    )

    # ── 일반 엣지 ──
    graph.add_edge("hyde", "embed_query")
    graph.add_edge("embed_query", "retrieve")
    graph.add_edge("retrieve",    "rerank")
    graph.add_edge("rerank",      "generate")
    graph.add_edge("generate",    END)

    return graph.compile()