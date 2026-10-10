import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_services import rerank_documents

def retriever_node(state: AgentState):
    query = state["current_query"]

    with logfire.span("Knowledge Retrieval"):
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Qdrant returned {len(raw_results)} candidates for: {query!r}")
        print(f"[retriever] raw={len(raw_results)} query={query!r}")

        if not raw_results:
            return {
                "documents": [],
                "status": "No documents found",
                "plan": state["plan"] + ["Retrieval: 0 results from Qdrant"],
            }

        doc_contents = [doc["content"] for doc in raw_results]
        reranked = rerank_documents(query, doc_contents, top_n=5)
        print(f"[retriever] reranked={len(reranked)}")

    return {
        "documents": [f"CONTENT: {d}" for d in reranked],
        "status": "Found technical context",
        "plan": state["plan"] + [f"Context Retrieved ({len(reranked)} chunks)"],
    }