import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_services import rerank_documents

def retriever_node(state: AgentState):
    """
    The retriever determines if a seach is needed based on the entire conversation
    """
    query = state["current_query"]

    with logfire.span("Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for : {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Qdrant")

        doc_contents = [doc['content'] for doc in raw_results]

        with logfire.span("Semantic reranking"):
            reranked_results = rerank_documents(query, doc_contents, top_n=5)
            logfire.info(f"Reranking Complete. Kept top 5 most relevant chunks")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_results]

    return {
        "documents": formatted_docs,
        "status": f"Found technical context",
        "plan": state["plan"] + ["Context Retrieved"],
    }