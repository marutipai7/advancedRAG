import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embeddings import embed_query
import traceback

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    User the model query_points interface.
    """
    try:
        query_vector = embed_query(query)

        # Using query_points - the moder standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })

        return results

    except Exception as e:
        print(f"[qdrant] search failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        logfire.warning(f"Qdrant search failed: {e}")
        return []