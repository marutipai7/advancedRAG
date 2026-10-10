import logfire
from flashrank import Ranker, RerankRequest
import time

_ranker: Ranker | None = None

def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank enginer lazily
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking
    """
    global _ranker
    if _ranker is None:
        logfire.info("Initializing FlashRank ranker")
        try:
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()

    return _ranker

def rerank_documents(query:str, documents:list[str], top_n:int = 5) -> list[str]:
    """
    why Flashrank?
    Standard Vector Search (Cosine Similarity) is fast but mathematically "fuzzy"
    FlashRank uses a Cross-Encoder approach which is much more precise but usually
    FlashRank solves this by using highly optimized, quantized ONNX models locally.

    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"[Reranker] Sending {len(documents)} documents for reranking...")

    try:
        ranker = _get_ranker()

        passages = [
            {
                "id": i,
                "text": doc,
            }
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        reranked_documents = []
        for res in results[:top_n]:
            reranked_documents.append(res["text"])

        duration = time.time() - start_time
        top_score = results[0]["score"] if results else 0
        logfire.info(f"[Reranker] Reranked {len(documents)} documents in {duration:.2f} seconds. Top Semantic score: {top_score:.4f}")

        return reranked_documents

    except Exception as e:
        logfire.error(f"[Reranker] Error reranking documents: {str(e)}")
        print(f"[reranker] failed: {type(e).__name__}: {e}")
        return documents[:top_n] 