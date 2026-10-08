import time
import logfire
from app.config import settings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK__DIM = 768    ## all-mpnet-base-v2

_active_model = None
_model_type = None      ## 'gemini' or fallback


def _probe_gemini():
    try:
        model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-2-preview',
                                             google_api_key=settings.GEMINI_API_KEY,
                                             )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (model: models/gemini-embedding-2-preview, 3072-dim)")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {str(e)}. We will use sentence-tansformers fallback (all-mpnet-base-v2, 768-dim) for embeddings.")
        return None

def _load_fallback():
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        logfire.info("Fallback embeddings ready (model: all-mpnet-base-v2, 768-dim)")
        return model
    except Exception as e:
        logfire.error(f"Fallback embeddings load failed: {str(e)}. Embeddings will not be available.")
        return None

def _init():
    global _active_model, _model_type
    _active_model = _probe_gemini()
    if _active_model is not None:
        _model_type = 'gemini'
    else:
        _active_model = _load_fallback()
        if _active_model is not None:
            _model_type = 'fallback'
        else:
            logfire.error("No embeddings model available. Please check your configuration.")

# def get_embedding_dim() -> int:
#     _init()
#     return _GEMINI_DIM if _model_type == 'gemini' else _FALLBACK__DIM

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == 'gemini':
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "qouta", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini embeddings rate limit hit. Retrying in {wait} seconds..."
                        f" (attempt {attempt + 1}/4)"
                        )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embeddings failed: {str(e)}")
                    raise
        raise RuntimeError("Gemini embeddings failed after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False, convert_to_numpy=True).tolist()

# def embed_query(query: str) -> list[float]:
#     _init()
#     if _model_type == 'gemini':
#         return _active_model.embed_query(query)
#     else:
#         return _active_model.encode([query])[0].tolist()

# def embed_texts(texts: list[str]) -> list[list[float]]:
#     _init()
#     all_embeddings: list[list[float]] = []
#     for i in range(0, len(texts), BATCH_SIZE):
#         batch = texts[i:i + BATCH_SIZE]
#         with logfire.span("Embed batch ", model=_model_type, start=i, size=len(batch)):
#             embeddings = _embed_batch(batch)
#             all_embeddings.extend(embeddings)
#     return all_embeddings


_model = SentenceTransformer("all-mpnet-base-v2")

def embed_query(query: str) -> list[float]:
    return _model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    return _model.encode(
        texts,
        batch_size=50,
        show_progress_bar=False,
        convert_to_numpy=True
    ).tolist()

def get_embedding_dim() -> int:
    return 768