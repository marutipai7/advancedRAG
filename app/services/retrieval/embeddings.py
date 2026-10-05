import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK__DIM = 768    ## all-mpnet-base-v2

_active_model = None
_model_type = None      ## 'gemini' or fallback


def _probe_gemini():
    pass

def _load_fallback():
    pass

def _init():
    pass

def get_embedding_dim() -> int:
    pass

def _embed_batch(batch: list[str]) -> list[list[float]]:
    pass

def embed_query(query: str) -> list[float]:
    pass

def embed_texts(texts: list[str]) -> list[list[float]]:
    pass