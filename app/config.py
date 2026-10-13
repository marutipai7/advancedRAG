import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    # Groq Reasoning Engine (Llama 3.3)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = "openai/gpt-oss-20b"

    # Portkey LLM Gateway
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    GROQ_SLUG = "advancedrag"
    GROQ_SLUG_2 = "rag2"

    # Qdrant Vector DB Settings 
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "enterprise_rag"

    # === Observability ===
    # LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
    # LOGFIRE_SERVICE_NAME = os.getenv("LOGFIRE_SERVICE_NAME")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



settings = Settings()