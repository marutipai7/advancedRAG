import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.gaurdrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

_rails = LLMRails | None = None

def initialize_rails() -> None:
    global _rails

    gaurd_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    _rails = LLMRails(
        config=config,
        llm=gaurd_llm,
        indicators=RAIL_INDICATORS,
    )

    logfire.info("🛡️ NeMo Guardrails initialized.")

def gaurd(message: str) -> tuple[bool, str | None]:
    if _rails is None:
        logfire.warning("🛡️ Guardrails not initialized yet.")
        return False, None

    with logfire.span("🛡️ Gaurdrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        content = result.get("content", "") if isinstance(result, dict) else result

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content      # caller returns immediately, skips RAG

        logfire.info("✅ Guardrails passed.")
        return False, None            # caller proceeds to LangGraph