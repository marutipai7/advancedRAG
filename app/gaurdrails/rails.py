import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.llm.providers import register_llm_provider
from app.config import settings
from app.gaurdrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS
import re
_rails : LLMRails | None = None

def initialize_rails() -> None:
    global _rails

    gaurd_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0.1,
    )

    # register_llm_provider("groq", lambda **kwargs: gaurd_llm)

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    _rails = LLMRails(
        config=config,
        llm=gaurd_llm,
        # indicators=RAIL_INDICATORS,
    )

    logfire.info("🛡️ NeMo Guardrails initialized.")

async def gaurd(message: str) -> tuple[bool, str | None]:
    if _rails is None:
        logfire.warning("🛡️ Guardrails not initialized yet.")
        return False, None

    with logfire.span("🛡️ Gaurdrails Check"):
        result = await _rails.generate_async(messages=[{"role": "user", "content": message}])
        print("RAW:", result)
        _rails.explain().print_llm_calls_summary()
        info = _rails.explain()
        for c in info.llm_calls:
            print("TASK:", c.task)
            print("COMPLETION:", repr(c.completion[:400]))
        print("MODEL:", getattr(_rails.llm, "model_name", _rails.llm))
        if isinstance(result, dict):
            content = result.get("content", "")
        elif hasattr(result, "content"):
            content = result.content
        else:
            content = str(result)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        # 2. Structural logic check: Did the planner map to a restriction flow?
        info = _rails.explain()
        # Look for the internal generation steps to see if next-step generation was subverted by a flow
        for call in info.llm_calls:
            if call.task == "generate_next_steps" and "refuse" in getattr(call, "completion", "").lower():
                fired = True
                # Override content with your standard clean fallback error message if needed
                content = "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!"
                break

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content      # caller returns immediately, skips RAG

        logfire.info("✅ Guardrails passed.")
        return False, None            # caller proceeds to LangGraph