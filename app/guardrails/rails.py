
import os
import yaml
import logfire

from nemoguardrails import LLMRails, RailsConfig

from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    RAIL_INDICATORS,
    YAML_CONTENT,
)

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """Initialize NeMo Guardrails with Groq."""

    global _rails

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured")

    model = os.getenv(
        "GUARDRAILS_MODEL",
        "openai/gpt-oss-20b",
    )

    # Parse the existing NeMo YAML configuration.
    config_data = yaml.safe_load(YAML_CONTENT) or {}

    # Preserve existing non-main model configurations.
    existing_models = config_data.get("models", [])

    config_data["models"] = [
        item for item in existing_models
        if item.get("type") != "main"
    ]

    # Configure Groq as the main NeMo model.
    config_data["models"].append({
        "type": "main",
        "engine": "openai",
        "model": model,
        "api_key_env_var": "GROQ_API_KEY",
        "parameters": {
            "base_url": "https://api.groq.com/openai/v1",
            "temperature": 0.0,
        },
    })
    config_data.setdefault("rails", {}).setdefault("dialog", {})["user_messages"] = {
        "embeddings_only": True,
        "embeddings_only_similarity_threshold": 0.5,
    }
    # IMPORTANT:
    # from_content validates the YAML and creates
    # proper NeMo Model objects.
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=yaml.safe_dump(
            config_data,
            sort_keys=False,
        ),
    )

    _rails = LLMRails(config)

    logfire.info(
        f"NeMo Guardrails initialized with Groq ({model})"
    )


def guard(message: str) -> tuple[bool, str | None]:
    """Validate user input through NeMo Guardrails."""

    if _rails is None:
        raise RuntimeError("Guardrails not initialized")

    with logfire.span("Guardrails Check"):
        try:
            result = _rails.generate(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            )
            info = _rails.explain()
            info.print_llm_calls_summary()
            print(info.colang_history)
            logfire.info(
                "NeMo raw response",
                response=str(result),
            )
            content = (
                result.get("content", "")
                if isinstance(result, dict)
                else str(result)
            )
            normalized_content = content.casefold()

            fired = any(
                indicator.casefold() in normalized_content
                for indicator in RAIL_INDICATORS
            )

            if fired:
                logfire.info("Guardrails fired")
                return True, content

            logfire.info("Guardrails passed")
            return False, None

        except Exception as exc:
            logfire.exception(
                "Guardrails execution failed",
                error=str(exc),
            )
            raise RuntimeError(
                "Guardrails validation failed"
            ) from exc
