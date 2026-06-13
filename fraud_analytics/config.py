from __future__ import annotations
import json
import re
import os
from typing import Type, TypeVar
from pydantic import BaseModel
from dotenv import load_dotenv

T = TypeVar("T", bound=BaseModel)

load_dotenv()

# LLM_PROVIDER: "anthropic" | "openai" | "greennode"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "greennode")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# GreenNode AI Platform (OpenAI-compatible endpoint)
AI_PLATFORM_API_KEY: str = os.getenv("AI_PLATFORM_API_KEY", "")
AI_PLATFORM_BASE_URL: str = os.getenv(
    "AI_PLATFORM_BASE_URL",
    "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1",
)
AI_PLATFORM_MODEL: str = os.getenv("AI_PLATFORM_MODEL", "gpt-4o")

ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
MAX_RETRIEVAL_DOCS: int = int(os.getenv("MAX_RETRIEVAL_DOCS", "5"))
MAX_VALIDATION_RETRIES: int = int(os.getenv("MAX_VALIDATION_RETRIES", "2"))


DEFAULT_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))


def get_llm(temperature: float = 0.1, max_tokens: int | None = None):
    from langchain_openai import ChatOpenAI

    tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

    if LLM_PROVIDER == "greennode":
        return ChatOpenAI(
            model=AI_PLATFORM_MODEL,
            api_key=AI_PLATFORM_API_KEY,
            base_url=AI_PLATFORM_BASE_URL,
            temperature=temperature,
            max_tokens=tokens,
        )

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=tokens,
        )

    # openai (standard)
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=tokens,
    )


def _extract_json(text: str) -> str:
    """Pull the first complete JSON object out of a string."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Incomplete JSON object in LLM response")


def structured_invoke(llm, messages: list, schema: Type[T]) -> T:
    """
    Invoke the LLM and return a validated Pydantic model.

    Tries three strategies in order:
      1. json_mode  — response_format={"type":"json_object"} (widely supported)
      2. function_calling — OpenAI tool-call based structured output
      3. manual     — plain text response, JSON extracted with regex + Pydantic parse
    """
    # Strategy 1: json_mode
    try:
        return llm.with_structured_output(schema, method="json_mode").invoke(messages)
    except Exception:
        pass

    # Strategy 2: function_calling
    try:
        return llm.with_structured_output(schema, method="function_calling").invoke(messages)
    except Exception:
        pass

    # Strategy 3: manual JSON extraction
    from langchain_core.messages import SystemMessage
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    instruction = (
        "\n\nIMPORTANT: Respond with a single valid JSON object ONLY. "
        "No markdown, no explanation, no extra text.\n"
        f"Required JSON schema:\n{schema_hint}"
    )
    patched = list(messages)
    if patched and hasattr(patched[0], "content"):
        first = patched[0]
        patched[0] = type(first)(content=first.content + instruction)
    response = llm.invoke(patched)
    json_str = _extract_json(response.content)
    return schema.model_validate(json.loads(json_str))
