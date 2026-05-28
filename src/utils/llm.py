"""
Single LLM client for the whole project.

Every LLM call goes through here so we can:
- Swap providers without touching agent code
- Apply temperature / token defaults consistently
- Mock in tests
"""

from __future__ import annotations

from src.core.config import config


def get_llm(temperature: float = 0.2, max_tokens: int = 1024):
    """Return a LangChain chat model based on what's configured."""
    if config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=config.OPENAI_API_KEY,
        )
    if config.GROQ_API_KEY:
        # Groq is API-compatible with the OpenAI SDK
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    raise RuntimeError(
        "No LLM provider configured. Set OPENAI_API_KEY or GROQ_API_KEY in .env"
    )


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    """Tiny convenience wrapper for single-turn calls."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(temperature=temperature)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if hasattr(resp, "content") else str(resp)
