"""
Prompt template loader. Templates are stored as .txt files
in prompts/en/ and prompts/ar/ so non-coders can iterate on them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.core.config import PROMPT_DIR
from src.core.state import Language


@lru_cache(maxsize=128)
def load_prompt(name: str, language: Language = "en") -> str:
    """Load prompts/<lang>/<name>.txt. Falls back to English if missing."""
    fpath = PROMPT_DIR / language / f"{name}.txt"
    if not fpath.exists():
        fpath = PROMPT_DIR / "en" / f"{name}.txt"
    if not fpath.exists():
        raise FileNotFoundError(f"Prompt template {name!r} not found.")
    return fpath.read_text(encoding="utf-8")


def render(template: str, **kwargs) -> str:
    """Simple `{var}` substitution. No surprises."""
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise KeyError(f"Missing template variable: {exc}") from exc
