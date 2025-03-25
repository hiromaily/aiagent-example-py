"""Create an LLM (Language Model) using OpenAI's API."""

from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.openai import OpenAI


def create_openai_llm(model: str | None, temperature: float = 0.5) -> OpenAI:
    """Create an LLM (Language Model) using OpenAI's API."""
    if not model:
        msg = "env 'OPENAI_MODEL' must be set"
        raise ValueError(msg)

    # Create an LLM (Language Model)
    return OpenAI(model=model, temperature=temperature)


def create_lmstudio_llm(model: str | None, temperature: float = 0.5) -> OpenAI:
    """Create an LLM (Language Model) using LMStudio's API."""
    if not model:
        msg = "env 'LMSTUDIO_MODEL' must be set"
        raise ValueError(msg)

    return LMStudio(
        model_name=model,
        base_url="http://localhost:1234/v1",
        temperature=temperature,
    )
