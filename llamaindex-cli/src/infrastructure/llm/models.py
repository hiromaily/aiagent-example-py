"""Create an LLM (Language Model) using OpenAI's API."""

from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.openai import OpenAI


def create_openai_llm(model: str, api_key: str, temperature: float = 0.5) -> OpenAI:
    """Create an LLM (Language Model) using OpenAI's API."""
    # Create an LLM (Language Model)
    return OpenAI(model=model, api_key=api_key, temperature=temperature)


def create_lmstudio_llm(model: str, temperature: float = 0.5) -> LMStudio:
    """Create an LLM (Language Model) using LMStudio's API."""
    # FIXME: somehow `https://api.openai.com/v1/chat/completions` is called
    if not model:
        msg = "env 'LMSTUDIO_MODEL' must be set"
        raise ValueError(msg)

    return LMStudio(
        model_name=model,
        base_url="http://localhost:1234/v1",
        temperature=temperature,
    )


def create_lmstudio_embedding_llm(model: str = "text-embedding-ada-002") -> OpenAIEmbedding:
    """Create an embedding LLM (Language Model) using LMStudio's API."""
    # Note: ValueError: 'text-embedding' is not a valid OpenAIEmbeddingModelType
    # model must be listed in OpenAIEmbeddingModeModel
    if not model:
        msg = "env 'LMSTUDIO_MODEL' must be set"
        raise ValueError(msg)

    return OpenAIEmbedding(
        model=model,
        api_key="lm-studio",
        api_base="http://localhost:1234/v1",
    )


def set_global_default_llm(llm: LLM, embed_model: OpenAIEmbedding) -> None:
    """Set global default LLM and embedding model."""
    Settings.llm = llm
    Settings.embed_model = embed_model
