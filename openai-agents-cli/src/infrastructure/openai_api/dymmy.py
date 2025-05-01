"""OpenAI dummy module class."""

from openai.types.embedding import Embedding

from .interface import OpenAIClientInterface


class OpenAIDummyClient(OpenAIClientInterface):
    """OpenAI API Dummy Client class."""

    def __init__(self) -> None:
        """Initialize OpenAI client."""

    def call_response(self, _instructions: str, _prompt: str) -> str:
        """Call Response API."""
        return "dummy response"

    def call_chat_completion(self, _instructions: str, _prompt: str) -> str:
        """Call Chat Completion API with instructions and prompt."""
        return "dummy response"

    def call_embeddings(self, _prompt: str | list[str]) -> list[Embedding]:
        """Call Embedding API with prompt."""
        return [Embedding(embedding=[0.1, 0.2, 0.3, 0.4, 0.5], index=0, object="embedding")]

    def call_web_search(self, _prompt: str) -> str:
        """Call Web Search API."""
        return "dummy web search response"
