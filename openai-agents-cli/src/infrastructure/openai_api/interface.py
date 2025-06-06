"""Interface module for OpenAIClient."""

from abc import ABC, abstractmethod

# from openai.types.embedding import Embedding
from entities.embedding.types import Embedding


class OpenAIClientInterface(ABC):
    """Interface for OpenAIClient."""

    @abstractmethod
    def call_response(self, instructions: str, prompt: str) -> str:
        """Call Response API with instructions and prompt."""

    @abstractmethod
    def call_chat_completion(self, instructions: str, prompt: str) -> str:
        """Call Chat Completion API with instructions and prompt."""

    @abstractmethod
    def call_embeddings(self, prompt: str | list[str]) -> list[Embedding]:
        """Call Embedding API with prompt."""

    @abstractmethod
    def call_web_search(self, prompt: str) -> str:
        """Call Web Search API with given prompt."""
