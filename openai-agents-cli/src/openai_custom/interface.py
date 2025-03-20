"""Interface module for OpenAIClient."""

from abc import ABC, abstractmethod


class OpenAIClientInterface(ABC):
    """Interface for OpenAIClient."""

    @abstractmethod
    def call_response(self, instructions: str, prompt: str) -> str:
        """Call Response API with instructions and prompt."""

    @abstractmethod
    def call_web_search(self, prompt: str) -> str:
        """Call Web Search API with given prompt."""
