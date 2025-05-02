"""Interface module for WebBrowser."""

from abc import ABC, abstractmethod


class WebClientInterface(ABC):
    """Interface for WebClient."""

    @abstractmethod
    def call_web_search(self, prompt: str) -> str:
        """Call Web Search API with given prompt."""
