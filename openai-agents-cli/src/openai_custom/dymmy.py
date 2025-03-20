"""OpenAI dummy module class."""

from openai_custom.interface import OpenAIClientInterface


class OpenAIDummyClient(OpenAIClientInterface):
    """OpenAI API Dummy Client class."""

    def __init__(self) -> None:
        """Initialize OpenAI client."""

    def call_response(self, _instructions: str, _prompt: str) -> str:
        """Call Response API."""
        return "dummy response"

    def call_web_search(self, _prompt: str) -> str:
        """Call Web Search API."""
        return "dummy web search response"
