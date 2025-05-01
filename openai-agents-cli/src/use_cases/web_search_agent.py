"""Web Search Agent Use Case."""

from infrastructure.openai_api.interface import OpenAIClientInterface


class WebSearchAgent:
    """Web Search Agent Use Case."""

    def __init__(
        self,
        openai_client: OpenAIClientInterface,
    ) -> None:
        """Initialize the WebSearchAgent with an OpenAI client."""
        self._openai_client = openai_client

    def query_news(self) -> str:
        """Query about news using Web Search."""
        # Initial prompt
        prompt = "What was a positive news story from today?"

        return self._openai_client.call_web_search(prompt)
