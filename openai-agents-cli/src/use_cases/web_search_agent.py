"""Web Search Agent Use Case."""

from infrastructure.web_browser.interface import WebClientInterface


class WebSearchAgent:
    """Web Search Agent Use Case."""

    def __init__(
        self,
        web_client: WebClientInterface,
    ) -> None:
        """Initialize the WebSearchAgent with an Web client."""
        self._web_client = web_client

    def query_news(self) -> str:
        """Query about news using Web Search."""
        # Initial prompt
        prompt = "What was a positive news story from today?"

        return self._web_client.call_web_search(prompt)
