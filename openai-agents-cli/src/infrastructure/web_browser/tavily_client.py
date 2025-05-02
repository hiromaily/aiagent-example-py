"""Tavily web client module class."""

from typing import cast

from tavily import TavilyClient

from .interface import WebClientInterface


class TavilyWebClient(WebClientInterface):
    """Tavily Web Client class."""

    def __init__(
        self,
        api_key: str,
    ) -> None:
        """Initialize Tavily client."""
        self._tavily_client = TavilyClient(api_key=api_key)

    def call_web_search(self, prompt: str) -> str:
        """Call Web Search API."""
        response = self._tavily_client.search(prompt)
        content = response["results"][0]["content"]
        return cast("str", content)
