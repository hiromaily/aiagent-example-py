"""Custom Agent Use Case."""

from enum import Enum

from openai import Embedding

from openai_custom.interface import OpenAIClientInterface


class APIMode(Enum):
    """API Mode."""

    RESPONSE_API = 1
    CHAT_COMPLETION_API = 2


class CustomTechnicalAgent:
    """Custom Technical Agent Use Case.

    This agent is designed to answer technical questions and provide step-by-step guidance
    on how to learn a specific technology.
    """

    def __init__(self, openai_client: OpenAIClientInterface) -> None:
        """Initialize the CustomTechnicalAgent with an OpenAI client."""
        self.openai_client = openai_client

    def query_tech_guide(self, user_query: str, mode: APIMode) -> str:
        """Query the agent with a user tech question."""
        # Initial prompt
        instructions = "You are an experienced software engineer."
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {user_query}
        """

        if mode == APIMode.RESPONSE_API:
            return self.openai_client.call_response(instructions, prompt)
        if mode == APIMode.CHAT_COMPLETION_API:
            return self.openai_client.call_chat_completion(instructions, prompt)
        msg = "Unknown API mode"
        raise ValueError(msg)

    def query(self, user_query: str, mode: APIMode) -> str:
        """Query the agent with a user question."""
        # Initial prompt
        instructions = "You are a helpful assistant."

        if mode == APIMode.RESPONSE_API:
            return self.openai_client.call_response(instructions, user_query)
        if mode == APIMode.CHAT_COMPLETION_API:
            return self.openai_client.call_chat_completion(instructions, user_query)
        msg = "Unknown API mode"
        raise ValueError(msg)

    def embedding(self, user_query: str) -> list[Embedding]:
        """Call embedding API with a user question."""
        return self.openai_client.call_embeddings(user_query)

    def query_news(self) -> str:
        """Query about news."""
        # Initial prompt
        prompt = "What was a positive news story from today?"

        return self.openai_client.call_web_search(prompt)
