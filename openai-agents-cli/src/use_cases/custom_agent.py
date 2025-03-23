"""Custom Agent Use Case."""

from openai import Embedding

from openai_custom.interface import OpenAIClientInterface


class CustomTechnicalAgent:
    """Custom Technical Agent Use Case.

    This agent is designed to answer technical questions and provide step-by-step guidance
    on how to learn a specific technology.
    """

    def __init__(self, openai_client: OpenAIClientInterface) -> None:
        """Initialize the CustomTechnicalAgent with an OpenAI client."""
        self.openai_client = openai_client

    def query(self, user_query: str) -> str:
        """Query the agent with a user question."""
        # Initial prompt
        instructions = "You are an experienced software engineer."
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {user_query}
        """

        return self.openai_client.call_response(instructions, prompt)

    def query_with_chat(self, user_query: str) -> str:
        """Query the agent with a user question."""
        # Initial prompt
        instructions = "You are an experienced software engineer."
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {user_query}
        """

        return self.openai_client.call_chat_completion(instructions, prompt)

    def embedding(self, user_query: str) -> list[Embedding]:
        """Call embedding API with a user question."""
        return self.openai_client.call_embeddings(user_query)

    def query_news(self) -> str:
        """Query about news."""
        # Initial prompt
        prompt = "What was a positive news story from today?"

        return self.openai_client.call_web_search(prompt)
