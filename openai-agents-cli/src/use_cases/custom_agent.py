"""Custom Agent Use Case."""

from openai_custom.client import OpenAIClient


class CustomTechnicalAgent:
    """Custom Technical Agent Use Case.

    This agent is designed to answer technical questions and provide step-by-step guidance
    on how to learn a specific technology.
    """

    def __init__(self, openai_client: OpenAIClient) -> None:
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

        # Step 1: Initial thought generation using OpenAI API
        return self.openai_client.call_openai_api(instructions, prompt)
