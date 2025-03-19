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
        instructions = "あなたは経験豊富なソフトウェアエンジニアです。"
        prompt = f"""
        ユーザーの知りたい技術に答えるために、以下の情報を提供してください。
        1. その技術の概要
        2. その技術を習得するために、Step by stepのガイダンス
        ユーザーの知りたい技術: {user_query}
        """

        # Step 1: Initial thought generation using OpenAI API
        return self.openai_client.call_openai_api(instructions, prompt)
