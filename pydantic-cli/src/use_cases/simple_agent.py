"""Simple Agent Use Case."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


def simple_agent_use_case(model: OpenAIModel) -> None:
    """Simple Agent Use Case."""
    agent1 = Agent(
        model=model,
        system_prompt=("You are a helpful customer support agent. Be concise and friendly."),
    )

    response = agent1.run_sync("How can I track my order #12345?")
    print("Response Data 1:\n ", response.data, "\n")
    print("All Messages:", response.all_messages(), "\n")

    response2 = agent1.run_sync(
        user_prompt="What was my previous question?",
        message_history=response.new_messages(),
    )
    print("Response Data 2:\n ", response2.data, "\n")
