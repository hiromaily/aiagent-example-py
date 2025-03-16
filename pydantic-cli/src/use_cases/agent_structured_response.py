"""Agent with Structured Response Use Case."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


class ResponseModel(BaseModel):
    """Structured response with metadata."""

    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")


def agent_structured_response_use_case(model: OpenAIModel) -> None:
    """Agent with Structured Response Use Case."""
    agent1 = Agent(
        model=model,
        result_type=ResponseModel,
        system_prompt=(
            "You are an intelligent customer support agent. Analyze queries carefully and provide structured responses."
        ),
    )

    response = agent1.run_sync("How can I track my order #12345?")
    print("Response Data (Structured):\n", response.data.model_dump_json(indent=2), "\n")
