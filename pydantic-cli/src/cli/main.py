"""main function for the CLI app."""

import os
import sys

# from typing import Dict, List, Optional
import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

from utils.markdown import to_markdown

# load .env file
load_dotenv()
openai_model = os.getenv("OPENAI_MODEL")
if not openai_model:
    print("Error: OPENAI_MODEL is not set or is empty")
    sys.exit(1)

# Iniiate OpenAI Model
model = OpenAIModel(openai_model)

# Create a Typer app
app = typer.Typer()


# -----------------------------------------------------------------------------
# Simple Agent Command
# ````
# python -m src.cli.main simple-agent
# ```
# -----------------------------------------------------------------------------
@app.command()
def simple_agent() -> None:
    """This example demonstrates the basic usage of PydanticAI agents.

    Key concepts:
    - Creating a basic agent with a system prompt
    - Running synchronous queries
    - Accessing response data, message history, and costs
    """
    agent1 = Agent(
        model=model,
        system_prompt=("You are a helpful customer support agent. Be concise and friendly."),
    )

    response = agent1.run_sync("How can I track my order #12345?")
    print("Response Data 1:\n ", response.data, "\n")
    print("All Messages:", response.all_messages(), "\n")

    # print("Cost:", response.cost())
    # debug
    # print("Debug Response properties:\n ", response.__dict__, "\n")
    # - data
    # - _result_tool_name
    # - _state
    #   - message_history
    #   - usage
    #   - retries
    #   - run_step
    # - _new_message_index

    response2 = agent1.run_sync(
        user_prompt="What was my previous question?",
        message_history=response.new_messages(),
    )
    print("Response Data 2:\n ", response2.data, "\n")


# -----------------------------------------------------------------------------
# Agent with Structured Response
# ```
# python -m src.cli.main agent-structured-response
# ```
# -----------------------------------------------------------------------------
@app.command()
def agent_structured_response() -> None:
    """Agent with Structured Response.

    This example shows how to get structured, type-safe responses from the agent.
    Key concepts:
    - Using Pydantic models to define response structure
    - Type validation and safety
    - Field descriptions for better model understanding
    """

    class ResponseModel(BaseModel):
        """Structured response with metadata."""

        response: str
        needs_escalation: bool
        follow_up_required: bool
        sentiment: str = Field(description="Customer sentiment analysis")

    agent1 = Agent(
        model=model,
        result_type=ResponseModel,
        system_prompt=(
            "You are an intelligent customer support agent. Analyze queries carefully and provide structured responses."
        ),
    )

    response = agent1.run_sync("How can I track my order #12345?")
    print("Response Data (Structured):\n", response.data.model_dump_json(indent=2), "\n")


# -----------------------------------------------------------------------------
# Agent with Structured Response & Dependencies
# ```
# python -m src.cli.main agent-structured-dependencies
# ```
# -----------------------------------------------------------------------------
@app.command()
def agent_structured_dependencies() -> None:
    """This example demonstrates how to use dependencies and context in agents.

    Key concepts:
    - Defining complex data models with Pydantic
    - Injecting runtime dependencies
    - Using dynamic system prompts
    """

    class ResponseModel(BaseModel):
        response: str
        needs_escalation: bool
        follow_up_required: bool
        sentiment: str = Field(description="Customer sentiment analysis")

    class Order(BaseModel):
        order_id: str
        status: str
        items: list[str]

    class CustomerDetails(BaseModel):
        customer_id: str
        name: str
        email: str
        orders: list[Order] | None = None

    agent1 = Agent(
        model=model,
        result_type=ResponseModel,
        deps_type=CustomerDetails,
        retries=3,
        system_prompt=(
            "You are an intelligent customer support agent. "
            "Analyze queries carefully and provide structured responses. "
            "Always greet the customer and provide a helpful response."
        ),
    )

    @agent1.system_prompt
    async def add_customer_name(ctx: RunContext[CustomerDetails]) -> str:
        return f"Customer details: {to_markdown(ctx.deps)}"

    customer = CustomerDetails(
        customer_id="1",
        name="John Doe",
        email="john.doe@example.com",
        orders=[
            Order(order_id="12345", status="shipped", items=["Blue Jeans", "T-Shirt"]),
        ],
    )

    response = agent1.run_sync(user_prompt="What did I order?", deps=customer)
    print("All Messages:\n", response.all_messages(), "\n")
    print("Response Data (Structured):\n", response.data.model_dump_json(indent=2), "\n")

    print(
        f"Customer Details:\n"
        f"Name: {customer.name}\n"
        f"Email: {customer.email}\n\n"
        "Response Details:\n"
        f"{response.data.response}\n\n"
        "Status:\n"
        f"Follow-up Required: {response.data.follow_up_required}\n"
        f"Needs Escalation: {response.data.needs_escalation}"
        f"\n"
    )


# -----------------------------------------------------------------------------
# Agent with Tools
# ```
# python -m src.cli.main agent-with-tools
# ```
# -----------------------------------------------------------------------------
@app.command()
def agent_with_tools() -> None:
    """Agent with Tools.

    This example shows how to enhance agents with custom tools.

    Key concepts:
    - Creating and registering tools
    - Accessing context in tools
    """
    shipping_info_db: dict[str, str] = {
        "12345": "Shipped on 2024-12-01",
        "67890": "Out for delivery",
    }

    class ResponseModel(BaseModel):
        response: str
        needs_escalation: bool
        follow_up_required: bool
        sentiment: str = Field(description="Customer sentiment analysis")

    class Order(BaseModel):
        order_id: str
        status: str
        items: list[str]

    class CustomerDetails(BaseModel):
        customer_id: str
        name: str
        email: str
        orders: list[Order] | None = None

    def get_shipping_info(ctx: RunContext[CustomerDetails]) -> str:
        return shipping_info_db[ctx.deps.orders[0].order_id]

    agent1 = Agent(
        model=model,
        result_type=ResponseModel,
        deps_type=CustomerDetails,
        retries=3,
        system_prompt=(
            "You are an intelligent customer support agent. "
            "Analyze queries carefully and provide structured responses. "
            "Use tools to look up relevant information."
            "Always greet the customer and provide a helpful response."
        ),
        tools=[Tool(get_shipping_info, takes_ctx=True)],
    )

    @agent1.system_prompt
    async def add_customer_name(ctx: RunContext[CustomerDetails]) -> str:
        return f"Customer details: {to_markdown(ctx.deps)}"

    customer = CustomerDetails(
        customer_id="1",
        name="John Doe",
        email="john.doe@example.com",
        orders=[Order(order_id="12345", status="shipped", items=["Blue Jeans", "T-Shirt"])],
    )

    response = agent1.run_sync(user_prompt="What's the status of my last order?", deps=customer)
    print("All Messages:\n", response.all_messages(), "\n")
    print("Response Data (Structured):\n", response.data.model_dump_json(indent=2), "\n")

    print(
        "Customer Details:\n"
        f"Name: {customer.name}\n"
        f"Email: {customer.email}\n\n"
        "Response Details:\n"
        f"{response.data.response}\n\n"
        "Status:\n"
        f"Follow-up Required: {response.data.follow_up_required}\n"
        f"Needs Escalation: {response.data.needs_escalation}"
        "\n"
    )


# -----------------------------------------------------------------------------
# Agent with Reflection and Self-Correction
# ```
# python -m src.cli.main agent-with-self-correction
# ```
# -----------------------------------------------------------------------------
@app.command()
def agent_with_self_correction() -> None:
    """Agent with Reflection and Self-Correction.

    This example demonstrates advanced agent capabilities with self-correction.

    Key concepts:
    - Implementing self-reflection
    - Handling errors gracefully with retries
    - Using ModelRetry for automatic retries
    - Decorator-based tool registration
    """
    shipping_info_db: dict[str, str] = {
        "#12345": "Shipped on 2024-12-01",
        "#67890": "Out for delivery",
    }

    class ResponseModel(BaseModel):
        response: str
        needs_escalation: bool
        follow_up_required: bool
        sentiment: str = Field(description="Customer sentiment analysis")

    class Order(BaseModel):
        order_id: str
        status: str
        items: list[str]

    class CustomerDetails(BaseModel):
        customer_id: str
        name: str
        email: str
        orders: list[Order] | None = None

    customer = CustomerDetails(
        customer_id="1",
        name="John Doe",
        email="john.doe@example.com",
    )

    agent1 = Agent(
        model=model,
        result_type=ResponseModel,
        deps_type=CustomerDetails,
        retries=3,
        system_prompt=(
            "You are an intelligent customer support agent. "
            "Analyze queries carefully and provide structured responses. "
            "Use tools to look up relevant information. "
            "Always greet the customer and provide a helpful response."
        ),
    )

    @agent1.tool_plain()
    def get_shipping_status(order_id: str) -> str:
        shipping_status = shipping_info_db.get(order_id)
        if shipping_status is None:
            message = (
                f"No shipping information found for order ID {order_id}. "
                "Make sure the order ID starts with a #: e.g, #624743 "
                "Self-correct this if needed and try again."
            )
            raise ModelRetry(message)
        return shipping_info_db[order_id]

    response = agent1.run_sync(user_prompt="What's the status of my last order 12345?", deps=customer)
    print("All Messages:\n", response.all_messages(), "\n")
    print("Response Data (Structured):\n", response.data.model_dump_json(indent=2), "\n")


if __name__ == "__main__":
    app()
