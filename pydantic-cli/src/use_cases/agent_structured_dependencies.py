"""Agent with Structured Dependencies Use Case."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from utils.markdown import to_markdown


class ResponseModel(BaseModel):
    """Response Model."""

    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")


class Order(BaseModel):
    """Order Model."""

    order_id: str
    status: str
    items: list[str]


class CustomerDetails(BaseModel):
    """Customer Details Model."""

    customer_id: str
    name: str
    email: str
    orders: list[Order] | None = None


def agent_structured_dependencies_use_case(model: OpenAIModel) -> None:
    """Agent with Structured Dependencies Use Case."""
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
        return f"Customer details: {to_markdown(ctx.deps)}"  # type: ignore[arg-type]

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
