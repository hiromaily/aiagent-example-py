"""Agent with Tools Use Case."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

from utils.markdown import to_markdown

shipping_info_db: dict[str, str] = {
    "12345": "Shipped on 2024-12-01",
    "67890": "Out for delivery",
}


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


def get_shipping_info(ctx: RunContext[CustomerDetails]) -> str:
    """Get Shipping Info."""
    return shipping_info_db[ctx.deps.orders[0].order_id]


def agent_with_tools_use_case(model: OpenAIModel) -> None:
    """Agent with Tools Use Case."""
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
