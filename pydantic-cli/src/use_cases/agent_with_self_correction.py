"""Agent with Reflection and Self-Correction Use Case."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel

shipping_info_db: dict[str, str] = {
    "#12345": "Shipped on 2024-12-01",
    "#67890": "Out for delivery",
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


def agent_with_self_correction_use_case(model: OpenAIModel) -> None:
    """Agent with Reflection and Self-Correction Use Case."""
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
