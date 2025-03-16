"""CustomerDetails entity module."""

from pydantic import BaseModel

from .order import Order


class CustomerDetails(BaseModel):
    """CustomerDetails Class."""

    customer_id: str
    name: str
    email: str
    orders: list[Order] | None = None
