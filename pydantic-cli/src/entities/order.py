"""Order entity."""

from pydantic import BaseModel


class Order(BaseModel):
    """Order Class."""

    order_id: str
    status: str
    items: list[str]
