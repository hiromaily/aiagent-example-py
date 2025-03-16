"""Entity example."""


class Entity:
    """Entity class."""

    def __init__(self, id: int, name: str) -> None:
        """Entity constructor."""
        self.id: int = id
        self.name: str = name

    def __repr__(self) -> str:
        """Entity representation."""
        return f"Entity(id={self.id}, name='{self.name}')"

    def to_dict(self) -> dict:
        """Entity to dictionary."""
        return {"id": self.id, "name": self.name}
