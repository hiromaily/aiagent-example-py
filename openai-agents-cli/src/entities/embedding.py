"""Embedding entity class."""


class Embedding:
    """Embedding entity class."""

    def __init__(self, embedding: list[float], index: int, object_type: str) -> None:
        """Initialize the Embedding."""
        self.embedding = embedding
        self.index = index
        self.object = object_type

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"embedding": self.embedding, "index": self.index, "object": self.object}

    @classmethod
    def from_dict(cls, dict_obj: dict) -> "Embedding":
        """Create an Embedding object from a dictionary."""
        return cls(
            embedding=dict_obj["embedding"],
            index=dict_obj.get("index", 0),
            object_type=dict_obj.get("object", "embedding"),
        )
