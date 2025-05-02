"""Embedding entity class.

This type manipulates the `from openai.types.embedding import Embedding`.
OpenAI's embedding models returns this type.
"""

from dataclasses import dataclass

import numpy as np
from openai.types.embedding import Embedding as OpenAIEmbedding

EmbeddingType = list[float] | int | str


class Embedding:
    """Embedding entity class."""

    def __init__(self, embedding: list[float], index: int, object_type: str) -> None:
        """Initialize the Embedding."""
        self.embedding = embedding
        self.index = index
        self.object = object_type

    def to_dict(self) -> dict[str, EmbeddingType]:
        """Convert to dictionary."""
        return {"embedding": self.embedding, "index": self.index, "object": self.object}

    @classmethod
    def from_dict(cls, dict_obj: dict[str, EmbeddingType]) -> "Embedding":
        """Create an Embedding object from a dictionary."""
        return cls(
            embedding=dict_obj["embedding"],  # type: ignore[arg-type]
            index=dict_obj.get("index", 0),  # type: ignore[arg-type]
            object_type=dict_obj.get("object", "embedding"),  # type: ignore[arg-type]
        )

    @classmethod
    def from_openai_embedding(cls, embeddings: list[OpenAIEmbedding]) -> list["Embedding"]:
        """Convert OpenAI Embedding to custom Embedding."""
        return [
            cls(
                embedding=embedding.embedding,
                index=embedding.index,
                object_type=embedding.object,
            )
            for embedding in embeddings
        ]

    def __repr__(self) -> str:
        """Representation of the Embedding object."""
        return f"Embedding(index={self.index}, embedding={self.embedding}, object='{self.object}')"


@dataclass
class EmbeddingItem:
    """Embedding item class for response of VectorDB `pgvector`."""

    question: str
    embedding: np.typing.NDArray[np.float64]
