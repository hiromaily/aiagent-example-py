"""Interface module for OpenAIClient."""

from abc import ABC, abstractmethod

import numpy as np

from entities.embedding import Embedding


class EmbeddingRepositoryInterface(ABC):
    """Interface for DocumentsRepository."""

    @abstractmethod
    def insert_embeddings(self, data: list[Embedding]) -> None:
        """Execute insert."""

    @abstractmethod
    def insert_item_contents(self, contents: list[str], embeddings: list[Embedding]) -> None:
        """Execute insert."""

    @abstractmethod
    def get_item_by_id(self, item_id: int) -> tuple[int, str, np.ndarray] | None:
        """Get content and embedding by id."""

    @abstractmethod
    def similarity_search(self, embedding: Embedding, top_k: int) -> list[tuple[str]] | None:
        """Execute similarity search."""

    @abstractmethod
    def close(self) -> None:
        """Close connection."""
