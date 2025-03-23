"""Interface module for OpenAIClient."""

from abc import ABC, abstractmethod

from entities.embedding import Embedding


class DocumentsRepositoryInterface(ABC):
    """Interface for DocumentsRepository."""

    @abstractmethod
    def insert_embeddings(self, data: list[Embedding]) -> None:
        """Execute insert."""

    @abstractmethod
    def similarity_search(self, embedding: Embedding, top_k: int) -> list[str]:
        """Execute similarity search."""

    @abstractmethod
    def close(self) -> None:
        """Close connection."""
