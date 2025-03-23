"""Interface module for OpenAIClient."""

from abc import ABC, abstractmethod


class DocumentsRepositoryInterface(ABC):
    """Interface for DocumentsRepository."""

    @abstractmethod
    def insert_embeddings(self, data: str) -> None:
        """Execute insert."""
