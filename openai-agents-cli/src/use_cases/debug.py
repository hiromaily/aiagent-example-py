"""Debug Use Case."""

from loguru import logger

from entities.embedding.utils import load_embedding
from infrastructure.repository.interface import EmbeddingRepositoryInterface


class DebugAgent:
    """Debug Agent Use Case."""

    def __init__(
        self,
        embedding_repo: EmbeddingRepositoryInterface,
    ) -> None:
        """Initialize the Debug Agent with an embedding repository."""
        self._embedding_repo = embedding_repo

    def embedding(self, file_path: str) -> None:
        """Embedding static file."""
        # load embedding JSON file
        embedding_list = load_embedding(file_path)

        # Insert into DB
        logger.debug("insert into db")
        self._embedding_repo.insert_embeddings(embedding_list)
        self._embedding_repo.close()
