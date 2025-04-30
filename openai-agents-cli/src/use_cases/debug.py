"""Debug Use Case."""

from loguru import logger

from embedding.embedding import load_embedding
from infrastructure.repository.interface import DocumentsRepositoryInterface


class DebugAgent:
    """Debug Agent Use Case."""

    def __init__(
        self,
        docs_repo: DocumentsRepositoryInterface,
    ) -> None:
        """Initialize the Debug Agent with an docs repository."""
        self._docs_repo = docs_repo

    def embedding(self, file_path: str) -> None:
        """Embedding static file."""
        # load embedding JSON file
        embedding_list = load_embedding(file_path)

        # Insert into DB
        logger.debug("insert into db")
        self._docs_repo.insert_embeddings(embedding_list)
        self._docs_repo.close()
