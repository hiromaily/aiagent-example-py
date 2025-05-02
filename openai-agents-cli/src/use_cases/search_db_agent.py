"""Search VectorDB Use Case."""

from loguru import logger

from infrastructure.repository.interface import EmbeddingRepositoryInterface


class SearchVectorDBAgent:
    """Search Vector DB Agent Use Case."""

    def __init__(
        self,
        embedding_repo: EmbeddingRepositoryInterface,
    ) -> None:
        """Initialize the Search Vector DB Agent with an embedding repository."""
        self._embedding_repo = embedding_repo

    def search_similarity(self, content_id: int) -> None:
        """Embedding static file."""
        # Search target item_content from DB `item_contents`
        embedding_item = self._embedding_repo.get_item_by_id(content_id)
        if embedding_item is None:
            return

        # Search similarity
        logger.debug("search similarity")
        similarities = self._embedding_repo.similarity_search(embedding_item.embedding, 3)
        print(similarities)
