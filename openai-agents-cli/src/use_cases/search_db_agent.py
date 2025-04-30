"""Search VectorDB Use Case."""

from loguru import logger

from infrastructure.repository.interface import DocumentsRepositoryInterface


class SearchVectorDBAgent:
    """Search Vector DB Agent Use Case."""

    def __init__(
        self,
        docs_repo: DocumentsRepositoryInterface,
    ) -> None:
        """Initialize the Search Vector DB Agent with an docs repository."""
        self._docs_repo = docs_repo

    def search_similarity(self, content_id: int) -> None:
        """Embedding static file."""
        # Search target item_content from DB `item_contents`
        result = self._docs_repo.get_item_by_id(content_id)
        if result is None:
            return

        # Search similarity
        logger.debug("search similarity")
        similarities = self._docs_repo.similarity_search(result[2], 3)
        print(similarities)
