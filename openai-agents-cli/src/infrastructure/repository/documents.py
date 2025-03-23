"""Documents VectorDB repository class."""

from loguru import logger
from pgvector.psycopg2 import register_vector

from entities.embedding import Embedding
from infrastructure.pgvector.client import PgVectorClient
from infrastructure.repository.interface import DocumentsRepositoryInterface


class DocumentsRepository(DocumentsRepositoryInterface):
    """Documents VectorDB repository class."""

    def __init__(self, pg_vector_client: PgVectorClient) -> None:
        """Initialize Documents VectorDB repository class."""
        if not pg_vector_client:
            msg = "`pg_vector_client` must be provided"
            raise ValueError(msg)

        self._pg_vector_client = pg_vector_client

    def insert_embeddings(self, data: list[Embedding]) -> None:
        """Insert data into Vector DB."""
        logger.debug("DocumentsRepository.insert_embeddings()")

        register_vector(self._pg_vector_client.get_conn())
        cur = self._pg_vector_client.get_cursor()

        query = "INSERT INTO embeddings (embedding) VALUES (%s)"
        for embedding in data:
            # logger.debug(f"embedding.embedding: {embedding.embedding}")
            cur.execute(query, (embedding.embedding,))
            self._pg_vector_client.get_conn().commit()

        cur.close()

    # WIP: Implement similarity search
    def similarity_search(self, embedding: Embedding, top_k: int = 5) -> list[str]:
        """Execute similarity search."""
        cur = self._pg_vector_client.get_cursor()
        query = "SELECT content FROM documents ORDER BY embedding <=> %s LIMIT %s"
        cur.execute(query, (embedding, top_k))
        return cur.fetchall()

    def close(self) -> None:
        """Close connection."""
        self._pg_vector_client.close()
