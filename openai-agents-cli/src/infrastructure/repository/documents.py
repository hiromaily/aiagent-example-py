"""Documents VectorDB repository class."""

from pgvector.psycopg2 import register_vector

from infrastructure.pgvector.conn import PgVectorClient
from infrastructure.repository.interface import DocumentsRepositoryInterface


class DocumentsRepository(DocumentsRepositoryInterface):
    """Documents VectorDB repository class."""

    def __init__(self, pg_vector_client: PgVectorClient) -> None:
        """Initialize Documents VectorDB repository class."""
        if not pg_vector_client:
            msg = "`pg_vector_client` must be provided"
            raise ValueError(msg)

        self._pg_vector_client = pg_vector_client

    def insert_embeddings(self, data: str) -> None:
        """Insert data into Vector DB."""
        register_vector(self._pg_vector_client.get_conn())
        cur = self._pg_vector_client.get_cursor()
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES %s", [(d["content"], d["embedding"]) for d in data]
        )
        cur.close()
