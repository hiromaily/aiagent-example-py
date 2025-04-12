"""Documents VectorDB repository class."""

from typing import cast

import numpy as np
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
        """Insert data into Vector DB `embeddings`."""
        logger.debug("DocumentsRepository.insert_embeddings()")

        register_vector(self._pg_vector_client.get_conn())
        cur = self._pg_vector_client.get_cursor()

        query = "INSERT INTO embeddings (embedding) VALUES (%s)"
        for embedding in data:
            # logger.debug(f"embedding.embedding: {embedding.embedding}")
            # parameters must be tuple
            cur.execute(query, (embedding.embedding,))
            self._pg_vector_client.get_conn().commit()

        cur.close()

    def insert_item_contents(self, contents: list[str], embeddings: list[Embedding]) -> None:
        """Insert data into Vector DB `item_contents`."""
        logger.debug("DocumentsRepository.insert_item_contents()")

        register_vector(self._pg_vector_client.get_conn())
        cur = self._pg_vector_client.get_cursor()

        query = "INSERT INTO item_contents (content, embedding) VALUES (%s, %s)"
        for content, embedding in zip(contents, embeddings, strict=False):
            # parameters must be tuple
            cur.execute(query, (content, embedding.embedding))
            self._pg_vector_client.get_conn().commit()

        cur.close()

    def get_item_by_id(self, item_id: int) -> tuple[int, str, np.ndarray] | None:
        """Get a record by id from item_contents."""
        logger.debug("DocumentsRepository.get_item_by_id()")

        cur = self._pg_vector_client.get_cursor()
        query = "SELECT content, embedding FROM item_contents WHERE id = %s"
        # parameters must be tuple
        cur.execute(query, (item_id,))
        item = cur.fetchone()
        cur.close()
        if item is None:
            return None
        # return item
        return cast("tuple[int, str, np.ndarray]", item)

    def similarity_search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[str]] | None:
        """Execute similarity search."""
        cur = self._pg_vector_client.get_cursor()
        query = "SELECT content FROM item_contents ORDER BY embedding <=> %s LIMIT %s"
        # parameters must be tuple
        cur.execute(query, (embedding, top_k))
        items = cur.fetchall()
        cur.close()
        if items is None:
            return None
        # return item
        return cast("list[tuple[str]]", items)

    def close(self) -> None:
        """Close connection."""
        self._pg_vector_client.close()
