"""Embedding VectorDB repository class."""

import numpy as np
from loguru import logger
from pgvector.psycopg2 import register_vector

from entities.embedding.types import Embedding, EmbeddingItem
from infrastructure.repository.interface import EmbeddingRepositoryInterface
from infrastructure.vectordb.pgvector.client import PgVectorClient


class PgVectorEmbeddingRepository(EmbeddingRepositoryInterface):
    """Documents VectorDB repository class."""

    def __init__(self, pg_vector_client: PgVectorClient, is_large_embedding: bool) -> None:
        """Initialize Documents VectorDB repository class."""
        if not pg_vector_client:
            msg = "`pg_vector_client` must be provided"
            raise ValueError(msg)

        self._pg_vector_client = pg_vector_client
        self._embeddings_table = "embeddings_large" if is_large_embedding else "embeddings"
        self._item_contents_table = "item_contents_large" if is_large_embedding else "item_contents"

    def insert_embeddings(self, data: list[Embedding]) -> None:
        """Insert embeddings data into `embeddings` table."""
        logger.debug("DocumentsRepository.insert_embeddings()")

        register_vector(self._pg_vector_client.get_conn())
        cur = self._pg_vector_client.get_cursor()

        query = f"INSERT INTO {self._embeddings_table} (embedding) VALUES (%s)"  # noqa: S608
        for embedding in data:
            # logger.debug(f"embedding.embedding: {embedding.embedding}")
            # parameters must be tuple
            cur.execute(query, (embedding.embedding,))
            self._pg_vector_client.get_conn().commit()

        cur.close()

    def insert_item_contents(self, contents: list[str], embeddings: list[Embedding]) -> None:
        """Insert content, embedding into `item_contents` table."""
        logger.debug("DocumentsRepository.insert_item_contents()")

        register_vector(self._pg_vector_client.get_conn())
        cur = self._pg_vector_client.get_cursor()

        query = f"INSERT INTO {self._item_contents_table} (content, embedding) VALUES (%s, %s)"  # noqa: S608
        for content, embedding in zip(contents, embeddings, strict=False):
            # parameters must be tuple
            cur.execute(query, (content, embedding.embedding))
            self._pg_vector_client.get_conn().commit()

        cur.close()

    def get_item_by_id(self, item_id: int) -> EmbeddingItem | None:
        """Get a record by id from `item_contents` table."""
        logger.debug("DocumentsRepository.get_item_by_id()")

        cur = self._pg_vector_client.get_cursor()
        query = f"SELECT content, embedding FROM {self._item_contents_table} WHERE id = %s"  # noqa: S608
        # parameters must be tuple
        cur.execute(query, (item_id,))
        item = cur.fetchone()
        cur.close()
        if item is None:
            return None

        # return item
        # return cast("tuple[str, np.typing.NDArray[np.float64]]", item)
        content, embedding = item
        return EmbeddingItem(question=content, embedding=embedding)

    def similarity_search(self, embedding: np.typing.NDArray[np.float64], top_k: int = 5) -> list[str] | None:
        """Execute similarity search."""
        cur = self._pg_vector_client.get_cursor()
        query = f"SELECT content FROM {self._item_contents_table} ORDER BY embedding <=> %s LIMIT %s"  # noqa: S608
        # parameters must be tuple
        cur.execute(query, (embedding, top_k))
        items = cur.fetchall()
        cur.close()
        if items is None:
            return None
        # convert list[tuple[str]] to list[str]
        # return cast("list[tuple[str]]", items)
        return [str(item[0]) for item in items]

    def close(self) -> None:
        """Close connection."""
        self._pg_vector_client.close()
