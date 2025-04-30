"""OpenAI module class."""

import psycopg2


class PgVectorClient:
    """pgvector client class."""

    def __init__(self, host: str, port: int, db_name: str, user: str, password: str) -> None:
        """Initialize PgVectorClient."""
        self._conn = psycopg2.connect(host=host, port=port, database=db_name, user=user, password=password)

    def get_conn(self) -> psycopg2.extensions.connection:
        """Get conn."""
        return self._conn

    def get_cursor(self) -> psycopg2.extensions.cursor:
        """Get cursor."""
        return self._conn.cursor()

    def close(self) -> None:
        """Close connection."""
        self._conn.close()
