"""OpenAI module class."""

import psycopg2


class PgVectorClient:
    """pgvector client class."""

    def __init__(self, host: str, port: int, db_name: str, user: str, password: str) -> None:
        """Initialize PgVectorClient."""
        # Note: if mypy is working properly, this check can be removed

        # if not host:
        #     msg = "`host` must be provided"
        #     raise ValueError(msg)

        # if not port:
        #     msg = "`port` must be provided"
        #     raise ValueError(msg)

        # if not db_name:
        #     msg = "`db_name` must be provided"
        #     raise ValueError(msg)

        # if not user:
        #     msg = "`user` must be provided"
        #     raise ValueError(msg)

        # if not password:
        #     msg = "`password` must be provided"
        #     raise ValueError(msg)

        self._conn = psycopg2.connect(host=host, port=port, database=db_name, user=user, password=password)

    def get_conn(self) -> psycopg2.extensions.connection:
        """Get conn."""
        return self._conn

    def get_cursor(self) -> psycopg2.extensions.cursor:
        """Get cursor."""
        return self._conn.cursor()

    def close(self) -> None:
        """Close connection."""
        return self._conn.close()
