"""Registry Class."""

import os

from infrastructure.pgvector.client import PgVectorClient
from infrastructure.repository.documents import DocumentsRepository
from infrastructure.repository.interface import DocumentsRepositoryInterface
from openai_custom.client import OpenAIClient
from openai_custom.dymmy import OpenAIDummyClient
from openai_custom.interface import OpenAIClientInterface


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, environment: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._environment = environment
        self._openai_client = self._build_openai_client()
        self._pg_client = self._build_pg_client()
        self._documents_repository = self._build_documents_repository()

    def _build_openai_client(self) -> OpenAIClientInterface:
        """Build the OpenAI client based on the environment."""
        if self._environment == "prod":
            # use OpenAI API
            openai_model = os.getenv("OPENAI_MODEL")
            api_key = os.getenv("OPENAI_API_KEY")
            is_local_llm = False
            openai_client = OpenAIClient(model=openai_model, api_key=api_key, is_local_llm=is_local_llm)
        elif self._environment == "dev":
            # use local LLM
            openai_model = os.getenv("OPENAI_MODEL")
            api_key = os.getenv("OPENAI_API_KEY")
            server_url = os.getenv("OPENAI_SERVER_URL")
            if server_url is None:
                msg = "`OPENAI_SERVER_URL` must be provided"
                raise ValueError(msg)

            is_local_llm = True
            openai_client = OpenAIClient(
                model=openai_model, api_key=api_key, base_url=server_url, is_local_llm=is_local_llm
            )
        elif self.environment == "test":
            openai_client = OpenAIDummyClient()
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        return openai_client

    def _build_pg_client(self) -> PgVectorClient:
        """Build the PostgreSQL client."""
        host = os.getenv("PG_HOST")
        port = os.getenv("PG_PORT")
        db_name = os.getenv("PG_DB_NAME")
        user = os.getenv("PG_USER")
        password = os.getenv("PG_PASSWORD")

        return PgVectorClient(host=host, port=port, db_name=db_name, user=user, password=password)

    def _build_documents_repository(self) -> DocumentsRepositoryInterface:
        """Build the DocumentsRepository."""
        return DocumentsRepository(self._pg_client)

    def get_openai_client(self) -> OpenAIClientInterface:
        """Get the OpenAI client."""
        return self._openai_client

    def get_docs_repository(self) -> DocumentsRepositoryInterface:
        """Get the Documents Repository."""
        return self._documents_repository
