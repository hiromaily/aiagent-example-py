"""Registry Class."""

from loguru import logger

from env.env import EnvSettings
from infrastructure.pgvector.client import PgVectorClient
from infrastructure.repository.documents import DocumentsRepository
from infrastructure.repository.interface import DocumentsRepositoryInterface
from openai_custom.client import OpenAIClient
from openai_custom.dymmy import OpenAIDummyClient
from openai_custom.interface import OpenAIClientInterface


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, tool: str, model: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._settings = EnvSettings()
        self._tool = tool
        self._openai_client = self._build_openai_client(model)
        self._pg_client = self._build_pg_client()
        self._documents_repository = self._build_documents_repository()

    def _build_openai_client(self, model: str) -> OpenAIClientInterface:
        """Build the OpenAI client based on the environment."""
        # Note: it's ok that only variable declaration with type hint [Pending]
        openai_client: OpenAIClientInterface
        if self._tool == "openai":
            # use OpenAI API
            logger.debug(f"use OpenAI API: tool: {self._tool}, model:{model}")
            openai_client = OpenAIClient(
                model=model,
                api_key=self._settings.OPENAI_API_KEY,
                is_local_llm=False,
            )
        elif self._tool == "lmstudio":
            # use local LLM
            logger.debug(f"use LocalLLM API: tool: {self._tool}, model:{model}")
            openai_client = OpenAIClient(
                model=model,
                api_key="lm-studio",
                base_url="http://localhost:1234/v1",
                is_local_llm=True,
            )
        elif self._tool == "ollama":
            logger.debug(f"use LocalLLM API: tool: {self._tool}, model:{model}")
            openai_client = OpenAIClient(
                model=model,
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                is_local_llm=True,
            )
        elif self._settings.APP_ENV == "test":
            logger.debug("use Dummy API")
            openai_client = OpenAIDummyClient()
        else:
            msg = f"Unknown LLM toolkit: {self._tool}"
            raise ValueError(msg)

        return openai_client

    def _build_pg_client(self) -> PgVectorClient:
        """Build the PostgreSQL client."""
        return PgVectorClient(
            host=self._settings.PG_HOST,
            port=self._settings.PG_PORT,
            db_name=self._settings.PG_DB_NAME,
            user=self._settings.PG_USER,
            password=self._settings.PG_PASSWORD,
        )

    def _build_documents_repository(self) -> DocumentsRepositoryInterface:
        """Build the DocumentsRepository."""
        return DocumentsRepository(self._pg_client)

    def get_openai_client(self) -> OpenAIClientInterface:
        """Get the OpenAI client."""
        return self._openai_client

    def get_docs_repository(self) -> DocumentsRepositoryInterface:
        """Get the Documents Repository."""
        return self._documents_repository
