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

    def __init__(self) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._settings = EnvSettings()
        self._openai_client = self._build_openai_client()
        self._pg_client = self._build_pg_client()
        self._documents_repository = self._build_documents_repository()

    def _build_openai_client(self) -> OpenAIClientInterface:
        """Build the OpenAI client based on the environment."""
        # openai_model = os.getenv("OPENAI_MODEL")
        # if openai_model is None:
        #     msg = "`OPENAI_MODEL` must be provided"
        #     raise ValueError(msg)

        # api_key = os.getenv("OPENAI_API_KEY")
        # if api_key is None:
        #     msg = "`OPENAI_API_KEY` must be provided"
        #     raise ValueError(msg)

        # Note: it's ok tthat only variable declaration with type hint [Pending]
        openai_client: OpenAIClientInterface
        if self._settings.APP_ENV == "prod":
            # use OpenAI API
            logger.debug(f"use OpenAI API: model:{self._settings.OPENAI_MODEL}")
            openai_client = OpenAIClient(
                model=self._settings.OPENAI_MODEL, api_key=self._settings.OPENAI_API_KEY, is_local_llm=False
            )
        elif self._settings.APP_ENV == "dev":
            # use local LLM
            logger.debug(f"use LocalLLM API: model:{self._settings.OPENAI_MODEL}")
            openai_client = OpenAIClient(
                model=self._settings.OPENAI_MODEL,
                api_key=self._settings.OPENAI_API_KEY,
                base_url=self._settings.OPENAI_SERVER_URL,
                is_local_llm=True,
            )
        elif self._settings.APP_ENV == "test":
            logger.debug("use Dummy API")
            openai_client = OpenAIDummyClient()
        else:
            msg = "Unknown environment"
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
