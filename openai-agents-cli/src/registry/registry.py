"""Registry Class."""

from loguru import logger

from env.env import EnvSettings
from infrastructure.repository.embedding import PgVectorEmbeddingRepository
from infrastructure.repository.interface import EmbeddingRepositoryInterface
from infrastructure.vectordb.pgvector.client import PgVectorClient
from openai_custom.client import APIMode, OpenAIClient
from openai_custom.dymmy import OpenAIDummyClient
from openai_custom.interface import OpenAIClientInterface
from use_cases.debug import DebugAgent
from use_cases.prompting import PromptingPatternAgent
from use_cases.query_agent import QueryAgent
from use_cases.search_db_agent import SearchVectorDBAgent
from use_cases.web_search_agent import WebSearchAgent


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, tool: str, model: str, embedding_model: str | None = None) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._settings = EnvSettings()
        self._tool = tool
        self._model = model
        self._embedding_model = embedding_model

    # --------------------------------------------------------------------------
    # OpenAI Client
    # --------------------------------------------------------------------------

    def _build_openai_client(self, model: str, embedding_model: str | None) -> OpenAIClientInterface:
        """Build the OpenAI client based on the environment."""
        # Note: it's ok that only variable declaration with type hint [Pending]
        openai_client: OpenAIClientInterface
        if self._tool == "openai":
            # use OpenAI API
            logger.debug(f"use OpenAI API: tool: {self._tool}, model:{model}, embedding_model:{embedding_model}")
            openai_client = OpenAIClient(
                model=model,
                api_key=self._settings.OPENAI_API_KEY,
                is_local_llm=False,
            )
        elif self._tool == "lmstudio":
            if not embedding_model:
                msg = "embedding_model must be provided"
                raise ValueError(msg)

            # use local LLM: LM Studio API
            logger.debug(f"use LocalLLM API: tool: {self._tool}, model:{model}, embedding_model:{embedding_model}")
            openai_client = OpenAIClient(
                model=model,
                embedding_model=embedding_model,
                api_key="lm-studio",
                base_url="http://localhost:1234/v1",
                is_local_llm=True,
            )
        elif self._tool == "ollama":
            if not embedding_model:
                msg = "embedding_model must be provided"
                raise ValueError(msg)

            # use local LLM: Ollama API
            logger.debug(f"use LocalLLM API: tool: {self._tool}, model:{model}, embedding_model:{embedding_model}")
            openai_client = OpenAIClient(
                model=model,
                embedding_model=embedding_model,
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

    def _build_openai_specific_client(self, model: str, embedding_model: str | None) -> OpenAIClientInterface:
        """Build the OpenAI client.

        Openai client is Dummy client when tools is not OpenAI.
        """
        openai_client: OpenAIClientInterface
        if self._tool == "openai":
            # use OpenAI API
            logger.debug(f"use OpenAI API: tool: {self._tool}, model:{model}, embedding_model:{embedding_model}")
            openai_client = OpenAIClient(
                model=model,
                api_key=self._settings.OPENAI_API_KEY,
                is_local_llm=False,
            )
        else:
            logger.debug("use Dummy API")
            openai_client = OpenAIDummyClient()
        return openai_client

    # --------------------------------------------------------------------------
    # VectorDB
    # --------------------------------------------------------------------------

    def _build_pg_client(self) -> PgVectorClient:
        """Build the PostgreSQL client."""
        return PgVectorClient(
            host=self._settings.PG_HOST,
            port=self._settings.PG_PORT,
            db_name=self._settings.PG_DB_NAME,
            user=self._settings.PG_USER,
            password=self._settings.PG_PASSWORD,
        )

    def _build_embedding_repository(self) -> EmbeddingRepositoryInterface:
        """Build the EmbeddingRepository."""
        pg_client = self._build_pg_client()
        # is_large_embedding = True if self._tool == "openai" else False
        is_large_embedding = bool(self._tool == "openai")
        return PgVectorEmbeddingRepository(pg_client, is_large_embedding)

    # --------------------------------------------------------------------------
    # Use cases
    # --------------------------------------------------------------------------
    def _build_prompt_agent_usecase(self, chat: bool) -> PromptingPatternAgent:
        self._openai_client = self._build_openai_client(self._model, self._embedding_model)
        embedding_repository = self._build_embedding_repository()
        api_mode = APIMode.CHAT_COMPLETION_API if chat else APIMode.RESPONSE_API
        return PromptingPatternAgent(self._openai_client, embedding_repository, self._tool, api_mode)

    def _build_query_agent_usecase(self, chat: bool) -> QueryAgent:
        self._openai_client = self._build_openai_client(self._model, self._embedding_model)
        embedding_repository = self._build_embedding_repository()
        api_mode = APIMode.CHAT_COMPLETION_API if chat else APIMode.RESPONSE_API
        return QueryAgent(self._openai_client, embedding_repository, self._tool, api_mode)

    def _build_web_search_agent_usecase(self) -> WebSearchAgent:
        self._openai_client = self._build_openai_specific_client(self._model, self._embedding_model)
        return WebSearchAgent(self._openai_client)

    def _build_debug_agent_usecase(self) -> DebugAgent:
        embedding_repository = self._build_embedding_repository()
        return DebugAgent(embedding_repository)

    def _build_search_vector_db_usecase(self) -> SearchVectorDBAgent:
        embedding_repository = self._build_embedding_repository()
        return SearchVectorDBAgent(embedding_repository)

    # --------------------------------------------------------------------------
    # Getter for use cases
    # --------------------------------------------------------------------------

    def get_prompt_agent(self, chat: bool) -> PromptingPatternAgent:
        """Get the PromptingPattern Agent."""
        return self._build_prompt_agent_usecase(chat)

    def get_query_agent(self, chat: bool) -> QueryAgent:
        """Get the Query Agent."""
        return self._build_query_agent_usecase(chat)

    def get_web_search_agent(self) -> WebSearchAgent:
        """Get the Web Search Agent."""
        return self._build_web_search_agent_usecase()

    def get_debug_agent(self) -> DebugAgent:
        """Get the Debug Agent."""
        return self._build_debug_agent_usecase()

    def get_search_vector_db_usecase(self) -> SearchVectorDBAgent:
        """Get the Search Vector DB Agent."""
        return self._build_search_vector_db_usecase()
