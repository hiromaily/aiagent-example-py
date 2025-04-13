"""Registry Class."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from loguru import logger

from agents.workflow import (
    build_financial_tavily_tool_workflow,
    build_financial_yahoo_financial_tool_workflow,
    build_mathematical_tool_workflow,
    build_standard_workflow,
)
from env.env import EnvSettings
from infrastructure.llm.models import (
    create_lmstudio_embedding_llm,
    create_lmstudio_llm,
    create_ollama_embedding_llm,
    create_ollama_llm,
    create_openai_llm,
)
from infrastructure.storages.document import DocumentList, StorageMode
from use_cases.any_question import AnyQuestionAgent
from use_cases.query_docs import DocsAgent
from use_cases.query_image import QueryImageAgent
from use_cases.tech_question import TechQuestionAgent
from use_cases.tool import ToolAgent


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, model: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._settings = EnvSettings()
        self._llm = self._build_llm(model, self._settings.OPENAI_API_KEY)

    def _build_llm(self, model: str, api_key: str, temperature: float = 0.5) -> LLM:
        """Build the LLM based on the environment."""
        if self._settings.APP_ENV == "prod":
            # use OpenAI API
            logger.debug(f"use OpenAI API: {model}")
            llm = create_openai_llm(model, api_key, temperature)
        elif self._settings.APP_ENV == "dev":
            # use local LLM
            logger.debug(f"use local LLM {model}, toolkit: {self._settings.LLM_TOOLKIT}")
            if self._settings.LLM_TOOLKIT == "lmstudio":
                # Note: `FunctionCallingLLM` doesn't work with `LMStudio`
                llm = create_lmstudio_llm(model, temperature)
            elif self._settings.LLM_TOOLKIT == "ollama":
                llm = create_ollama_llm(model, temperature)
            else:
                msg = f"Unknown LLM toolkit: {self._settings.LLM_TOOLKIT}"
                raise ValueError(msg)
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        return llm

    def _build_document(self, storage_mode: str) -> Document:
        """Build the document."""
        logger.debug(f"storage_mode: {storage_mode}")
        mode = StorageMode.from_str(storage_mode)
        return DocumentList(mode).get_document()

    def _build_query(self, embedding_model: str) -> BaseQueryEngine:
        """Build the LlamaIndex query."""
        # Create an index from the documents
        if self._settings.APP_ENV == "prod":
            # use OpenAI API
            logger.debug("use OpenAI API")
            embed_model = OpenAIEmbedding(model=embedding_model, api_key=self._settings.OPENAI_API_KEY)
            index = VectorStoreIndex.from_documents(self._document, embed_model=embed_model)
        elif self._settings.APP_ENV == "dev":
            # use local LLM
            logger.debug("use local LLM")
            if self._settings.LLM_TOOLKIT == "lmstudio":
                embed_model = create_lmstudio_embedding_llm(embedding_model)
            elif self._settings.LLM_TOOLKIT == "ollama":
                embed_model = create_ollama_embedding_llm(embedding_model)
            else:
                msg = f"Unknown LLM toolkit: {self._settings.LLM_TOOLKIT}"
                raise ValueError(msg)

            index = VectorStoreIndex.from_documents(self._document, embed_model=embed_model)
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        # Create a query engine
        return index.as_query_engine(self._llm)

    def _build_docs_usecase(self) -> DocsAgent:
        """Build the docs usecase."""
        return DocsAgent(self._query_engine)

    def _build_tech_question_usecase(self) -> TechQuestionAgent:
        """Build the tech question usecase."""
        return TechQuestionAgent(self._llm)

    def _build_any_question_usecase(self) -> AnyQuestionAgent:
        """Build the any question usecase."""
        agent = build_standard_workflow(self._llm)
        return AnyQuestionAgent(agent)

    def _build_query_image_usecase(self) -> QueryImageAgent:
        """Build the query image usecase."""
        return QueryImageAgent(self._llm)

    def _build_tool_usecase(self) -> ToolAgent:
        """Build the query image usecase."""
        self._mathematical_tool_workflow = build_mathematical_tool_workflow(self._llm)
        self._yahoo_financial_tool_workflow = build_financial_yahoo_financial_tool_workflow(self._llm)
        self._tavily_tools_workflow = build_financial_tavily_tool_workflow(self._llm, self._settings.TAVILY_API_KEY)

        return ToolAgent(
            self._llm,
            self._mathematical_tool_workflow,
            self._yahoo_financial_tool_workflow,
            self._tavily_tools_workflow,
        )

    # def get_llm(self) -> LLM:
    #     """Get the LLM."""
    #     return self._llm

    # def get_document(self) -> LLM:
    #     """Get the LLM."""
    #     return self._document

    # def get_query(self) -> BaseQueryEngine:
    #     """Get the LlamaIndex Query."""
    #     return self._query_engine

    def get_query_docs_usecase(self, storage_mode: str, embedding_model: str) -> DocsAgent:
        """Get the docs usecase."""
        self._document = self._build_document(storage_mode)
        self._query_engine = self._build_query(embedding_model)
        self._docs_usecase = self._build_docs_usecase()

        return self._docs_usecase

    def get_tech_question_usecase(self) -> TechQuestionAgent:
        """Get the tech question usecase."""
        self._tech_question_usecase = self._build_tech_question_usecase()

        return self._tech_question_usecase

    def get_any_question_usecase(self) -> AnyQuestionAgent:
        """Get the any question usecase."""
        self._any_question_usecase = self._build_any_question_usecase()

        return self._any_question_usecase

    def get_query_image_usecase(self) -> QueryImageAgent:
        """Get the query image usecase."""
        self._tech_query_image_usecase = self._build_query_image_usecase()

        return self._tech_query_image_usecase

    def get_tool_usecase(self) -> ToolAgent:
        """Get the tool usecase."""
        self._tool_usecase = self._build_tool_usecase()

        return self._tool_usecase
