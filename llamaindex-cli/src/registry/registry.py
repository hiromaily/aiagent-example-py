"""Registry Class."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from loguru import logger

from entities.ai_tools import add, multiply
from infrastructure.llm.models import create_lmstudio_embedding_llm, create_lmstudio_llm, create_openai_llm
from infrastructure.storages.document import DocumentList, StorageMode
from use_cases.query_docs import DocsAgent
from use_cases.query_image import QueryImageAgent
from use_cases.tech_question import TechQuestionAgent
from use_cases.tool import ToolAgent


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, environment: str, model: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._environment = environment
        self._llm = self._build_llm(model)

    def _build_llm(self, model: str) -> LLM:
        """Build the LLM based on the environment."""
        if self._environment == "prod":
            # use OpenAI API
            logger.debug(f"use OpenAI API: {model}")
            # Create an LLM (Language Model)
            llm = create_openai_llm(model, 0.5)
        elif self._environment == "dev":
            # use local LLM
            logger.debug(f"use local LLM {model}")
            llm = create_lmstudio_llm(model, 0.5)
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        return llm

    def _build_document(self, storage_mode: str) -> Document:
        """Build the document."""
        logger.debug(f"storage_mode: {storage_mode}")
        mode = StorageMode.from_str(storage_mode)
        return DocumentList(mode).get_document()

    def _build_query(self) -> BaseQueryEngine:
        """Build the LlamaIndex query."""
        # Create an index from the documents
        if self._environment == "prod":
            # use OpenAI API
            logger.debug("use OpenAI API")
            # embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
            index = VectorStoreIndex.from_documents(self._document)
        elif self._environment == "dev":
            # use local LLM
            logger.debug("use local LLM")
            embed_model = create_lmstudio_embedding_llm()
            index = VectorStoreIndex.from_documents(self._document, embed_model=embed_model)
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        # Create a query engine
        return index.as_query_engine(self._llm)

    def _build_mathematical_tool_workflow(self) -> FunctionAgent:
        """Build the mathematical tool workflow."""
        return FunctionAgent(
            name="mathematical tools",
            description="Agent that can perform basic mathematical operations.",
            tools=[multiply, add],
            llm=self._llm,
            system_prompt="You are an agent that can perform basic mathematical operations using tools.",
        )

    def _build_docs_usecase(self) -> DocsAgent:
        """Build the docs usecase."""
        return DocsAgent(self._query_engine)

    def _build_tech_question_usecase(self) -> TechQuestionAgent:
        """Build the docs usecase."""
        return TechQuestionAgent(self._llm)

    def _build_query_image_usecase(self) -> QueryImageAgent:
        """Build the query image usecase."""
        return QueryImageAgent(self._llm)

    def _build_tool_usecase(self) -> ToolAgent:
        """Build the query image usecase."""
        return ToolAgent(self._llm, self._tool_workflow)

    # def get_llm(self) -> LLM:
    #     """Get the LLM."""
    #     return self._llm

    # def get_document(self) -> LLM:
    #     """Get the LLM."""
    #     return self._document

    # def get_query(self) -> BaseQueryEngine:
    #     """Get the LlamaIndex Query."""
    #     return self._query_engine

    def get_query_docs_usecase(self, storage_mode: str) -> DocsAgent:
        """Get the docs usecase."""
        self._document = self._build_document(storage_mode)
        self._query_engine = self._build_query()
        self._docs_usecase = self._build_docs_usecase()

        return self._docs_usecase

    def get_tech_question_docs_usecase(self) -> TechQuestionAgent:
        """Get the tech question usecase."""
        self._tech_question_usecase = self._build_tech_question_usecase()

        return self._tech_question_usecase

    def get_query_image_usecase(self) -> QueryImageAgent:
        """Get the query image usecase."""
        self._tech_query_image_usecase = self._build_query_image_usecase()

        return self._tech_query_image_usecase

    def get_tool_usecase(self) -> ToolAgent:
        """Get the tool usecase."""
        self._tool_workflow = self._build_mathematical_tool_workflow()
        self._tool_usecase = self._build_tool_usecase()

        return self._tool_usecase
