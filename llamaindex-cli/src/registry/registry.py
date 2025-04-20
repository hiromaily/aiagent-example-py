"""Registry Class."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from loguru import logger

from agents.ai_tools import get_search_web, record_notes, review_report, write_report
from agents.workflow import (
    build_financial_tavily_tool_workflow,
    build_financial_yahoo_financial_tool_workflow,
    build_mathematical_tool_workflow,
    build_multi_workflow,
    build_research_workflow,
    build_review_workflow,
    build_standard_workflow,
    build_write_workflow,
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
from infrastructure.storages.github import GithubDocumentList
from use_cases.any_question import AnyQuestionAgent
from use_cases.github_index import GithubIndex
from use_cases.multi_agent import MultiAgent
from use_cases.query_docs import DocsAgent
from use_cases.query_image import QueryImageAgent
from use_cases.tech_question import TechQuestionAgent
from use_cases.tool import ToolAgent


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, tool: str, model: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._settings = EnvSettings()
        self._tool = tool
        self._llm = self._build_llm(model, self._settings.OPENAI_API_KEY)

    def _build_llm(self, model: str, api_key: str, temperature: float = 0.5) -> LLM:
        """Build the LLM based on the environment."""
        if self._tool == "openai":
            # use OpenAI API
            logger.debug(f"use OpenAI API: {model}, toolkit: {self._tool}")
            llm = create_openai_llm(model, api_key, temperature)
        elif self._tool == "lmstudio":
            # use local LLM
            logger.debug(f"use local LLM {model}, toolkit: {self._tool}")
            # Note: `FunctionCallingLLM` doesn't work with `LMStudio`
            llm = create_lmstudio_llm(model, temperature)
        elif self._tool == "ollama":
            # use local LLM
            logger.debug(f"use local LLM {model}, toolkit: {self._tool}")
            llm = create_ollama_llm(model, temperature)
        else:
            msg = f"Unknown LLM toolkit: {self._tool}"
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
        if self._tool == "openai":
            # use OpenAI API
            logger.debug(f"use OpenAI API: {embedding_model}, toolkit: {self._tool}")
            embed_model = OpenAIEmbedding(model=embedding_model, api_key=self._settings.OPENAI_API_KEY)
            index = VectorStoreIndex.from_documents(self._document, embed_model=embed_model)
        elif self._tool == "lmstudio":
            # use local LLM
            logger.debug(f"use local LLM {embedding_model}, toolkit: {self._tool}")
            embed_model = create_lmstudio_embedding_llm(embedding_model)
        elif self._tool == "ollama":
            # use local LLM
            logger.debug(f"use local LLM {embedding_model}, toolkit: {self._tool}")
            embed_model = create_ollama_embedding_llm(embedding_model)
        else:
            msg = f"Unknown LLM toolkit: {self._tool}"
            raise ValueError(msg)

        index = VectorStoreIndex.from_documents(self._document, embed_model=embed_model)

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
        mathematical_tool_workflow = build_mathematical_tool_workflow(self._llm)
        yahoo_financial_tool_workflow = build_financial_yahoo_financial_tool_workflow(self._llm)
        tavily_tools_workflow = build_financial_tavily_tool_workflow(self._llm, self._settings.TAVILY_API_KEY)

        return ToolAgent(
            self._llm,
            mathematical_tool_workflow,
            yahoo_financial_tool_workflow,
            tavily_tools_workflow,
        )

    def _build_multi_agent_usecase(self) -> MultiAgent:
        """Build the multi agent usecase."""
        search_web = get_search_web(self._settings.TAVILY_API_KEY)
        research_workflow = build_research_workflow(self._llm, [search_web, record_notes])
        write_workflow = build_write_workflow(self._llm, [write_report])
        review_workflow = build_review_workflow(self._llm, [review_report])
        multi_workflow = build_multi_workflow([research_workflow, write_workflow, review_workflow])

        return MultiAgent(multi_workflow)

    def _build_github_index_usecase(self, embedding_model: str) -> GithubIndex:
        """Build the github index usecase."""
        if self._tool == "openai":
            # use OpenAI API
            logger.debug(f"use OpenAI API embedding_model: {embedding_model}, toolkit: {self._tool}")
            embed_model = OpenAIEmbedding(model=embedding_model, api_key=self._settings.OPENAI_API_KEY)
        elif self._tool == "lmstudio":
            # use local LLM
            logger.debug(f"use local embedding_model: {embedding_model}, toolkit: {self._tool}")
            embed_model = create_lmstudio_embedding_llm(embedding_model)
        elif self._tool == "ollama":
            # use local LLM
            logger.debug(f"use local embedding_model: {embedding_model}, toolkit: {self._tool}")
            embed_model = create_ollama_embedding_llm(embedding_model)
        else:
            msg = f"Unknown LLM toolkit: {self._tool}"
            raise ValueError(msg)

        github_docs = GithubDocumentList(
            self._settings.GITHUB_TOKEN, self._settings.GITHUB_OWNER, self._settings.GITHUB_REPO
        )
        # TODO: storage
        vector_store = SimpleVectorStore()

        return GithubIndex(self._llm, embed_model, github_docs, vector_store)

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

    def get_multi_agent_usecase(self) -> MultiAgent:
        """Get the multi agent usecase."""
        self._multi_agent_usecase = self._build_multi_agent_usecase()
        return self._multi_agent_usecase

    def get_github_index_usecase(self, embedding_model: str) -> GithubIndex:
        """Get the multi agent usecase."""
        self._github_index_usecase = self._build_github_index_usecase(embedding_model)
        return self._github_index_usecase
