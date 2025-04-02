"""Registry Class."""

import os

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from loguru import logger

from infrastructure.llm.models import create_lmstudio_embedding_llm, create_lmstudio_llm, create_openai_llm
from infrastructure.storages.document import DocumentList, StorageMode
from use_cases.query_docs import DocsAgent


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, environment: str, storage_mode: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._environment = environment
        self._llm = self._build_llm()
        self._document = self._build_document(storage_mode)
        self._query_engine = self._build_query()
        self._docs_usecase = self._build_docs_usecase()

    def _build_llm(self) -> LLM:
        """Build the LLM based on the environment."""
        if self._environment == "prod":
            # use OpenAI API
            logger.debug("use OpenAI API")
            # Create an LLM (Language Model)
            model = os.getenv("OPENAI_MODEL")
            llm = create_openai_llm(model, 0.5)
        elif self._environment == "dev":
            # use local LLM
            logger.debug("use local LLM")
            model = os.getenv("LMSTUDIO_MODEL")
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
            index = VectorStoreIndex.from_documents(self._document, llm=self._llm)
        elif self._environment == "dev":
            # use local LLM
            logger.debug("use local LLM")
            embed_model = create_lmstudio_embedding_llm()
            index = VectorStoreIndex.from_documents(self._document, llm=self._llm, embed_model=embed_model)
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        # Create a query engine
        return index.as_query_engine()

    def _build_docs_usecase(self) -> Document:
        """Build the docs usecase."""
        return DocsAgent(self._query_engine)

    # def get_llm(self) -> LLM:
    #     """Get the LLM."""
    #     return self._llm

    # def get_document(self) -> LLM:
    #     """Get the LLM."""
    #     return self._document

    # def get_query(self) -> BaseQueryEngine:
    #     """Get the LlamaIndex Query."""
    #     return self._query_engine

    def get_usecase(self) -> DocsAgent:
        """Get the docs usecase."""
        return self._docs_usecase
