"""Registry Class."""

import os

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM

from infrastructure.llm.models import create_lmstudio_llm, create_openai_llm
from infrastructure.storages.document import DocumentList, StorageMode


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, environment: str, storage_mode: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._environment = environment
        self._llm = self._build_llm()
        self._document = self._build_document(storage_mode)
        self._query_engine = self._build_query()

    def _build_llm(self) -> LLM:
        """Build the LLM based on the environment."""
        if self._environment == "prod":
            # use OpenAI API
            # Create an LLM (Language Model)
            model = os.getenv("OPENAI_MODEL")
            llm = create_openai_llm(model, 0.5)
        elif self._environment == "dev":
            # use local LLM
            model = os.getenv("LMSTUDIO_MODEL")
            llm = create_lmstudio_llm(model, 0.5)
        # elif self.environment == "test":
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        return llm

    def _build_document(self, storage_mode: str) -> Document:
        """Build the document."""
        mode = StorageMode.from_str(storage_mode)
        return DocumentList(mode).get_document()

    def _build_query(self) -> BaseQueryEngine:
        """Build the LlamaIndex query."""
        # Create an index from the documents
        index = VectorStoreIndex.from_documents(self._document, llm=self._llm)

        # Create a query engine
        return index.as_query_engine()

    # def get_llm(self) -> LLM:
    #     """Get the LLM."""
    #     return self._llm

    # def get_document(self) -> LLM:
    #     """Get the LLM."""
    #     return self._document

    def get_query(self) -> BaseQueryEngine:
        """Get the LlamaIndex Query."""
        return self._query_engine
