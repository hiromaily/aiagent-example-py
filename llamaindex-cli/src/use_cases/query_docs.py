"""Docs Agent Use Case."""

# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.base_query_engine import BaseQueryEngine


class DocsAgent:
    """Docs Agent Use Case."""

    def __init__(self, query_engine: BaseQueryEngine) -> None:
        """Initialize the DocsAgent with a query engine."""
        self._query_engine = query_engine

    def check_up_docs(self) -> None:
        """Check up documents."""
        # Run a query
        response = self._query_engine.query("What is this document about?")
        print(response, "\n")

        # response = self._query_engine.query("Summarize the content of these documents.")
        # print(response, "\n")

    def check_up_llamaindex_docs(self) -> None:
        """Check up documents about LlamaIndex."""
        # Run a query
        response = self._query_engine.query("What is LlamaIndex and what does it do?")
        print(response)
