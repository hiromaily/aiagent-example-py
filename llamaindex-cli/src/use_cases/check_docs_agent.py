"""Docs Agent Use Case."""

import os

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

from llm.models import create_lmstudio_llm, create_openai_llm


def check_up_docs() -> None:
    """Check up documents."""
    # Load documents from a directory
    documents = SimpleDirectoryReader("storage").load_data()

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Run a query
    response = query_engine.query("What is this document about?")
    print(response, "\n")

    response = query_engine.query("Summarize the content of these documents.")
    print(response, "\n")


def check_up_openai_docs() -> None:
    """Check up documents with OpenAI specific model."""
    # Load documents from a directory
    documents = SimpleDirectoryReader("storage").load_data()

    # Create an LLM (Language Model)
    openai_model = os.getenv("OPENAI_MODEL")
    llm = create_openai_llm(openai_model, 0.5)

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents, llm=llm)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Run a query
    response = query_engine.query("What is the main topic of these documents?")

    print(response)


def check_up_openai_embedded_docs() -> None:
    """Check up documents with OpenAI LLM and embedding model."""
    # Create some sample documents. This works as kind of data source.
    documents = [
        Document(text="LlamaIndex is a data framework for LLM applications."),
        Document(text="It provides tools for indexing, querying, and augmenting LLM capabilities."),
        Document(text="LlamaIndex supports various data connectors and index types."),
    ]

    # Create an LLM (Language Model)
    openai_model = os.getenv("OPENAI_MODEL")
    llm = create_openai_llm(openai_model, 0)
    embed_model = OpenAIEmbedding()

    # Create a vector store index
    index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Run a query
    response = query_engine.query("What is LlamaIndex and what does it do?")
    print(response)


def check_up_lmstudio_docs() -> None:
    """Check up documents with LMStudio specific model."""
    # Load documents from a directory
    documents = SimpleDirectoryReader("storage").load_data()

    # Create an LLM (Language Model)
    model = os.getenv("LMSTUDIO_MODEL")
    llm = create_lmstudio_llm(model, 0.5)

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents, llm=llm)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Run a query
    response = query_engine.query("What is the main topic of these documents?")

    print(response)
