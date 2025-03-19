"""Docs Agent Use Case."""

import os

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def check_up_docs():
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


def check_up_openai_docs():
    # Load documents from a directory
    documents = SimpleDirectoryReader("storage").load_data()

    openai_model = os.getenv("OPENAI_MODEL")
    if not openai_model:
        raise ValueError("env 'OPENAI_MODEL' must be set")

    # Create an LLM (Language Model)
    llm = OpenAI(model=openai_model, temperature=0.5)

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents, llm=llm)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Run a query
    response = query_engine.query("What is the main topic of these documents?")

    print(response)


def check_up_openai_embedded_docs():
    # Create some sample documents. This works as kind of data source.
    documents = [
        Document(text="LlamaIndex is a data framework for LLM applications."),
        Document(text="It provides tools for indexing, querying, and augmenting LLM capabilities."),
        Document(text="LlamaIndex supports various data connectors and index types."),
    ]

    # Initialize OpenAI LLM and embedding model
    openai_model = os.getenv("OPENAI_MODEL")
    if not openai_model:
        raise ValueError("env 'OPENAI_MODEL' must be set")

    llm = OpenAI(model=openai_model, temperature=0)
    embed_model = OpenAIEmbedding()

    # Create a vector store index
    index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Run a query
    response = query_engine.query("What is LlamaIndex and what does it do?")
    print(response)
