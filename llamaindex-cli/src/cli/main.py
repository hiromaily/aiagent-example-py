"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.lmstudio import LMStudio
from loguru import logger

from registry.registry import DependencyRegistry

# Create a Typer app
app = typer.Typer()


@app.command()
def docs_agent(
    storage_mode: str = typer.Option("dir", "--storage", "-s", help="storage mode. text or dir"),
) -> None:
    """Checking up documents agent."""
    logger.debug("docs_agent()")

    if storage_mode == "":
        msg = "parameter `--storage` must be provided"
        raise ValueError(msg)

    # Initialization
    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment)
    docs_agent = registry.get_query_docs_usecase(storage_mode)

    # Execute
    if storage_mode == "dir":
        docs_agent.check_up_docs()
    else:
        docs_agent.check_up_llamaindex_docs()


@app.command()
def tech_question_agent(
    question: str = typer.Option("", "--question", "-q", help="question to ask"),
) -> None:
    """Answer the question."""
    logger.debug("tech_question_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    # Initialization
    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment)
    tech_question_agent = registry.get_tech_question_docs_usecase()

    # Execute
    tech_question_agent.ask(question)


@app.command()
def local_llm() -> None:
    """Use Local LLM. For just example."""
    # Load documents from a directory
    documents = SimpleDirectoryReader("storage").load_data()
    # Create an LLM (Language Model)
    llm = LMStudio(
        model_name="llama3",
        base_url="http://localhost:1234/v1",
        temperature=0.5,
    )
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key="lm-studio",
        api_base="http://localhost:1234/v1",
    )

    # Create an index from the documents
    # Note: `llm`` does not need to be set into VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    # Create a query engine
    # Note: `llm`` must be set into as_query_engine() as well.
    query_engine = index.as_query_engine(llm=llm)
    # Run a query
    response = query_engine.query("What is the main topic of these documents?")
    print(response)


@app.callback()
def main(local: bool = False) -> None:
    """First endpoint after app()."""
    logger.debug("main()")

    # load .env file
    if local:
        logger.debug("Running in local mode")
        load_dotenv(dotenv_path=".env.dev")
    else:
        load_dotenv()

    environment = os.getenv("APP_ENV", "dev")
    logger.debug(f"environment: {environment}")


if __name__ == "__main__":
    logger.info("Starting CLI app")
    app()
