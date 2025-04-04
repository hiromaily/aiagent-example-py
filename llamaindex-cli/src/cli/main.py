"""main function for the CLI app."""

import asyncio
import os

import typer
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.lmstudio import LMStudio
from loguru import logger

from registry.registry import DependencyRegistry
from use_cases.tool import ToolAgent

# Create a Typer app
app = typer.Typer()

# OPENAI_MODEL=gpt-4o
# LMSTUDIO_MODEL=llama3


@app.command()
def docs_agent(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    storage_mode: str = typer.Option("dir", "--storage", "-s", help="storage mode. text or dir"),
) -> None:
    """Checking up documents agent."""
    logger.debug("docs_agent()")

    if storage_mode == "":
        msg = "parameter `--storage` must be provided"
        raise ValueError(msg)

    # Initialization
    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment, model)
    docs_agent = registry.get_query_docs_usecase(storage_mode)

    # Execute
    if storage_mode == "dir":
        docs_agent.check_up_docs()
    else:
        docs_agent.check_up_llamaindex_docs()


@app.command()
def tech_question_agent(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    question: str = typer.Option("", "--question", "-q", help="question to ask"),
    stream: bool = False,
    chat: bool = False,
) -> None:
    """Answer the question."""
    logger.debug("tech_question_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    # Initialization
    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment, model)
    tech_question_agent = registry.get_tech_question_docs_usecase()

    # Execute
    if stream:
        tech_question_agent.ask_stream(question)
    elif chat:
        tech_question_agent.ask_by_chat(question)
    else:
        tech_question_agent.ask(question)


@app.command()
def query_image_agent(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    image_path: str = typer.Option("", "--image", "-i", help="image path to query"),
) -> None:
    """Query about image."""
    logger.debug("query_image_agent()")

    if image_path == "":
        msg = "parameter `--image` must be provided"
        raise ValueError(msg)

    # Initialization
    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment, model)
    query_image_agent = registry.get_query_image_usecase()

    # Execute
    query_image_agent.ask(image_path)


@app.command()
def calc_tool_agent(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
) -> None:
    """Calc Tool Agent."""
    logger.debug("calc_tool_agent()")

    # Initialization
    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment, model)
    calc_tool_agent = registry.get_tool_usecase()

    # Execute
    asyncio.run(_async_calc_tool_agent(calc_tool_agent))


async def _async_calc_tool_agent(calc_tool_agent: ToolAgent) -> None:
    """Calc Tool Agent for Async."""
    await calc_tool_agent.ask_calc()


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
