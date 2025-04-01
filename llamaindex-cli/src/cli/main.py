"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv
from loguru import logger

from registry.registry import DependencyRegistry
from use_cases.query_docs import (  # check_up_docs_default,
    check_up_lmstudio_docs,
    check_up_openai_docs,
    check_up_openai_embedded_docs,
)

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

    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment, storage_mode)
    query_engine = registry.get_query()

    # Execute
    response = query_engine.query("What is this document about?")
    print(response, "\n")

    response = query_engine.query("Summarize the content of these documents.")
    print(response, "\n")
    # check_up_docs_default()


@app.command()
def docs_openai_agent() -> None:
    """Checking up documents with OpenAI agent."""
    logger.debug("docs_openai_agent()")
    check_up_openai_docs()


@app.command()
def docs_openai_embedded_agent() -> None:
    """Checking up documents with OpenAI agent."""
    logger.debug("docs_openai_embedded_agent()")
    check_up_openai_embedded_docs()


@app.command()
def docs_lmstudio_agent() -> None:
    """Checking up documents with LMStudio agent."""
    logger.debug("docs_lmstudio_agent()")
    check_up_lmstudio_docs()


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
