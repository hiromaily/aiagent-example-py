"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv
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
    registry = DependencyRegistry(environment, storage_mode)
    docs_agent = registry.get_usecase()

    # Execute
    if storage_mode == "dir":
        docs_agent.check_up_docs()
    else:
        docs_agent.check_up_llamaindex_docs()


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
