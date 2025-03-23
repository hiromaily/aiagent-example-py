"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv
from loguru import logger

from registry.registry import DependencyRegistry
from use_cases.custom_agent import CustomTechnicalAgent
from use_cases.load_embedding import load_embedding

# Create a Typer app
app = typer.Typer()


@app.command()
def custom_tech_agent(
    question: str = typer.Option("", "--question", "-q", help="Question to ask the tech agent."),
) -> None:
    """Custom technology agent."""
    logger.debug("custom_tech_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    response = agent.query(question)
    print(response)


@app.command()
def custom_tech_chat_agent(
    question: str = typer.Option("", "--question", "-q", help="Question to ask the tech agent."),
) -> None:
    """Custom technology agent."""
    logger.debug("custom_tech_chat_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    response = agent.query_with_chat(question)
    print(response)
    # call embeddings
    if environment == "prod":
        data = agent.embedding(question)
        print(data)


@app.command()
def news_agent() -> None:
    """News agent."""
    logger.debug("news_agent()")

    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    response = agent.query_news()
    print(response)


@app.command()
def embedding() -> None:
    """Embedding command."""
    logger.debug("embedding()")

    # load embedding JSON file
    embedding_list = load_embedding()
    print(embedding_list)


@app.callback()
def main(local: bool = False) -> None:
    """Frist endpoint after app()."""
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
