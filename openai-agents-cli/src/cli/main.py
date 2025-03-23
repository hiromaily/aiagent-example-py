"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv
from loguru import logger

from registry.registry import DependencyRegistry
from use_cases.custom_agent import APIMode, CustomTechnicalAgent
from use_cases.load_embedding import load_embedding

# Create a Typer app
app = typer.Typer()


def _query(question: str, mode: APIMode) -> None:
    """Query function."""
    logger.debug("_query()")

    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    response = agent.query_tech_guide(question, mode)
    print(response)
    # call embeddings
    if environment != "prod":
        return

    # call embeddings API
    embedding_list = agent.embedding(question)
    print(embedding_list)
    # Insert into DB
    logger.debug("insert into db")
    docs_repo = registry.get_docs_repository()
    docs_repo.insert_embeddings(embedding_list)
    docs_repo.close()


@app.command()
def query_tech_guide(
    question: str = typer.Option("", "--question", "-q", help="Question to ask the tech agent."),
) -> None:
    """Custom technology agent."""
    logger.debug("custom_tech_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    _query(question, APIMode.RESPONSE_API)


@app.command()
def query_tech_guide_chat(
    question: str = typer.Option("", "--question", "-q", help="Question to ask the tech agent."),
) -> None:
    """Custom technology agent."""
    logger.debug("custom_tech_chat_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    _query(question, APIMode.CHAT_COMPLETION_API)


def query(
    question: str = typer.Option("", "--question", "-q", help="Question to ask the agent."),
) -> None:
    """Chat agent."""
    logger.debug("query()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    environment = os.getenv("APP_ENV", "dev")
    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    response = agent.query(question, APIMode.CHAT_COMPLETION_API)
    print(response)
    # call embeddings
    if environment != "prod":
        return

    # call embeddings API
    embedding_list = agent.embedding(question)
    print(embedding_list)
    # Insert into DB
    logger.debug("insert into db")
    docs_repo = registry.get_docs_repository()
    docs_repo.insert_embeddings(embedding_list)
    docs_repo.close()


@app.command()
def news_agent() -> None:
    """News agent by web search."""
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
    environment = os.getenv("APP_ENV", "dev")

    # load embedding JSON file
    embedding_list = load_embedding()

    # Insert into DB
    logger.debug("insert into db")
    registry = DependencyRegistry(environment)
    docs_repo = registry.get_docs_repository()
    docs_repo.insert_embeddings(embedding_list)

    docs_repo.close()


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
