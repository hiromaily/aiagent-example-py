"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv
from loguru import logger

from registry.registry import DependencyRegistry
from use_cases.custom_agent import CustomTechnicalAgent

# load .env file
# load_dotenv()
load_dotenv(dotenv_path=".env.dev")

environment = os.getenv("APP_ENV", "dev")

# Create a Typer app
app = typer.Typer()


@app.command()
def custom_tech_agent(
    question: str = typer.Option("", "--question", "-q", help="Question to ask the tech agent."),
) -> None:
    """Custom technology agent."""
    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

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
    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    response = agent.query_with_chat(question)
    print(response)


@app.command()
def news_agent() -> None:
    """News agent."""
    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    response = agent.query_news()
    print(response)


@app.command()
def debug() -> None:
    """Debug command."""
    logger.debug("Debugging...")


if __name__ == "__main__":
    logger.info("Starting CLI app")
    logger.debug(f"environment: {environment}")

    app()
