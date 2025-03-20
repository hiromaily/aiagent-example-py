"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv

from registry.registry import DependencyRegistry
from use_cases.custom_agent import CustomTechnicalAgent

# load .env file
load_dotenv()

environment = os.getenv("ENV", "development")

# Create a Typer app
app = typer.Typer()


@app.command()
def custom_tech_agent() -> None:
    """Custom technology agent."""
    registry = DependencyRegistry(environment)
    openai_client = registry.get_openai_client()
    agent = CustomTechnicalAgent(openai_client)

    # execute
    question = "OpenAI"
    response = agent.query(question)
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


if __name__ == "__main__":
    app()
