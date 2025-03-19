"""main function for the CLI app."""

import os

import typer
from dotenv import load_dotenv

from openai_custom.client import OpenAIClient
from use_cases.custom_agent import CustomTechnicalAgent

# load .env file
load_dotenv()

# Create a Typer app
app = typer.Typer()


@app.command()
def custom_tech_agent() -> None:
    """Custom technology agent."""
    api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    openai_client = OpenAIClient(api_key, openai_model)
    agent = CustomTechnicalAgent(openai_client)

    # execute
    question = "OpenAI"
    response = agent.query(question)
    print(response)

@app.command()
def docs_agent() -> None:
    """Checking up documents agent."""
    print("Checking up documents agent")


if __name__ == "__main__":
    app()
