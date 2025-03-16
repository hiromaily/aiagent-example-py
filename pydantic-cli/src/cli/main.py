"""main function for the CLI app."""

import typer
from dotenv import load_dotenv

from infrastructure.model import get_openai_model
from use_cases.agent_structured_dependencies import agent_structured_dependencies_use_case
from use_cases.agent_structured_response import agent_structured_response_use_case
from use_cases.agent_with_self_correction import agent_with_self_correction_use_case
from use_cases.agent_with_tools import agent_with_tools_use_case
from use_cases.simple_agent import simple_agent_use_case

# load .env file
load_dotenv()
model = get_openai_model()

# Create a Typer app
app = typer.Typer()


@app.command()
def simple_agent() -> None:
    """This example demonstrates the basic usage of PydanticAI agents."""
    simple_agent_use_case(model)


@app.command()
def agent_structured_response() -> None:
    """Agent with Structured Response."""
    agent_structured_response_use_case(model)


@app.command()
def agent_structured_dependencies() -> None:
    """Agent with Structured Response & Dependencies."""
    agent_structured_dependencies_use_case(model)


@app.command()
def agent_with_tools() -> None:
    """Agent with Tools."""
    agent_with_tools_use_case(model)


@app.command()
def agent_with_self_correction() -> None:
    """Agent with Reflection and Self-Correction."""
    agent_with_self_correction_use_case(model)


if __name__ == "__main__":
    app()
