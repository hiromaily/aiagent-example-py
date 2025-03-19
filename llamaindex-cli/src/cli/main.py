"""main function for the CLI app."""

import typer
from dotenv import load_dotenv

# from openai.client import OpenAIClient
# from use_cases.custom_agent import CustomTechnicalAgent
from use_cases.check_docs_agent import check_up_docs, check_up_openai_docs, check_up_openai_embedded_docs

# load .env file
load_dotenv()

# Create a Typer app
app = typer.Typer()


@app.command()
def docs_agent() -> None:
    """Checking up documents agent."""
    check_up_docs()


@app.command()
def docs_openai_agent() -> None:
    """Checking up documents with OpenAI agent."""
    check_up_openai_docs()


@app.command()
def docs_openai_embedded_agent() -> None:
    """Checking up documents with OpenAI agent."""
    check_up_openai_embedded_docs()


# @app.command()
# def custom_tech_agent() -> None:
#     """Custom technology agent."""
#     api_key = os.getenv("OPENAI_API_KEY")
#     openai_model = os.getenv("OPENAI_MODEL")
#     openai_client = OpenAIClient(api_key, openai_model)
#     agent = CustomTechnicalAgent(openai_client)

#     # execute
#     question = "OpenAI"
#     response = agent.query(question)
#     print(response)


if __name__ == "__main__":
    app()
