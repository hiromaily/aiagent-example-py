"""main function for the CLI app."""

import typer

# from dotenv import load_dotenv
from loguru import logger

from env.env import EnvSettings
from registry.registry import DependencyRegistry

# Create a Typer app
app = typer.Typer()


@app.command()
def query_tech_guide(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    embedding_model: str = typer.Option(
        "text-embedding-ada-002", "--embedding-model", "-e", help="LLM embedding model name"
    ),
    question: str = typer.Option("", "--question", "-q", help="Question to ask the tech agent."),
    chat: bool = False,
) -> None:
    """Custom technology agent."""
    logger.info("custom_tech_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    # Initialization
    registry = DependencyRegistry(tool, model, embedding_model)
    agent = registry.get_query_agent(chat)

    # Execute
    agent.query_tech_guide(question)


@app.command()
def query_common(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    embedding_model: str = typer.Option(
        "text-embedding-ada-002", "--embedding-model", "-e", help="LLM embedding model name"
    ),
    question: str = typer.Option("", "--question", "-q", help="Question to ask the agent."),
    chat: bool = False,
) -> None:
    """Query common question."""
    logger.info("common query()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    # Initialization
    registry = DependencyRegistry(tool, model, embedding_model)
    agent = registry.get_query_agent(chat)

    # execute
    agent.query_common(question)


@app.command()
def prompt_pattern(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    embedding_model: str = typer.Option(
        "text-embedding-ada-002", "--embedding-model", "-e", help="LLM embedding model name"
    ),
    chat: bool = False,
    pattern: str = typer.Option("zero-shot", "--pattern", "-p", help="Prompting pattern: zero-shot, few-shot, etc."),
) -> None:
    """Prompt pattern agent."""
    logger.info("prompt_pattern_agent()")

    # Initialization
    registry = DependencyRegistry(tool, model, embedding_model)
    agent = registry.get_prompt_agent(chat)

    # Execute
    agent.call(pattern)


@app.command()
def news_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
) -> None:
    """News agent by web search.

    This agent works with only OpenAI API.
    """
    logger.debug("news_agent()")

    registry = DependencyRegistry(tool, model)
    agent = registry.get_web_search_agent()

    # execute
    response = agent.query_news()
    print(response)


@app.command()
def embedding(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
) -> None:
    """Embedding command for debug."""
    logger.debug("embedding()")

    registry = DependencyRegistry(tool, model)
    debug_agent = registry.get_debug_agent()
    debug_agent.embedding("storage/embedding01.json")


@app.command()
def search_similarity(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    content_id: int = typer.Option(0, "--id", "-q", help="item_contents.id."),
) -> None:
    """Search similarity."""
    logger.debug("search_similarity()")

    if content_id == 0:
        msg = "parameter `--id` must be provided"
        raise ValueError(msg)

    registry = DependencyRegistry(tool, model)
    search_vector_db = registry.get_search_vector_db_usecase()

    # Search target item_content from DB `item_contents`
    search_vector_db.search_similarity(content_id)


@app.callback()
def main(env: str = ".env") -> None:
    """First endpoint after app()."""
    logger.debug("main()")

    # load .env file
    # if local:
    #     logger.debug("Running in local mode")
    #     load_dotenv(dotenv_path=".env.dev")
    # else:
    #     load_dotenv()

    # environment = os.getenv("APP_ENV", "dev")
    EnvSettings.set_env(env)


if __name__ == "__main__":
    logger.info("Starting CLI app")
    app()
