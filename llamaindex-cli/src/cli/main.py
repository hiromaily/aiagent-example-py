"""main function for the CLI app."""

import asyncio

import typer

# from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.ollama import Ollama
from loguru import logger

from env.env import EnvSettings
from registry.registry import DependencyRegistry
from use_cases.any_question import AnyQuestionAgent
from use_cases.multi_agent import MultiAgent
from use_cases.tool import ToolAgent

# Create a Typer app
app = typer.Typer()

# OPENAI_MODEL=gpt-4o
# LMSTUDIO_MODEL=llama3


@app.command()  # type: ignore[misc]
def docs_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    embedding_model: str = typer.Option(
        "text-embedding-ada-002", "--embedding-model", "-e", help="LLM embedding model name"
    ),
    storage_mode: str = typer.Option("dir", "--storage", "-s", help="storage mode. text or dir"),
) -> None:
    """Checking up documents agent."""
    logger.debug("docs_agent()")

    if storage_mode == "":
        msg = "parameter `--storage` must be provided"
        raise ValueError(msg)

    # Initialization
    registry = DependencyRegistry(tool, model)
    docs_agent = registry.get_query_docs_usecase(storage_mode, embedding_model)

    # Execute
    if storage_mode == "dir":
        docs_agent.check_up_docs()
    else:
        docs_agent.check_up_llamaindex_docs()


@app.command()  # type: ignore[misc]
def tech_question_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    question: str = typer.Option("", "--question", "-q", help="question to ask"),
    stream: bool = False,
    chat: bool = False,
) -> None:
    """Answer the question."""
    logger.debug("tech_question_agent()")

    if question == "":
        msg = "parameter `--question` must be provided"
        raise ValueError(msg)

    # Initialization
    registry = DependencyRegistry(tool, model)
    tech_question_agent = registry.get_tech_question_usecase()

    # Execute
    if stream:
        tech_question_agent.ask_stream(question)
    elif chat:
        tech_question_agent.ask_by_chat(question)
    else:
        tech_question_agent.ask(question)


@app.command()  # type: ignore[misc]
def query_image_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    image_path: str = typer.Option("", "--image", "-i", help="image path to query"),
) -> None:
    """Query about image."""
    logger.debug("query_image_agent()")

    if image_path == "":
        msg = "parameter `--image` must be provided"
        raise ValueError(msg)

    # Initialization
    registry = DependencyRegistry(tool, model)
    query_image_agent = registry.get_query_image_usecase()

    # Execute
    query_image_agent.ask(image_path)


@app.command()  # type: ignore[misc]
def calc_tool_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    question: str = typer.Option("What is 1 + 1?", "--question", "-q", help="question to ask"),
) -> None:
    """Calc Tool Agent."""
    logger.debug("calc_tool_agent()")

    # Initialization
    registry = DependencyRegistry(tool, model)
    calc_tool_agent = registry.get_tool_usecase()

    # Execute
    asyncio.run(_async_calc_tool_agent(calc_tool_agent, question))


async def _async_calc_tool_agent(calc_tool_agent: ToolAgent, question: str) -> None:
    """Calc Tool Agent for Async."""
    await calc_tool_agent.ask_calc(question)


@app.command()  # type: ignore[misc]
def finance_tool_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    company: str = typer.Option("NVIDIA", "--company", "-q", help="company name to get stock price"),
    tavily: bool = False,
    stream: bool = False,
) -> None:
    """Finance Tool Agent."""
    logger.debug("finance_tool_agent()")

    # Initialization
    registry = DependencyRegistry(tool, model)
    tool_agent = registry.get_tool_usecase()

    # Execute
    asyncio.run(_async_finance_tool_agent(tool_agent, company, tavily, stream))


async def _async_finance_tool_agent(tool_agent: ToolAgent, company: str, tavily: bool, stream: bool) -> None:
    """Calc Tool Agent for Async."""
    if tavily:
        if stream:
            await tool_agent.ask_finance_by_tavily_streaming(company)
        else:
            await tool_agent.ask_finance_by_tavily(company)
    else:
        await tool_agent.ask_finance(company)


@app.command()  # type: ignore[misc]
def conversation_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
) -> None:
    """Conversation."""
    logger.debug("conversation_agent()")

    # Initialization
    registry = DependencyRegistry(tool, model)
    any_question_agent = registry.get_any_question_usecase()

    # Execute
    asyncio.run(_async_conversation_agent(any_question_agent))


async def _async_conversation_agent(any_question_agent: AnyQuestionAgent) -> None:
    """Run  for Async."""
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("GPT: bye!")
            break
        await any_question_agent.ask(user_input)


@app.command()  # type: ignore[misc]
def multi_agent(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
) -> None:
    """Multi Agent."""
    logger.debug("multi_agent()")

    # Initialization
    registry = DependencyRegistry(tool, model)
    multi_agent = registry.get_multi_agent_usecase()

    # Execute
    asyncio.run(_async_multi_agent(multi_agent))


async def _async_multi_agent(multi_agent: MultiAgent) -> None:
    """Multi Agent for Async."""
    await multi_agent.report_about_web()


@app.command()  # type: ignore[misc]
def git_indexer(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model name"),
    embedding_model: str = typer.Option(
        "text-embedding-ada-002", "--embedding-model", "-e", help="LLM embedding model name"
    ),
) -> None:
    """Github docs Indexer."""
    logger.debug("git_indexer()")

    # Initialization
    registry = DependencyRegistry(tool, model)
    github_index = registry.get_github_index_usecase(embedding_model)

    # Execute
    # FIXME:  No such file or directory: './llamaindex-cli/storage/github/docs/docstore.json'
    github_index.index("storage/github/docs")


@app.command()  # type: ignore[misc]
def local_llm(
    tool: str = typer.Option("openai", "--tool", "-t", help="LLM tool name: openai, ollama, lmstudio"),
) -> None:
    """Use Local LLM. For just example."""
    # Load documents from a directory
    documents = SimpleDirectoryReader("storage/news").load_data()
    logger.debug(f"tool: {tool}")
    if tool == "lmstudio":
        llm = LMStudio(
            model_name="llama3",
            base_url="http://localhost:1234/v1",
            temperature=0.5,
        )
        embed_model = OpenAIEmbedding(
            model_name="text-embedding",
            api_key="lm-studio",
            api_base="http://localhost:1234/v1",
        )
    elif tool == "ollama":
        llm = Ollama(
            model="llama3.2",
            # base_url="http://localhost:11434",
            temperature=0.5,
            request_timeout=60.0,
        )
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
    else:
        msg = f"Unknown LLM toolkit: {tool}"
        raise ValueError(msg)

    # Create an index from the documents
    # Note: `llm`` does not need to be set into VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    # Create a query engine
    # Note: `llm`` must be set into as_query_engine() as well.
    query_engine = index.as_query_engine(llm=llm)
    # Run a query
    response = query_engine.query("What is the main topic of these documents?")
    print(response)


@app.callback()  # type: ignore[misc]
def main(env: str = ".env") -> None:
    """First endpoint after app()."""
    logger.debug("main()")

    # load .env file
    # if local:
    #     logger.debug("Running in local mode")
    #     load_dotenv(dotenv_path=".env.dev")
    # else:
    #     load_dotenv()
    EnvSettings.set_env(env)
    logger.debug(f"env file: {env}")


if __name__ == "__main__":
    logger.info("Starting CLI app")
    app()
