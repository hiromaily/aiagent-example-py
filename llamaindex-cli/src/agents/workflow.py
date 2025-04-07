"""Builds the workflow for the agent."""

import os
from collections.abc import Callable

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool

from agents.ai_tools import add, build_financial_tools, build_tavily_tools, multiply


def build_mathematical_tool_workflow(llm: LLM) -> FunctionAgent:
    """Build the mathematical tool workflow."""
    return FunctionAgent(
        name="mathematical tools",
        description="Agent that can perform basic mathematical operations.",
        tools=[multiply, add],
        llm=llm,
        system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    )


def build_financial_yahoo_financial_tool_workflow(llm: LLM) -> FunctionAgent:
    """Build the financial tool workflow using YahooFinanceToolSpec."""
    return _build_financial_tool_workflow(llm, build_financial_tools())


def build_financial_tavily_tool_workflow(llm: LLM) -> FunctionAgent:
    """Build the financial tool workflow using Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        msg = "API key is required for Tavily tools."
        raise ValueError(msg)

    return _build_financial_tool_workflow(llm, build_tavily_tools(api_key))


def _build_financial_tool_workflow(llm: LLM, tools: list[BaseTool | Callable] | None) -> FunctionAgent:
    """Build the financial tool workflow."""
    return FunctionAgent(
        name="FinancialAgent",
        description="Useful for performing financial operations.",
        llm=llm,
        tools=tools,
        system_prompt="You are a helpful assistant.",
    )
