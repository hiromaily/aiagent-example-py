"""Builds the workflow for the agent."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from agents.ai_tools import add, build_financial_tools, multiply


def build_mathematical_tool_workflow(llm: LLM) -> FunctionAgent:
    """Build the mathematical tool workflow."""
    return FunctionAgent(
        name="mathematical tools",
        description="Agent that can perform basic mathematical operations.",
        tools=[multiply, add],
        llm=llm,
        system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    )


def build_financial_tool_workflow(llm: LLM) -> FunctionAgent:
    """Build the financial tool workflow."""
    return FunctionAgent(
        name="Agent",
        description="Useful for performing financial operations.",
        llm=llm,
        tools=build_financial_tools(),
        system_prompt="You are a helpful assistant.",
    )
