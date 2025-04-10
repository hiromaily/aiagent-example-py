"""AI tools for performing basic arithmetic operations."""

from typing import cast

from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product."""
    return a * b


def build_financial_tools() -> list[BaseTool] | None:
    """Build the financial tools."""
    finance_tools = YahooFinanceToolSpec().to_tool_list()
    finance_tools.extend([multiply, add])
    return cast("list[FunctionTool]", finance_tools)


# Ref: https://docs.tavily.com/documentation/integrations/llamaindex
def build_tavily_tools(api_key: str) -> list[BaseTool] | None:
    """Build the Tavily tools."""
    if not api_key:
        msg = "API key is required for Tavily tools."
        raise ValueError(msg)

    return cast("list[FunctionTool]", TavilyToolSpec(api_key=api_key).to_tool_list())
