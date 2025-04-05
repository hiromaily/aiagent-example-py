"""AI tools for performing basic arithmetic operations."""

from collections.abc import Callable

from llama_index.core.tools import BaseTool
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product."""
    return a * b


def build_financial_tools() -> list[BaseTool | Callable] | None:
    """Build the financial tools."""
    finance_tools = YahooFinanceToolSpec().to_tool_list()
    finance_tools.extend([multiply, add])
    return finance_tools
