"""AI tools for performing basic arithmetic operations."""

from typing import cast

from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Context
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
def build_tavily_tools(api_key: str) -> list[BaseTool]:
    """Build the Tavily tools."""
    # if not api_key:
    #     msg = "API key is required for Tavily tools."
    #     raise ValueError(msg)

    return cast("list[FunctionTool]", TavilyToolSpec(api_key=api_key).to_tool_list())


def get_search_web(api_key: str) -> BaseTool:
    """Get search web from the Tavily tools."""
    # return build_tavily_tools(api_key)[0]
    return cast("FunctionTool", build_tavily_tools(api_key)[0])


async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.set("state", current_state)
    return "Notes recorded."


async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.get("state")
    current_state["report_content"] = report_content
    await ctx.set("state", current_state)
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Report reviewed."
