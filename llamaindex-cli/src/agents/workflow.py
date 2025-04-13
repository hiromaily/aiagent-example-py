"""Builds the workflow for the agent."""

from llama_index.core.agent.workflow import AgentWorkflow, BaseWorkflowAgent, FunctionAgent
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


def build_financial_tavily_tool_workflow(llm: LLM, api_key: str) -> FunctionAgent:
    """Build the financial tool workflow using Tavily."""
    return _build_financial_tool_workflow(llm, build_tavily_tools(api_key))


def _build_financial_tool_workflow(llm: LLM, tools: list[BaseTool] | None) -> FunctionAgent:
    """Build the financial tool workflow."""
    return FunctionAgent(
        name="FinancialAgent",
        description="Useful for performing financial operations.",
        llm=llm,
        tools=tools,
        system_prompt="You are a helpful assistant.",
    )


def build_standard_workflow(llm: LLM) -> AgentWorkflow:
    """Build the standard workflow."""
    return AgentWorkflow.from_tools_or_functions(
        llm=llm,
        tools_or_functions=[],
        system_prompt="You are a helpful assistant.",
        initial_state={},
    )


def build_research_workflow(llm: LLM, tools: list[BaseTool]) -> FunctionAgent:
    """Build the research workflow."""
    return FunctionAgent(
        name="ResearchAgent",
        description="Useful for searching the web for information on a given topic and recording notes on the topic.",
        system_prompt=(
            "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
            "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic."
        ),
        llm=llm,
        tools=tools,
        # tools=[search_web, record_notes],
        can_handoff_to=["WriteAgent"],
    )


def build_write_workflow(llm: LLM, tools: list[BaseTool]) -> FunctionAgent:
    """Build the write workflow."""
    return FunctionAgent(
        name="WriteAgent",
        description="Useful for writing a report on a given topic.",
        system_prompt=(
            "You are the WriteAgent that can write a report on a given topic. "
            "Your report should be in a markdown format. The content should be grounded in the research notes. "
            "Once the report is written, you should get feedback at least once from the ReviewAgent."
        ),
        llm=llm,
        tools=tools,
        # tools=[write_report],
        can_handoff_to=["ReviewAgent", "ResearchAgent"],
    )


def build_review_workflow(llm: LLM, tools: list[BaseTool]) -> FunctionAgent:
    """Build the write workflow."""
    return FunctionAgent(
        name="ReviewAgent",
        description="Useful for reviewing a report and providing feedback.",
        system_prompt=(
            "You are the ReviewAgent that can review a report and provide feedback. "
            "Your feedback should either approve the current report or request changes for the WriteAgent to implement."
        ),
        llm=llm,
        tools=tools,
        # tools=[review_report],
        can_handoff_to=["WriteAgent"],
    )


def build_multi_workflow(agents: list[BaseWorkflowAgent]) -> AgentWorkflow:
    """Build the multi workflow."""
    return AgentWorkflow(
        agents=agents,
        # agents=[research_agent, write_agent, review_agent],
        root_agent=agents[0].name,
        # root_agent=research_agent.name,
        initial_state={
            "research_notes": {},
            "report_content": "Not written yet.",
            "review": "Review required.",
        },
    )
