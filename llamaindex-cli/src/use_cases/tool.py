"""Query Image Agent Use Case."""

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.llms import LLM


class ToolAgent:
    """Tool Agent Use Case."""

    def __init__(
        self, llm: LLM, calc_workflow: AgentWorkflow, financial_workflow: AgentWorkflow, tavily_workflow: AgentWorkflow
    ) -> None:
        """Initialize the QueryImageAgent with a LLM."""
        self._llm = llm
        self._calc_workflow = calc_workflow
        self._financial_workflow = financial_workflow
        self._tavily_workflow = tavily_workflow

    async def ask_calc(self, question: str) -> None:
        """Ask the calculation()."""
        response = await self._calc_workflow.run(user_msg=question)
        print(response)

    async def ask_finance(self, company: str) -> None:
        """Ask the calculation()."""
        question = f"What's the current stock price of {company}"
        response = await self._financial_workflow.run(user_msg=question)
        print(response)

    async def ask_finance_by_tavily(self, company: str) -> None:
        """Ask the calculation()."""
        question = f"What's the current stock price of {company}"
        response = await self._tavily_workflow.run(user_msg=question)
        print(response)
