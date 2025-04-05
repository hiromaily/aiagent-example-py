"""Query Image Agent Use Case."""

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.llms import LLM


class ToolAgent:
    """Tool Agent Use Case."""

    def __init__(self, llm: LLM, calc_workflow: AgentWorkflow, financial_workflow: AgentWorkflow) -> None:
        """Initialize the QueryImageAgent with a LLM."""
        self._llm = llm
        self._calc_workflow = calc_workflow
        self._financial_workflow = financial_workflow

    async def ask_calc(self, question: str) -> None:
        """Ask the calculation()."""
        response = await self._calc_workflow.run(user_msg=question)
        print(response)

    async def ask_finance(self, company: str) -> None:
        """Ask the calculation()."""
        question = f"Get the stock price of {company}"
        response = await self._calc_workflow.run(user_msg=question)
        print(response)
