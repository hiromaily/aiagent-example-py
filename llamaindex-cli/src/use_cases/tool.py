"""Query Image Agent Use Case."""

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.llms import LLM


class ToolAgent:
    """Tool Agent Use Case."""

    def __init__(self, llm: LLM, workflow: AgentWorkflow) -> None:
        """Initialize the QueryImageAgent with a LLM."""
        self._llm = llm
        self._workflow = workflow

    async def ask_calc(self) -> None:
        """Ask the calculation()."""
        response = await self._workflow.run(user_msg="What is 20+(2*4)?")
        print(response)
