"""Any Question Agent Use Case."""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context


class AnyQuestionAgent:
    """Any Question Agent Use Case."""

    def __init__(self, agent: FunctionAgent) -> None:
        """Initialize the AnyQuestionAgent with a LLM."""
        self._agent_work_flow = agent
        self._ctx = Context(self._agent_work_flow)

    async def ask(self, question: str) -> None:
        """Ask the question by agent()."""
        await self._agent_work_flow.run(question)
        response = await self._agent_work_flow.run(user_msg=question, ctx=self._ctx)
        print(response)
