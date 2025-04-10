"""Tech Question Agent Use Case."""

from llama_index.core.agent.workflow import AgentStream, ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context


class ReActAgentWrapper:
    """ReActAgent wrapper."""

    # tools is required for ReActAgent
    def __init__(self, llm: LLM, tools: list[BaseTool]) -> None:
        """Initialize the ReActAgentWrapper with a LLM."""
        self._agent = ReActAgent(
            name="ReActAgent",
            description="Agent that can perform basic conversation.",
            llm=llm,
            tools=tools,
        )
        self._ctx = Context(self._agent)

    async def run(self, question: str) -> None:
        """Run ReActAgentWrapper agent."""
        handler = self._agent.run(question, ctx=self._ctx)
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)

        response = await handler
        print(str(response))
