"""Tech Question Agent Use Case."""

from collections.abc import Sequence

from llama_index.core.agent.workflow import AgentStream, ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context


class ReActAgentWrapper:
    """ReActAgent wrapper."""

    # tools is required for ReActAgent
    # Sequence is part of list
    def __init__(self, llm: LLM, tools: Sequence[BaseTool]) -> None:
        """Initialize the ReActAgentWrapper with a LLM."""
        self._agent = ReActAgent(
            name="ReActAgent",
            description="Agent that can perform basic conversation.",
            llm=llm,
            tools=tools,  # type: ignore[arg-type]
        )
        self._ctx = Context(self._agent)  # type: ignore[arg-type]

    async def run(self, question: str) -> None:
        """Run ReActAgentWrapper agent."""
        handler = self._agent.run(question, ctx=self._ctx)
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)

        response = await handler
        print(str(response))
