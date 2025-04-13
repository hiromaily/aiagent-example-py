"""Multi Agent Use Case."""

from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    ToolCall,
    ToolCallResult,
)


class MultiAgent:
    """Multi Agent Use Case."""

    def __init__(self, agent: AgentWorkflow) -> None:
        """Initialize the MultiAgent with a LLM."""
        self._agent_workflow = agent

    async def run(self) -> None:
        """Ask the question by agent()."""
        handler = self._agent_workflow.run(
            user_msg="""
            Write me a report on the history of the web. Briefly describe the history
            of the world wide web, including the development of the internet and the
            development of the web, including 21st century developments.
        """
        )

        current_agent = None
        # current_tool_calls = ""
        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"\n{'=' * 50}")
                print(f"🤖 Agent: {current_agent}")
                print(f"{'=' * 50}\n")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print("📤 Output:", event.response.content)
                if event.tool_calls:
                    print(
                        "🛠️  Planning to use tools:",
                        [call.tool_name for call in event.tool_calls],
                    )
            elif isinstance(event, ToolCallResult):
                print(f"🔧 Tool Result ({event.tool_name}):")
                print(f"  Arguments: {event.tool_kwargs}")
                print(f"  Output: {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"🔨 Calling Tool: {event.tool_name}")
                print(f"  With arguments: {event.tool_kwargs}")
