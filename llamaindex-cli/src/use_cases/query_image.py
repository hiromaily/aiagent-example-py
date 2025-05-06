"""Query Image Agent Use Case."""

from llama_index.core.llms import LLM, ChatMessage, ImageBlock, TextBlock
from loguru import logger


class QueryImageAgent:
    """Query Images Agent Use Case."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the QueryImageAgent with a LLM."""
        self._llm = llm

    def ask(self, image_path: str) -> None:
        """Ask the question by chat()."""
        logger.debug(f"image_path: {image_path}")

        messages = [
            ChatMessage(
                role="user",
                blocks=[
                    ImageBlock(path=image_path),  # type: ignore[arg-type]
                    TextBlock(text="Describe the image in a few sentences."),
                ],
            )
        ]
        response = self._llm.chat(messages)
        print(response)
