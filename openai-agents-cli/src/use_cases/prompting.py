"""Prompting Pattern Use Case."""

from loguru import logger

from infrastructure.repository.interface import EmbeddingRepositoryInterface
from openai_custom.client import APIMode
from openai_custom.interface import OpenAIClientInterface


class PromptingPatternAgent:
    """Prompting Pattern Agent Use Case."""

    def __init__(
        self,
        openai_client: OpenAIClientInterface,
        embedding_repo: EmbeddingRepositoryInterface,
        tool: str,
        api_mode: APIMode,
    ) -> None:
        """Initialize the PromptingPatternAgent with an OpenAI client."""
        self._openai_client = openai_client
        self._embedding_repo = embedding_repo
        self._tool = tool
        self._api_mode = api_mode

    def call(self, pattern: str) -> None:
        """Endpoint with pattern."""
        if pattern == "zero-shot":
            self.zero_shot()
        elif pattern == "few-shot":
            self.few_shot()
        elif pattern == "roll":
            self.roll_prompting()
        else:
            msg = f"Unknown pattern: {pattern}"
            raise ValueError(msg)

    def zero_shot(self) -> None:
        """1. ZeroShot Prompting."""
        logger.info("ZeroShot Prompting")
        question = "What are the top 10 Python libraries for AI?"
        # execute without instructions
        logger.info(f"query question: instructions: None, question: {question}")
        response = self._query("", question)
        print(response)

        # execute with instructions
        instructions = "You are an experienced software engineer."
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)


    def few_shot(self) -> None:
        """2. FewShot Prompting."""
        logger.info("FewShot Prompting")
        question = """
        Answer Positive or Negative to the following question. I give a few examples.
        Q: "I love this product! It's amazing."
        A: Positive

        Q: "This is the worst experience I've ever had."
        A: Negative.

        Q: "I finally achieve my goal!"
        """
        # execute without instructions
        logger.info(f"query question: instructions: None, question: {question}")
        response = self._query("", question)
        print(response)

        # execute with instructions
        instructions = "You are a helpful assistant."
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def roll_prompting(self) -> None:
        """3. Roll Prompting."""
        logger.info("Roll Prompting")
        instructions = "あなたはショッピングサイトのカスタマーサポート担当者です。"
        question = """
        ショッピングサイトのカスタマーサポートとしてお客様からの問い合わせに対して、適切な回答をしてください。
        以下の問い合わせを頂きました。

        Q: "商品が届かないのですが、どうなっていますか？"        
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)


    def _query(self, instructions: str, prompt: str) -> str:
        if self._api_mode == APIMode.RESPONSE_API:
            return self._openai_client.call_response(instructions, prompt)
        if self._api_mode == APIMode.CHAT_COMPLETION_API:
            return self._openai_client.call_chat_completion(instructions, prompt)
        msg = "Unknown API mode"
        raise ValueError(msg)
