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
        elif pattern == "emotion":
            self.emotion_prompting()
        elif pattern == "cot":
            self.chain_of_thought()
        elif pattern == "tot":
            self.tree_of_thoughts()
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
        # 1. execute without instructions
        logger.info(f"query question: instructions: None, question: {question}")
        response = self._query("", question)
        print(response)

        # 2. execute with instructions
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

        Q: "商品が届かないのですが、どうなっていますか?"
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def emotion_prompting(self) -> None:
        """4. Emotion Prompting."""
        logger.info("Emotion Prompting")
        instructions = "あなたは私の良き理解者です。ポジティブに応援しましょう。"
        question = """
        AIによって社会が大きく変わろうとしています。この状況でもソフトウェアエンジニアとして働き続けることができるのでしょうか?。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def chain_of_thought(self) -> None:
        """5. Chain Of Thought Prompting. + Self-Consistency."""
        logger.info("Chain Of Thought Prompting")
        instructions = "You are a helpful assistant."
        # 1. execute normal question
        question = """
        Please count the number of characters in the word `Hallucinations`.
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 2. execute with step-by-step
        question = """
        Please count the number of characters in the word `Hallucinations`. Think it step-by-step.
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 3. Self-Consistency
        logger.info("ask again for self-consistency")
        response = self._query(instructions, question)
        print(response)

    def tree_of_thoughts(self) -> None:
        """6. Tree of Thoughts Prompting."""
        logger.info("Tree of Thoughts Prompting")
        instructions = "You are a helpful business consultant."
        # 1. execute main topic
        question = """
        新しいエコフレンドリーなカフェを立ち上げるためのアイデアをいくつか提案してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 2. execute idea creation
        question = """
        頂いたアイデアのうちの１つを掘り下げてみましょう。3つの異なるアプローチとアイデアを挙げてください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 3. execute idea evaluation
        question = """
        それぞれのアイデアについて、実現可能性・効果・独自性の観点から評価してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 4. execute research deeply
        question = """
        最も有望なアイデアについて、具体的な実施計画や必要なリソース、リスクとその対策を詳しく説明してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 5. execute decision making
        question = """
        すべての評価を踏まえ、最も適切と思われるアイデアを1つ選び、その理由を述べてください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def generated_knowledge(self) -> None:
        """7. Generated Knowledge Prompting."""
        instructions = "You are good at geography."
        # 1. execute question 1 for knowledge
        question = """
        東京都の面積を教えて。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 2. execute question 2 for knowledge
        question = """
        大阪府の面積を教えて。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 3. execute final question from generated knowledge
        question = """
        以上のことから、東京都と大阪府ではどちらが広いですか?
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
