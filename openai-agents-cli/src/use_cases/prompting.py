"""Prompting Pattern Use Case."""

from string import Template

from loguru import logger

from infrastructure.openai_api.client import APIMode
from infrastructure.openai_api.interface import OpenAIClientInterface
from infrastructure.repository.interface import EmbeddingRepositoryInterface


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
        elif pattern == "cot2":
            self.chain_of_thought2()
        elif pattern == "tot":
            self.tree_of_thoughts()
        elif pattern == "generated":
            self.generated_knowledge()
        elif pattern == "reflection":
            self.reflection_prompting()
        elif pattern == "meta":
            self.meta_prompting()
        elif pattern == "prompt-chaining":
            self.prompt_chaining()
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

    def chain_of_thought2(self) -> None:
        """5-2. Chain Of Thought Prompting."""
        logger.info("Chain Of Thought Prompting 2")
        instructions = "You are an experienced software engineer."
        # 1. execute normal question
        question = """
        以下のエラーメッセージの原因を特定し、解決策を提案してください。
        ```
        エラー: `TypeError: unsupported operand type(s) for +: 'int' and 'str'`
        ```
        手順:
        1. エラーの種類と発生箇所を特定
        2. 変数のデータ型を確認
        3. 型変換の必要性を判断
        4. 修正コード例を提示」
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def tree_of_thoughts(self) -> None:
        """6. Tree of Thoughts Prompting."""
        logger.info("Tree of Thoughts Prompting")
        instructions = "You are a helpful business consultant."
        # 1. 複数の提案を生成
        question = """
        新しいエコフレンドリーなカフェを立ち上げるためのアイデアをいくつか提案してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 2. 取得した提案を掘り下げる
        question = """
        頂いたアイデアのうちの１つを掘り下げてみましょう。3つの異なるアプローチとアイデアを挙げてください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 3. アイデアの評価を行う
        question = """
        それぞれのアイデアについて、実現可能性・効果・独自性の観点から評価してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 4. 更なるアイデアのブラッシュアップ
        question = """
        最も有望なアイデアについて、具体的な実施計画や必要なリソース、リスクとその対策を詳しく説明してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

        # 5. 最終決定
        question = """
        すべての評価を踏まえ、最も適切と思われるアイデアを1つ選び、その理由を述べてください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def generated_knowledge(self) -> None:
        """7. Generated Knowledge Prompting."""
        logger.info("Generated Knowledge Prompting")
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

    def reflection_prompting(self) -> None:
        """8. Reflection."""
        logger.info("Reflection Prompting")
        instructions = "あなたはソフトウェアアーキテクトです。"
        # 1. `3`番目で自己検証を行なっているのがポイント
        question = """
        PythonでWebアプリケーションを開発することを考えています。以下の手順で回答してください。
        1. 要件に基づき3つの候補フレームワークを提案
        2. 各候補について「パフォーマンス」「学習曲線」「コミュニティ規模」の観点で評価
        3. 自らの評価を批判的に検証し、潜在的な見落としを指摘
        4. 最終推奨案を選択し、その理由を説明 (反対意見への反論を含むこと)
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def meta_prompting(self) -> None:
        """9. Meta Prompting."""
        logger.info("Meta Prompting")
        instructions = "あなたは生成AIのプロンプトエンジニアです。"
        # 1. 効果的なプロンプトを生成する
        # Define the template
        template = Template("""
        以下の要素を含む最適なプロンプトを生成してください。
        1. 目的: $objective
        2. 対象読者: $audience
        3. 出力形式: $output_format
        4. 生成後、そのプロンプトの改善点を自己診断してください
        """)

        # Set dynamic parameters
        params = {
            "objective": "キャリアチェンジを考えており、AIエンジニアとしてのキャリアパスを知りたい",
            "audience": "ソフトウェアエンジニアとしての既に十分な経験がある人",
            "output_format": "時系列でのキャリアパスを示すリスト",
        }

        # Generate the question dynamically
        question = template.substitute(params)
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)

    def prompt_chaining(self) -> None:
        """10. Prompt Chaining."""
        logger.info("Prompt Chaining")
        instructions = "You are an experienced software engineer."
        # 1. 課題定義
        question = """
        1日1億リクエストを処理するECサイトのデータベースボトルネック解決策を3案提示してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)
        # 2. 詳細評価
        question = """
        1番効果的と思われる案を選び、想定されるコスト増加とパフォーマンス向上効果を定量化してください。
        """
        logger.info(f"query question: instructions: {instructions}, question: {question}")
        response = self._query(instructions, question)
        print(response)
        # 3. 実装計画
        question = """
        追加で、具体的な設計パターンを比較してください。
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
