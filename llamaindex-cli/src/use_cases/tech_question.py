"""Tech Question Agent Use Case."""

# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import LLM


class TechQuestionAgent:
    """Tech Question Agent Use Case."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the TechQuestionAgent with a LLM."""
        self._llm = llm

    def ask(self, question: str) -> None:
        """Ask the question."""
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {question}
        """

        response = self._llm.complete(prompt)
        print(response)
