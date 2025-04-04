"""Tech Question Agent Use Case."""

# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import LLM, ChatMessage


class TechQuestionAgent:
    """Tech Question Agent Use Case."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the TechQuestionAgent with a LLM."""
        self._llm = llm

    def ask(self, question: str) -> None:
        """Ask the question by complete()."""
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {question}
        """

        response = self._llm.complete(prompt)
        print(response)

    def ask_stream(self, question: str) -> None:
        """Ask the question using stream."""
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {question}
        """

        handle = self._llm.stream_complete(prompt)
        for token in handle:
            print(token.delta, end="", flush=True)

    def ask_by_chat(self, question: str) -> None:
        """Ask the question by chat()."""
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {question}
        """

        messages = [
            ChatMessage(role="system", content="You are an experienced software engineer."),
            ChatMessage(role="user", content=prompt),
        ]
        response = self._llm.chat(messages)
        print(response)
