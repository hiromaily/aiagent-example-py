"""Query Agent Use Case."""

from loguru import logger

from infrastructure.openai_api.client import APIMode
from infrastructure.openai_api.interface import OpenAIClientInterface
from infrastructure.repository.interface import EmbeddingRepositoryInterface


class QueryAgent:
    """Query Agent Use Case."""

    def __init__(
        self,
        openai_client: OpenAIClientInterface,
        embedding_repo: EmbeddingRepositoryInterface,
        tool: str,
        api_mode: APIMode,
    ) -> None:
        """Initialize the QueryAgent with an OpenAI client."""
        self._openai_client = openai_client
        self._embedding_repo = embedding_repo
        self._tool = tool
        self._api_mode = api_mode

    def query_tech_guide(self, user_query: str) -> None:
        """Query the agent with a user tech question.

        This agent is designed to answer technical questions and provide step-by-step guidance
        on how to learn a specific technology.
        """
        # execute
        logger.debug(f"query question: question: {user_query}")
        response = self._query_tech_guide(user_query)
        print(response)

        # call embeddings if tool is OpenAI
        # if self._tool != "openai":
        #     logger.info("not implemented yet")
        #     return

        # call embeddings API
        logger.debug("call embedding()")
        embedding_list = self._openai_client.call_embeddings(user_query)
        print(embedding_list)

        # Insert into DB
        logger.debug("insert into db `embeddings` table")
        self._embedding_repo.insert_embeddings(embedding_list)
        self._embedding_repo.close()

    def query_common(self, user_query: str) -> None:
        """Query the agent with a user common question."""
        # execute
        response = self._query(user_query)
        print(response)

        # call embeddings
        # if self._tool != "openai":
        #     logger.info("not implemented yet")
        #     return

        # call embeddings API
        logger.debug("call agent.embedding()")
        embedding_list = self._openai_client.call_embeddings(user_query)
        print(embedding_list)
        # Insert into DB
        logger.debug("insert into db `item_contents` table")
        self._embedding_repo.insert_item_contents([user_query], embedding_list)
        self._embedding_repo.close()

    def query_news(self) -> str:
        """Query about news using Web Search."""
        # Initial prompt
        prompt = "What was a positive news story from today?"

        return self._openai_client.call_web_search(prompt)

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------

    def _query_tech_guide(self, user_query: str) -> str:
        """Query the agent with a user tech question."""
        # Initial prompt
        instructions = "You are an experienced software engineer."
        prompt = f"""
        Please provide the following information to answer the user's question about the technology:
        1. Overview of the technology
        2. Step-by-step guidance to learn the technology
        User's question about the technology: {user_query}
        """

        if self._api_mode == APIMode.RESPONSE_API:
            return self._openai_client.call_response(instructions, prompt)
        if self._api_mode == APIMode.CHAT_COMPLETION_API:
            return self._openai_client.call_chat_completion(instructions, prompt)
        msg = "Unknown API mode"
        raise ValueError(msg)

    def _query(self, user_query: str) -> str:
        """Query the agent with a user common question."""
        # Initial prompt
        instructions = "You are a helpful assistant."

        if self._api_mode == APIMode.RESPONSE_API:
            return self._openai_client.call_response(instructions, user_query)
        if self._api_mode == APIMode.CHAT_COMPLETION_API:
            return self._openai_client.call_chat_completion(instructions, user_query)
        msg = "Unknown API mode"
        raise ValueError(msg)
