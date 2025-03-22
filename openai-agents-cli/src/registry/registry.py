"""Registry Class."""

import os

from openai_custom.client import OpenAIClient
from openai_custom.dymmy import OpenAIDummyClient
from openai_custom.interface import OpenAIClientInterface


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, environment: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self._environment = environment
        self._openai_client = self._build_openai_client()

    def _build_openai_client(self) -> OpenAIClientInterface:
        """Build the OpenAI client based on the environment."""
        if self._environment == "prod":
            # use OpenAI API
            openai_model = os.getenv("OPENAI_MODEL")
            api_key = os.getenv("OPENAI_API_KEY")
            is_local_llm = False
            openai_client = OpenAIClient(model=openai_model, api_key=api_key, is_local_llm=is_local_llm)
        elif self._environment == "dev":
            # use local LLM
            openai_model = os.getenv("OPENAI_MODEL")
            api_key = os.getenv("OPENAI_API_KEY")
            server_url = os.getenv("OPENAI_SERVER_URL")
            if server_url is None:
                msg = "`OPENAI_SERVER_URL` must be provided"
                raise ValueError(msg)

            is_local_llm = True
            openai_client = OpenAIClient(
                model=openai_model, api_key=api_key, base_url=server_url, is_local_llm=is_local_llm
            )
        elif self.environment == "test":
            openai_client = OpenAIDummyClient()
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        return openai_client

    def get_openai_client(self) -> OpenAIClientInterface:
        """Get the OpenAI client."""
        return self._openai_client
