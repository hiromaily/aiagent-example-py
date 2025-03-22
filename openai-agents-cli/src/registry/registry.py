"""Registry Class."""

import os

from openai_custom.client import OpenAIClient
from openai_custom.dymmy import OpenAIDummyClient
from openai_custom.interface import OpenAIClientInterface


class DependencyRegistry:
    """Dependency Registry."""

    def __init__(self, environment: str) -> None:
        """Initialize the DependencyRegistry with the environment."""
        self.environment = environment
        self.openai_client = self._build_openai_client()

    def _build_openai_client(self) -> OpenAIClientInterface:
        """Build the OpenAI client based on the environment."""
        if self.environment == "prod":
            # use OpenAI API
            api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL")
            openai_client = OpenAIClient(openai_model, api_key)
        elif self.environment == "dev":
            # use local LLM
            openai_model = os.getenv("OPENAI_MODEL")
            server_url = os.getenv("OPENAI_SERVER_URL")
            if server_url is None:
                msg = "`OPENAI_SERVER_URL` must be provided"
                raise ValueError(msg)

            openai_client = OpenAIClient(openai_model, None, server_url)
        elif self.environment == "test":
            openai_client = OpenAIDummyClient()
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        return openai_client

    def get_openai_client(self) -> OpenAIClientInterface:
        """Get the OpenAI client."""
        return self.openai_client
