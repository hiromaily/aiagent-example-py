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
        if self.environment == "production":
            api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL")
            openai_client = OpenAIClient(api_key, openai_model)
        elif self.environment == "development":
            openai_client = OpenAIDummyClient()
        else:
            msg = "Unknown environment"
            raise ValueError(msg)

        return openai_client

    def get_openai_client(self) -> OpenAIClientInterface:
        """Get the OpenAI client."""
        return self.openai_client
