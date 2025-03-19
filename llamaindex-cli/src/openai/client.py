"""OpenAI module class."""

from openai import OpenAI


class OpenAIClient:
    """OpenAI Client class."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize OpenAI client."""
        if not api_key:
            msg = "API key must be provided"
            raise ValueError(msg)
        if not model:
            msg = "Model must be provided"
            raise ValueError(msg)

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call_openai_api(self, instructions: str, prompt: str) -> str:
        """Method to interact with OpenAI API directly."""
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text
