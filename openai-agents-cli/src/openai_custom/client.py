"""OpenAI module class."""

from openai import OpenAI

from openai_custom.interface import OpenAIClientInterface


class OpenAIClient(OpenAIClientInterface):
    """OpenAI API Client class."""

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

    def call_response(self, instructions: str, prompt: str) -> str:
        """Call Response API."""
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text

    def call_web_search(self, prompt: str) -> str:
        """Call Web Search API."""
        completion = self.client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={},
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return completion.choices[0].message.content
