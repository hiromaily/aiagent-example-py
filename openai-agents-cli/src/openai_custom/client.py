"""OpenAI module class."""

from openai import OpenAI
from openai.types.embedding import Embedding

from .interface import OpenAIClientInterface


class OpenAIClient(OpenAIClientInterface):
    """OpenAI API Client class."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None, is_local_llm: bool = False) -> None:
        """Initialize OpenAI client."""
        if not model:
            msg = "Model must be provided"
            raise ValueError(msg)

        if not api_key:
            msg = "API key must be provided"
            raise ValueError(msg)

        self._model = model
        self._is_local_llm = is_local_llm
        if not base_url:
            self._client = OpenAI(api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def call_response(self, instructions: str, prompt: str) -> str:
        """Call Response API."""
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text

    def call_chat_completion(self, instructions: str, prompt: str) -> str:
        """Call Chat Completion API."""
        # developer or system
        system_role = "system" if self._is_local_llm else "developer"
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": system_role, "content": instructions},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return completion.choices[0].message.content

    def call_embeddings(self, prompt: str | list[str]) -> list[Embedding]:
        """Call Embeddings API."""
        response = self._client.embeddings.create(model="text-embedding-ada-002", input=prompt, encoding_format="float")
        # return response.data[0].embedding
        return response.data

    def call_web_search(self, prompt: str) -> str:
        """Call Web Search API."""
        completion = self._client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={
                "search_context_size": "low",
                "user_location": {
                    "type": "approximate",
                    "approximate": {
                        "country": "JP",
                        "city": "Tokyo",
                        "region": "Tokyo",
                    },
                },
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return completion.choices[0].message.content
