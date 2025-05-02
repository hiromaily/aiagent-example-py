"""OpenAI API module class."""

from enum import Enum
from typing import cast

from openai import OpenAI
from openai.types.embedding import Embedding

from infrastructure.web_browser.interface import WebClientInterface

from .interface import OpenAIClientInterface


class APIMode(Enum):
    """API Mode."""

    RESPONSE_API = 1
    CHAT_COMPLETION_API = 2


class OpenAIClient(OpenAIClientInterface, WebClientInterface):
    """OpenAI API Client class."""

    def __init__(
        self,
        model: str,
        api_key: str,
        embedding_model: str = "text-embedding-ada-002",
        base_url: str | None = None,
        is_local_llm: bool = False,
    ) -> None:
        """Initialize OpenAI client."""
        if not model:
            msg = "Model must be provided"
            raise ValueError(msg)

        if not api_key:
            msg = "API key must be provided"
            raise ValueError(msg)

        self._model = model
        self._embedding_model = embedding_model
        self._is_local_llm = is_local_llm
        if not base_url:
            self._client = OpenAI(api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._message_histories: list[dict[str, str]] = []  # message history
        self._previous_response_id: str | None = None  # message history

    def clear(self) -> None:
        """Clear message history."""
        self._message_histories = []
        self._previous_response_id = None

    def call_response(self, instructions: str, prompt: str) -> str:
        """Call Response API."""
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=prompt,
            previous_response_id=self._previous_response_id,
        )
        # save response id
        self._previous_response_id = response.id
        return cast("str", response.output_text)

    def call_chat_completion(self, instructions: str, prompt: str) -> str:
        """Call Chat Completion API."""
        if len(self._message_histories) > 0:
            # add message history to the prompt
            self._message_histories.append({"role": "user", "content": prompt})
        else:
            # create new message
            system_role = "system" if self._is_local_llm else "developer"
            self._message_histories.append({"role": system_role, "content": instructions})
            self._message_histories.append({"role": "user", "content": prompt})

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self._message_histories,
        )
        # save message history
        self._message_histories.append({"role": "assistant", "content": completion.choices[0].message.content})
        # return completion.choices[0].message.content
        return cast("str", completion.choices[0].message.content)

    def call_embeddings(self, prompt: str | list[str]) -> list[Embedding]:
        """Call Embeddings API."""
        response = self._client.embeddings.create(model=self._embedding_model, input=prompt, encoding_format="float")
        # return response.data
        return cast("list[Embedding]", response.data)

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
        # return completion.choices[0].message.content
        return cast("str", completion.choices[0].message.content)
