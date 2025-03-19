"""OpenAI module class."""

from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key, model):
        if not api_key:
            raise ValueError("API key must be provided")
        if not model:
            raise ValueError("Model name must be provided")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call_openai_api(self, instructions, prompt):
        """Method to interact with OpenAI API directly."""
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text
