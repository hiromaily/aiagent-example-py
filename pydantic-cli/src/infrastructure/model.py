"""This module contains the functions to get the OpenAI model."""

import os
import sys

from pydantic_ai.models.openai import OpenAIModel


def get_openai_model() -> OpenAIModel:
    """Get the OpenAI model."""
    openai_model = os.getenv("OPENAI_MODEL")
    if not openai_model:
        print("Error: OPENAI_MODEL is not set or is empty")
        sys.exit(1)
    return OpenAIModel(openai_model)
