"""This module contains the response model for the CLI application."""

from pydantic import BaseModel, Field


class ResponseModel(BaseModel):
    """Response model."""

    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")
