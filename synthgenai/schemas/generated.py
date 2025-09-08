"""Generated content models for the SynthGenAI package."""

from typing import Literal

from pydantic import BaseModel

from synthgenai.schemas.messages import (
    InstructMessage,
    PreferenceChosen,
    PreferencePrompt,
    PreferenceRejected,
)


class GeneratedText(BaseModel):
    """Pydantic model for the generated text."""

    text: str


class GeneratedInstructText(BaseModel):
    """Pydantic model for the generated text in the Instruct dataset."""

    messages: list[InstructMessage]


class GeneratedPreferenceText(BaseModel):
    """Pydantic model for the generated text in the Preference dataset."""

    prompt: list[PreferencePrompt]
    chosen: list[PreferenceChosen]
    rejected: list[PreferenceRejected]


class GeneratedSummaryText(BaseModel):
    """Pydantic model for the generated summary text."""

    text: str
    summary: str


class GeneratedSentimentAnalysis(BaseModel):
    """Pydantic model for the generated sentiment analysis."""

    prompt: str
    label: Literal["positive", "negative", "neutral"]


class GeneratedTextClassification(BaseModel):
    """Pydantic model for the generated text classification."""

    prompt: str
    label: str
