"""Dataset entry models for the SynthGenAI package."""

from typing import List, Literal

from pydantic import BaseModel

from synthgenai.schemas.messages import (
    InstructMessage,
    PreferenceChosen,
    PreferencePrompt,
    PreferenceRejected,
)


class EntryKeywords(BaseModel):
    """Pydantic model for the keywords in the generated text."""

    keywords: List[str]


class EntryLabels(BaseModel):
    """Pydantic model for the labels in the generated text."""

    labels: List[str]


class EntryRawDataset(BaseModel):
    """Pydantic model for raw dataset entry."""

    keyword: str
    topic: str
    language: str
    text: str


class EntryInstructionDataset(BaseModel):
    """Pydantic model for instruction dataset entry."""

    keyword: str
    topic: str
    language: str
    messages: List[InstructMessage]


class EntryPreferenceDataset(BaseModel):
    """Pydantic model for preference dataset entry."""

    keyword: str
    topic: str
    language: str
    prompt: List[PreferencePrompt]
    chosen: List[PreferenceChosen]
    rejected: List[PreferenceRejected]


class EntrySummaryDataset(BaseModel):
    """Pydantic model for summary dataset entry."""

    keyword: str
    topic: str
    language: str
    text: str
    summary: str


class EntrySentimentDataset(BaseModel):
    """Pydantic model for sentiment analysis dataset entry."""

    keyword: str
    topic: str
    language: str
    prompt: str
    label: Literal["positive", "negative", "neutral"]


class EntryTextClassificationDataset(BaseModel):
    """Pydantic model for text classification dataset entry."""

    keyword: str
    topic: str
    language: str
    prompt: str
    label: str
