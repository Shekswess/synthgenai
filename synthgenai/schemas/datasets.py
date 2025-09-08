"""Dataset entry models for the SynthGenAI package."""

from typing import Union

from pydantic import BaseModel

from synthgenai.schemas.generated import (
    GeneratedInstructText,
    GeneratedPreferenceText,
    GeneratedSentimentAnalysis,
    GeneratedSummaryText,
    GeneratedText,
    GeneratedTextClassification,
)


class EntryKeywords(BaseModel):
    """Pydantic model for the keywords in the generated text."""

    keywords: list[str]


class EntryLabels(BaseModel):
    """Pydantic model for the labels in the generated text."""

    labels: list[str]


class EntryDataset(BaseModel):
    """Pydantic model for the dataset entry."""

    keyword: str
    topic: str
    language: str
    generated_entry: Union[
        GeneratedText,
        GeneratedInstructText,
        GeneratedPreferenceText,
        GeneratedSummaryText,
        GeneratedSentimentAnalysis,
        GeneratedTextClassification,
    ]
