"""Pydantic schemas for the SynthGenAI package.

This module provides a modular organization of Pydantic models used throughout
the SynthGenAI package, divided into logical categories:

- enums: Enumerations and constants
- config: Configuration models for LLM and dataset settings
- messages: Message models for various chat and prompt formats
- generated: Models for generated content across different dataset types
- datasets: Dataset entry models and related structures
"""


from .config import DatasetConfig, DatasetGeneratorConfig, LLMConfig
from .datasets import EntryDataset, EntryKeywords, EntryLabels
from .enums import DatasetType
from .generated import (
    GeneratedInstructText,
    GeneratedPreferenceText,
    GeneratedSentimentAnalysis,
    GeneratedSummaryText,
    GeneratedText,
    GeneratedTextClassification,
)
from .messages import (
    InputMessage,
    InstructMessage,
    PreferenceChosen,
    PreferencePrompt,
    PreferenceRejected,
)

__all__ = [
    "DatasetType",
    "LLMConfig",
    "DatasetConfig",
    "DatasetGeneratorConfig",
    "InputMessage",
    "InstructMessage",
    "PreferencePrompt",
    "PreferenceChosen",
    "PreferenceRejected",
    "GeneratedText",
    "GeneratedInstructText",
    "GeneratedPreferenceText",
    "GeneratedSummaryText",
    "GeneratedSentimentAnalysis",
    "GeneratedTextClassification",
    "EntryKeywords",
    "EntryLabels",
    "EntryDataset",
]
