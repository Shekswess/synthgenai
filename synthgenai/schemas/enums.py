"""Enumerations for the SynthGenAI package."""

from enum import Enum


class DatasetType(str, Enum):
    """Enum for the dataset types."""

    RAW = "Raw Dataset"
    INSTRUCTION = "Instruction Dataset"
    PREFERENCE = "Preference Dataset"
    SUMMARIZATION = "Summarization Dataset"
    SENTIMENT_ANALYSIS = "Sentiment Analysis Dataset"
    TEXT_CLASSIFICATION = "Text Classification Dataset"
