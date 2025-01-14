"""Pydantic models for the SynthGenAI package."""

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field, AnyUrl


class DatasetType(str, Enum):
    """Enum for the dataset types."""

    RAW = "Raw Dataset"
    INSTRUCTION = "Instruction Dataset"
    PREFERENCE = "Preference Dataset"
    SUMMARIZATION = "Summarization Dataset"
    SENTIMENT_ANALYSIS = "Sentiment Analysis Dataset"


class LLMConfig(BaseModel):
    """
    Pydantic model for the LLM configuration.

    Attributes:
        model (str): The model name of the LLM.
        temperature (float): The temperature value from 0.0 to 1.0.
        top_p (float): The top_p value from 0.0 to 1.0.
        max_tokens (int): The maximum number of tokens to generate completions from 1000 to max value.
        api_base (AnyUrl): The API base URL.
        api_key (str): The API key.
    """

    model: str = Field(..., min_length=1)
    temperature: float = Field(None, ge=0.0, le=1.0)
    top_p: float = Field(None, ge=0.0, le=1.0)
    max_tokens: int = Field(None, gt=1000)
    api_base: AnyUrl = Field(None)
    api_key: str = Field(None)


class DatasetConfig(BaseModel):
    """
    Pydantic model for the dataset configuration.

    Attributes:
        topic (str): The topic of the dataset.
        domains (list[str]): The domains of the dataset.
        language (str): The language of the dataset.
        additional_description (str): The additional description of the dataset.
        num_entries (int): The number of entries to generate.
    """

    topic: str = Field(..., min_length=1)
    domains: list[str] = Field(..., min_items=1)
    language: str = Field("English", min_length=1)
    additional_description: str = Field("", max_length=1000)
    num_entries: int = Field(1000, gt=1)


class DatasetGeneratorConfig(BaseModel):
    """Pydantic model for the dataset generator configuration."""

    dataset_config: DatasetConfig
    llm_config: LLMConfig


class InputMessage(BaseModel):
    """Pydantic model for a message in the generated text."""

    role: Literal["system", "user"]
    content: str


class EntryKeywords(BaseModel):
    """Pydantic model for the keywords in the generated text."""

    keywords: list[str]


class GeneratedText(BaseModel):
    """Pydantic model for the generated text."""

    text: str


class InstructMessage(BaseModel):
    """Pydantic model for a message in the Instruct dataset."""

    role: Literal["system", "user", "assistant"]
    content: str


class GeneratedInstructText(BaseModel):
    """Pydantic model for the generated text in the Instruct dataset."""

    messages: list[InstructMessage]


class PreferencePrompt(BaseModel):
    """Pydantic model for the prompt in the Preference dataset."""

    role: Literal["system", "user"]
    content: str


class PreferenceChosen(BaseModel):
    """Pydantic model for the chosen text in the Preference dataset."""

    role: Literal["assistant"]
    content: str


class PreferenceRejected(BaseModel):
    """Pydantic model for the rejected text in the Preference dataset."""

    role: Literal["assistant"]
    content: str


class GeneratedPreferenceText(BaseModel):
    """Pydantic model for the generated text in the Preference dataset."""

    prompt: list[PreferencePrompt]
    chosen: PreferenceChosen
    rejected: PreferenceRejected


class GeneratedSummaryText(BaseModel):
    """Pydantic model for the generated summary text."""

    text: str
    summary: str


class GeneratedSentimentAnalysis(BaseModel):
    """Pydantic model for the generated sentiment analysis."""

    prompt: str
    label: Literal["positive", "negative", "neutral"]


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
    ]
