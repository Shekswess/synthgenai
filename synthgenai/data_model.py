"""Pydantic models for the SynthGen package."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    """Enum for the dataset types."""

    RAW = "Raw Dataset"
    INSTRUCTION = "Instruction Dataset"
    PREFERENCE = "Preference Dataset"


class LLMConfig(BaseModel):
    """
    Pydantic model for the LLM configuration.

    Attributes:
        model (str): The model name of the LLM.
        temperature (float): The temperature value from 0.0 to 1.0.
        top_p (float): The top_p value from 0.0 to 1.0.
        max_tokens (int): The maximum number of tokens to generate completions from 1000 to max value.
    """

    model: str = Field(..., min_length=1)
    temperature: float = Field(None, ge=0.0, le=1.0)
    top_p: float = Field(None, ge=0.0, le=1.0)
    max_tokens: int = Field(None, gt=1000)


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
    language: str = Field(..., min_length=1)
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


class EntryRawDataset(BaseModel):
    """Pydantic model for the Raw dataset."""

    keyword: str
    topic: str
    language: str
    generated_text: GeneratedText


class InstructMessage(BaseModel):
    """Pydantic model for a message in the Instruct dataset."""

    role: Literal["system", "user", "assistant"]
    content: str


class InstructGeneratedText(BaseModel):
    """Pydantic model for the generated text in the Instruct dataset."""

    messages: list[InstructMessage]


class EntryInstructDataset(BaseModel):
    """Pydantic model for the Instruct dataset."""

    keyword: str
    topic: str
    language: str
    generated_text: InstructGeneratedText


class PreferenceMessage(BaseModel):
    """Pydantic model for a message in the Preference dataset."""

    role: Literal["user", "system", "assistant"]
    content: str
    option: Literal["chosen", "rejected"] = None


class PreferenceGeneratedText(BaseModel):
    """Pydantic model for the generated text in the Preference dataset."""

    messages: list[PreferenceMessage]


class EntryPreferenceDataset(BaseModel):
    """Pydantic model for the Preference dataset."""

    keyword: str
    topic: str
    language: str
    generated_text: PreferenceGeneratedText
