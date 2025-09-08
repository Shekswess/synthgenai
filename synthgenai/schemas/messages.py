"""Message models for the SynthGenAI package."""

from typing import Literal

from pydantic import BaseModel


class InputMessage(BaseModel):
    """Pydantic model for a message in the generated text."""

    role: Literal["system", "user"]
    content: str


class InstructMessage(BaseModel):
    """Pydantic model for a message in the Instruct dataset."""

    role: Literal["system", "user", "assistant"]
    content: str


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
