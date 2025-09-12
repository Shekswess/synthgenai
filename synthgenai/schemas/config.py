"""Configuration models for the SynthGenAI package."""

from pydantic import AnyUrl, BaseModel, Field


class LLMConfig(BaseModel):
    """
    Pydantic model for the LLM configuration.

    Attributes:
        model (str): The model name of the LLM.
        temperature (float): Temperature (0.0-1.0) controlling randomness.
        top_p (float): Top_p value (0.0-1.0) for nucleus sampling.
        max_tokens (int): Maximum tokens for completions (min 1000).
        api_base (AnyUrl): The API base URL for the LLM service.
        api_key (str): The API key for authenticating with the LLM service.
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
        domains (list[str]): Dataset domains for different areas/categories.
        language (str): The language of the dataset, default is "English".
        additional_description (str): Additional dataset description.
        num_entries (int): Number of entries to generate (must be > 1).
    """

    topic: str = Field(..., min_length=1)
    domains: list[str] = Field(..., min_items=1)
    language: str = Field("English", min_length=1)
    additional_description: str = Field("", max_length=1000)
    num_entries: int = Field(1000, gt=10)


class DatasetGeneratorConfig(BaseModel):
    """
    Pydantic model for the dataset generator configuration.

    Attributes:
        dataset_config (DatasetConfig): The configuration for the dataset.
        llm_config (LLMConfig): The configuration for the LLM.
    """

    dataset_config: DatasetConfig
    llm_config: LLMConfig
