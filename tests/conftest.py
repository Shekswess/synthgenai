"""Pytest configuration and shared fixtures."""

import pytest

from synthgenai.schemas.config import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
)


@pytest.fixture
def sample_dataset_config():
    """Create a sample dataset configuration for testing."""
    return DatasetConfig(
        topic="Test Topic",
        domains=["test", "domain"],
        language="English",
        additional_description="Test description",
        num_entries=50,
    )


@pytest.fixture
def sample_llm_config():
    """Create a sample LLM configuration for testing."""
    return LLMConfig(model="gpt-4", temperature=0.7, max_tokens=2000)


@pytest.fixture
def sample_generator_config(sample_dataset_config, sample_llm_config):
    """Create a sample dataset generator configuration for testing."""
    return DatasetGeneratorConfig(
        dataset_config=sample_dataset_config, llm_config=sample_llm_config
    )
