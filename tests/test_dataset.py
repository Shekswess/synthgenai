"""Tests for the Dataset class."""

import os

from synthgenai.data_model import DatasetConfig
from synthgenai.dataset import Dataset

TEST_STRING = """
```
---
language:
- en
size_category:
- <1K
task_categories:
- text-generation
license:
- apache-2.0
tags:
- synthetic
- text
- synthgen
- healthcare
- machine-learning
- deep-learning
- artificial-intelligence
---

## Description

This synthetic dataset focuses on AI, machine learning, and deep learning applications in healthcare.

*   **Topic:** Artificial Intelligence, specifically in healthcare implementations of machine learning and deep learning.
*   **Language:** English
*   **Domains:** Machine Learning, Deep Learning
*   **Number of Entries:** 2
*   **Dataset Type:** This is a synthetic dataset, not a pre-defined type like a raw dataset or an instruction dataset.
*   **Model Used:** gemini/gemini-1.5-flash-8b. This large language model was used to generate the synthetic data.
*   **Size:**  The dataset contains fewer than 1000 entries.
*   **License:** Apache-2.0
```
"""


def test_dataset_initialization():
    """Test the initialization of the Dataset class."""
    dataset_config = DatasetConfig(
        topic="Test Topic",
        domains=["domain1", "domain2"],
        language="en",
        additional_description="Additional description",
        num_entries=10,
    )
    dataset = Dataset(dataset_config)
    assert dataset.topic == "Test Topic"
    assert dataset.domains == ["domain1", "domain2"]
    assert dataset.language == "en"
    assert dataset.additional_description == "Additional description"
    assert dataset.num_keywords == 10


def test_getters_and_setters():
    """Test the getters and setters of the Dataset class."""
    dataset_config = DatasetConfig(
        topic="Test Topic",
        domains=["domain1", "domain2"],
        language="en",
        additional_description="Additional description",
        num_entries=10,
    )
    dataset = Dataset(dataset_config)
    dataset.set_dataset_type("Instruction Dataset")
    assert dataset.get_dataset_type() == "Instruction Dataset"
    dataset.set_description("Test Description")
    assert dataset.get_description() == "Test Description"
    dataset.set_keywords(["keyword1", "keyword2"])
    assert dataset.get_keywords() == ["keyword1", "keyword2"]
    dataset.set_data([{"data": "test"}])
    assert dataset.get_data() == [{"data": "test"}]


def test_prepare_local_save(monkeypatch):
    """Test the _prepare_local_save method of the Dataset class."""
    dataset_config = DatasetConfig(
        topic="Test Topic",
        domains=["domain1", "domain2"],
        language="en",
        additional_description="Additional description",
        num_entries=10,
    )
    dataset = Dataset(dataset_config)
    monkeypatch.setattr(os, "makedirs", lambda x, exist_ok: None)
    path = dataset._prepare_local_save(None)
    assert "test_topic_dataset" in path


def test_get_hf_token(monkeypatch):
    """Test the _get_hf_token method of the Dataset class."""
    dataset_config = DatasetConfig(
        topic="Test Topic",
        domains=["domain1", "domain2"],
        language="en",
        additional_description="Additional description",
        num_entries=10,
    )
    dataset = Dataset(dataset_config)
    monkeypatch.setenv("HF_TOKEN", "test_token")
    token = dataset._get_hf_token(None)
    assert token == "test_token"
