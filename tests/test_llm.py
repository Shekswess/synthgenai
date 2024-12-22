"""Tests for the LLM class."""

import litellm
import pytest

from synthgenai.data_model import LLMConfig
from synthgenai.llm import LLM


def test_llm_initialization(monkeypatch):
    """Test the initialization of the LLM class."""
    monkeypatch.setenv("OPENAI_API_KEY", "api_key")
    llm_config = LLMConfig(
        model="gpt-3.5-turbo", temperature=0.7, top_p=0.9, max_tokens=1100
    )
    llm = LLM(llm_config)
    assert llm.model == "gpt-3.5-turbo"
    assert llm.temperature == 0.7
    assert llm.top_p == 0.9
    assert llm.max_tokens == 1100


def test_check_allowed_models():
    """Test the _check_allowed_models method."""
    llm_config = LLMConfig(
        model="disallowed-model", temperature=0.7, top_p=0.9, max_tokens=1100
    )
    with pytest.raises(ValueError):
        LLM(llm_config)


def test_check_llm_api_keys(monkeypatch):
    """Test the _check_llm_api_keys method."""
    monkeypatch.setenv("OPENAI_API_KEY", "api_key")
    llm_config = LLMConfig(
        model="gpt-3.5-turbo", temperature=0.7, top_p=0.9, max_tokens=1100
    )
    llm = LLM(llm_config)
    assert llm.model == "gpt-3.5-turbo"


def test_check_langfuse_api_keys(monkeypatch):
    """Test the _check_langfuse_api_keys method."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public_key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret_key")
    monkeypatch.setenv("LANGFUSE_HOST", "host")
    monkeypatch.setenv("OPENAI_API_KEY", "api_key")
    llm_config = LLMConfig(
        model="gpt-3.5-turbo", temperature=0.7, top_p=0.9, max_tokens=1100
    )
    llm = LLM(llm_config)
    assert "langfuse" in litellm.success_callback
    assert "langfuse" in litellm.failure_callback


def test_check_ollama():
    """Test the _check_ollama method"""
    llm_config = LLMConfig(
        model="ollama/model", temperature=0.7, top_p=0.9, max_tokens=1100
    )
    llm = LLM(llm_config)
    assert llm.api_base == "http://localhost:11434"
