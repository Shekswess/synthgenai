"""Unit tests for LLM module."""

import os
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from synthgenai.llm.llm import LLM
from synthgenai.schemas.config import LLMConfig


class TestLLM:
    """Test cases for LLM class."""

    def test_initialization_with_allowed_model(self):
        """Test LLM initialization with an allowed model."""
        config = LLMConfig(model="gpt-4")

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        assert llm.model == "gpt-4"
        assert llm.temperature is None
        assert llm.top_p is None
        assert llm.max_tokens is None

    def test_initialization_with_disallowed_model(self):
        """Test LLM initialization fails with disallowed model."""
        config = LLMConfig(model="invalid-model")

        with pytest.raises(ValueError, match="Model invalid-model is not allowed"):
            LLM(config)

    def test_set_and_get_temperature(self):
        """Test setting and getting temperature."""
        config = LLMConfig(model="gpt-4")

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        llm.set_temperature(0.5)
        assert llm.get_temperature() == 0.5

    def test_get_model(self):
        """Test getting model name."""
        config = LLMConfig(model="gpt-3.5-turbo")

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        assert llm.get_model() == "gpt-3.5-turbo"

    def test_set_response_format(self):
        """Test setting response format."""
        config = LLMConfig(model="gpt-4")

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        class TestModel(BaseModel):
            test_field: str

        llm.set_response_format(TestModel)
        assert llm.response_format == TestModel

    @patch("synthgenai.llm.llm.completion")
    def test_generate_success(self, mock_completion):
        """Test successful generation."""
        config = LLMConfig(model="gpt-4")

        # Mock the completion response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        messages = [{"role": "user", "content": "Hello"}]
        result = llm.generate(messages)

        assert result == "Test response"
        mock_completion.assert_called_once()

    @patch("synthgenai.llm.llm.acompletion")
    @pytest.mark.asyncio
    async def test_agenerate_success(self, mock_acompletion):
        """Test successful async generation."""
        config = LLMConfig(model="gpt-4")

        # Mock the completion response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test async response"
        mock_acompletion.return_value = mock_response

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        messages = [{"role": "user", "content": "Hello"}]
        result = await llm.agenerate(messages)

        assert result == "Test async response"
        mock_acompletion.assert_called_once()

    def test_check_api_keys_missing_openai(self):
        """Test API key validation for OpenAI models."""
        config = LLMConfig(model="gpt-4")

        # Ensure OPENAI_API_KEY is not set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
                LLM(config)

    def test_ollama_default_api_base(self):
        """Test Ollama model sets default API base."""
        config = LLMConfig(model="ollama/llama2")

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
        ):
            llm = LLM(config)

        assert llm.api_base == "http://localhost:11434"

    @patch("synthgenai.llm.llm.get_supported_openai_params")
    @patch("synthgenai.llm.llm.supports_response_schema")
    def test_check_response_format_supported(
        self, mock_supports_schema, mock_get_params
    ):
        """Test response format check when supported."""
        config = LLMConfig(model="gpt-4")

        mock_get_params.return_value = ["response_format"]
        mock_supports_schema.return_value = True

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        assert llm.check_response_format() is True

    @patch("synthgenai.llm.llm.get_supported_openai_params")
    @patch("synthgenai.llm.llm.supports_response_schema")
    def test_check_response_format_not_supported(
        self, mock_supports_schema, mock_get_params
    ):
        """Test response format check when not supported."""
        config = LLMConfig(model="gpt-4")

        mock_get_params.return_value = []
        mock_supports_schema.return_value = False

        with (
            patch.object(LLM, "_check_llm_api_keys"),
            patch.object(LLM, "_check_langfuse_api_keys"),
            patch.object(LLM, "_check_vllm"),
            patch.object(LLM, "_check_ollama"),
        ):
            llm = LLM(config)

        assert llm.check_response_format() is False
