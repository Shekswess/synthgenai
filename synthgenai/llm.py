"""LLM module"""

import os
from typing import Union

import litellm
from litellm import (
    acompletion,
    completion,
    get_supported_openai_params,
    supports_response_schema,
)
from loguru import logger

from .data_model import InputMessage, LLMConfig

ALLOWED_PREFIXES = [
    "groq/",
    "mistral/",
    "gemini/",
    "bedrock/",
    "sagemaker/",
    "claude",
    "gpt",
    "huggingface/",
    "ollama/",
    "hosted_vllm/",
    "azure/",
    "azure_ai/",
    "vertex_ai/",
]


class LLM:
    """LLM Class"""

    def __init__(self, llm_config: LLMConfig):
        """
        Initialize the LLM class

        Args:
            llm_config (LLMConfig): The configuration for the LLM
        """
        self.model = llm_config.model
        self.temperature = llm_config.temperature
        self.top_p = llm_config.top_p
        self.max_tokens = llm_config.max_tokens
        self.api_base = (
            str(llm_config.api_base) if llm_config.api_base is not None else None
        )
        self.api_key = llm_config.api_key
        self.response_format = None

        self._check_allowed_models()
        if self.api_key is None:
            self._check_llm_api_keys()
        self._check_langfuse_api_keys()
        self._check_vllm()
        self._check_ollama()

        logger.info(f"Initialized LLM model for synthetic dataset creation: {self.model}")

    def _check_allowed_models(self) -> None:
        """Check if the model is allowed"""
        if not any(self.model.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            logger.error(f"Model {self.model} is not allowed")
            raise ValueError(f"Model {self.model} is not allowed")

    def _check_llm_api_keys(self) -> None:
        """Check if the required API keys are set"""
        api_key_checks = {
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            "sagemaker": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            "claude": "ANTHROPIC_API_KEY",
            "gpt": "OPENAI_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "azure": ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"],
            "azure_ai": ["AZURE_AI_API_KEY", "AZURE_AI_API_BASE"],
            "vertex_ai": [
                "GOOGLE_APPLICATION_CREDENTIALS",
                "VERTEXAI_LOCATION",
                "VERTEXAI_PROJECT",
            ],
        }

        for key, env_vars in api_key_checks.items():
            if self.model.startswith(key):
                if isinstance(env_vars, list):
                    missing_vars = [
                        var for var in env_vars if os.environ.get(var) is None
                    ]
                    if missing_vars:
                        if key == "bedrock" or key == "sagemaker":
                            if all(
                                os.environ.get(var)
                                for var in [
                                    "AWS_ACCESS_KEY_ID",
                                    "AWS_SECRET_ACCESS_KEY",
                                    "AWS_REGION",
                                ]
                            ):
                                logger.info(
                                    "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are set"
                                )
                            elif all(
                                os.environ.get(var)
                                for var in ["AWS_REGION", "AWS_PROFILE"]
                            ):
                                logger.info(
                                    "AWS_REGION and AWS_PROFILE are set, using them for bedrock model"
                                )
                            else:
                                for var in missing_vars:
                                    logger.error(f"{var} is not set")
                                raise ValueError(f"{', '.join(missing_vars)} are not set")
                        elif key == "azure":
                            if all(
                                os.environ.get(var)
                                for var in [
                                    "AZURE_API_KEY",
                                    "AZURE_API_BASE",
                                    "AZURE_API_VERSION",
                                ]
                            ):
                                logger.info(
                                    "AZURE_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION are set"
                                )
                            elif all(
                                os.environ.get(var)
                                for var in ["AZURE_AD_TOKEN", "AZURE_API_TYPE"]
                            ):
                                logger.info(
                                    "AZURE_AD_TOKEN and AZURE_API_TYPE are set, using them for azure model"
                                )
                            else:
                                for var in missing_vars:
                                    logger.error(f"{var} is not set")
                                raise ValueError(f"{', '.join(missing_vars)} are not set")
                        else:
                            for var in missing_vars:
                                logger.error(f"{var} is not set")
                            raise ValueError(f"{', '.join(missing_vars)} are not set")
                else:
                    if os.environ.get(env_vars) is None:
                        logger.error(f"{env_vars} is not set")
                        raise ValueError(f"{env_vars} is not set")
        logger.info("Necessary LLM API Keys are set")

    def _check_langfuse_api_keys(self) -> None:
        """Check if the required API keys for Langfuse are set"""
        if (
            "LANGFUSE_PUBLIC_KEY" in os.environ
            and "LANGFUSE_SECRET_KEY" in os.environ
            and "LANGFUSE_HOST" in os.environ
        ):
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]
            logger.info("Langfuse API keys are set")
        else:
            logger.warning("Langfuse API keys are not set")

    def _check_vllm(self) -> None:
        """Check if the model is a VLLM model"""
        if self.model.startswith("hosted_vllm") and self.api_base is None:
            logger.error("VLLM model detected without API Base")
            raise ValueError("VLLM model detected without API Base")

    def _check_ollama(self) -> None:
        """Check if the model is an Ollama model"""
        if self.model.startswith("ollama") and self.api_base is None:
            self.api_base = "http://localhost:11434"
            logger.info(
                "Ollama model detected without API Base - setting API Base to localhost"
            )

    def set_temperature(self, temperature: Union[float, None]) -> None:
        """
        Set the temperature for the LLM

        Args:
            temperature (Union[float, None]): The temperature to set
        """
        self.temperature = temperature

    def get_temperature(self) -> Union[float, None]:
        """
        Get the temperature for the LLM

        Returns:
            Union[float, None]: The temperature if set, None otherwise
        """
        return self.temperature

    def get_model(self) -> str:
        """
        Get the model name

        Returns:
            str: The model name
        """
        return self.model

    def set_response_format(self, response_format: dict) -> None:
        """
        Set the response format for the LLM

        Args:
            response_format (dict): The response format to set
        """
        self.response_format = response_format

    def check_response_format(self) -> bool:
        """
        Check if the response format is supported by the LLM model

        Returns:
            bool: True if the response format is supported, False otherwise
        """
        custom_llm_provider = None
        if not self.model.startswith("ollama") and not self.model.startswith(
            "hosted_vllm"
        ):
            custom_llm_provider = self.model.split("/")[0]
        if "response_format" in get_supported_openai_params(
            model=self.model, custom_llm_provider=custom_llm_provider
        ) and supports_response_schema(
            model=self.model, custom_llm_provider=custom_llm_provider
        ):
            logger.info(f"JSON format is supported by the LLM model: {self.model}")
            return True
        else:
            logger.warning(f"JSON format is not supported by the LLM model: {self.model}")
            return False

    def generate(self, messages: list[InputMessage]) -> str:
        """
        Generate completions using the LLM API

        Args:
            messages (List[InputMessage]): List of messages to generate completions for using the LLM API

        Returns:
            str: The completion generated by the LLM API
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                response_format=self.response_format,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                api_key=self.api_key,
            )
            logger.info(f"Generated completions using LLM model: {self.model}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate completions: {e}")
            raise

    async def agenerate(self, messages: list[InputMessage]) -> str:
        """
        Generate completions using the LLM API asynchronously.

        Args:
            messages (List[InputMessage]): List of messages to generate completions for using the LLM API

        Returns:
            str: The completion generated by the LLM API
        """
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                response_format=self.response_format,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                api_key=self.api_key,
            )
            logger.info(f"Generated completions using LLM model: {self.model}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate completions: {e}")
            raise
