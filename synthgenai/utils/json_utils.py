"""JSON utility functions for the SynthGenAI package."""

import json

from loguru import logger


class JsonUtils:
    """Utility class for JSON operations."""

    @staticmethod
    def convert_keywords_labels(response: str) -> dict:
        """
        Convert a JSON string response to a dictionary.

        Args:
            response (str): The JSON string response.

        Returns:
            dict: The converted dictionary.

        Raises:
            ValueError: If the JSON response is invalid.
            TypeError: If the response is not a valid JSON object.
        """
        if "```json" in response:
            response = response.replace("```json", "").replace("```", "")

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            logger.error("Invalid JSON response")
            raise ValueError("Invalid JSON response")

        if not isinstance(response, dict):
            logger.error("LLM isn't returning a valid JSON object")
            raise TypeError("LLM isn't returning a valid JSON object")

        return response

    @staticmethod
    def convert_entry(response: str) -> dict:
        """
        Convert a JSON string response to a dictionary.

        Args:
            response (str): The JSON string response.

        Returns:
            dict: The converted dictionary.

        Note:
            This method returns an empty dict on error instead of raising
            exceptions, unlike convert_keywords_labels.
        """
        if "```json" in response:
            response = response.replace("```json", "").replace("```", "")

        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            logger.error("Invalid JSON response")
            return {}

        if not isinstance(parsed_response, dict):
            logger.error("LLM isn't returning a valid JSON object")
            return {}

        return parsed_response
