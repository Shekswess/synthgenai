"""JSON utility functions for the SynthGenAI package."""

import json
import re
from typing import Dict, Optional

from loguru import logger


class JsonUtils:
    """Utility class for JSON operations with robust parsing capabilities."""

    @staticmethod
    def _extract_json_from_text(response: str) -> Optional[str]:
        """
        Extract JSON content from text using regex patterns.

        Handles various formats:
        - Markdown code blocks with ```json
        - Multiple JSON objects in text
        - JSON wrapped in other text
        - Malformed JSON with common issues

        Args:
            response (str): The raw response text.

        Returns:
            Optional[str]: Extracted JSON string or None if not found.
        """
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid response received")
            return None

        response = response.strip()

        json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        json_blocks = re.findall(
            json_block_pattern, response, re.DOTALL | re.IGNORECASE
        )

        if json_blocks:
            json_content = json_blocks[-1].strip()
            logger.debug("Extracted JSON from code block")
            return json_content

        json_object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        json_matches = re.findall(json_object_pattern, response, re.DOTALL)

        if json_matches:
            for match in reversed(json_matches):
                try:
                    json.loads(match)
                    logger.debug("Found valid JSON object in text")
                    return match
                except json.JSONDecodeError:
                    continue

        json_array_pattern = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"
        array_matches = re.findall(json_array_pattern, response, re.DOTALL)

        if array_matches:
            for match in reversed(array_matches):
                try:
                    json.loads(match)
                    logger.debug("Found valid JSON array in text")
                    return match
                except json.JSONDecodeError:
                    continue

        delimiter_patterns = [
            r"(?:^|\n)\s*(\{.*?\})\s*(?:\n|$)",
            r"```\s*(\{.*?\})\s*```",
            r"json\s*:\s*(\{.*?\})",
            r"response\s*:\s*(\{.*?\})",
        ]

        for pattern in delimiter_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in reversed(matches):
                    try:
                        json.loads(match)
                        logger.debug("Found valid JSON with delimiter pattern")
                        return match
                    except json.JSONDecodeError:
                        continue

        if response.strip().startswith(("{", "[")):
            logger.debug("Attempting to parse entire response as JSON")
            return response

        logger.warning("No valid JSON found in response")
        return None

    @staticmethod
    def _fix_common_json_issues(json_str: str) -> str:
        """
        Fix common JSON formatting issues.

        Args:
            json_str (str): The JSON string to fix.

        Returns:
            str: Fixed JSON string.
        """
        if not json_str:
            return json_str

        # Remove common prefixes/suffixes
        json_str = json_str.strip()

        # Fix trailing commas
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        # Fix single quotes to double quotes (but not inside strings)
        json_str = re.sub(r"(?<!\\)'([^']*?)(?<!\\)'", r'"\1"', json_str)

        # Fix missing quotes around keys
        json_str = re.sub(r"(\w+)(\s*:\s*)", r'"\1"\2', json_str)

        # Fix double-quoted keys that got double-quoted again
        json_str = re.sub(r'""([^"]+)""', r'"\1"', json_str)

        # Remove any leading/trailing non-JSON characters
        json_str = json_str.strip(" \t\n\r.,;")

        return json_str

    @staticmethod
    def _parse_json_safely(json_str: str) -> Optional[Dict]:
        """
        Safely parse JSON string with multiple attempts and error recovery.

        Args:
            json_str (str): The JSON string to parse.

        Returns:
            Optional[Dict]: Parsed dictionary or None if parsing fails.
        """
        if not json_str:
            return None

        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
            elif (
                isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], dict)
            ):
                logger.debug("Received JSON array, using first element")
                return result[0]
        except json.JSONDecodeError as e:
            logger.debug(f"Initial JSON parse failed: {e}")

        try:
            fixed_json = JsonUtils._fix_common_json_issues(json_str)
            result = json.loads(fixed_json)
            if isinstance(result, dict):
                logger.debug("Successfully parsed JSON after fixing common issues")
                return result
            elif (
                isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], dict)
            ):
                logger.debug("Received JSON array after fixes, using first element")
                return result[0]
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed after fixes: {e}")

        nested_patterns = [
            r'"(?:data|response|result|content)"\s*:\s*(\{.*?\})',
            r":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
        ]

        for pattern in nested_patterns:
            matches = re.findall(pattern, json_str, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, dict):
                        logger.debug("Successfully extracted nested JSON")
                        return result
                except json.JSONDecodeError:
                    continue

        logger.warning("All JSON parsing attempts failed")
        return None

    @staticmethod
    def _unwrap_response(parsed_response: Dict) -> Dict:
        """
        Unwrap nested response structures.

        Args:
            parsed_response (Dict): The parsed JSON response.

        Returns:
            Dict: Unwrapped response or original if no wrapper found.
        """
        if not isinstance(parsed_response, dict):
            return parsed_response

        # Check for common wrapper keys
        wrapper_keys = ["data", "response", "result", "content", "output", "answer"]

        for wrapper_key in wrapper_keys:
            if wrapper_key in parsed_response:
                wrapped_content = parsed_response[wrapper_key]
                if isinstance(wrapped_content, dict):
                    logger.debug(f"Unwrapping response from '{wrapper_key}' key")
                    return wrapped_content
                elif isinstance(wrapped_content, list) and len(wrapped_content) > 0:
                    if isinstance(wrapped_content[0], dict):
                        logger.debug(
                            f"Unwrapping first item from '{wrapper_key}' array"
                        )
                        return wrapped_content[0]

        return parsed_response

    @staticmethod
    def convert_keywords_labels(response: str) -> Dict:
        """
        Convert a JSON string response to a dictionary with robust parsing.

        Args:
            response (str): The JSON string response.

        Returns:
            Dict: The converted dictionary.

        Raises:
            ValueError: If the JSON response is invalid after all parsing
                attempts.
            TypeError: If the response is not a valid JSON object.
        """
        logger.debug("Starting robust JSON parsing for keywords/labels")

        json_content = JsonUtils._extract_json_from_text(response)
        if not json_content:
            logger.error("No JSON content found in response")
            raise ValueError("No valid JSON found in response")

        parsed_response = JsonUtils._parse_json_safely(json_content)
        if not parsed_response:
            logger.error("Failed to parse JSON content")
            raise ValueError("Invalid JSON response after all parsing attempts")

        if not isinstance(parsed_response, dict):
            logger.error("Parsed response is not a dictionary")
            raise TypeError("LLM response is not a valid JSON object")

        unwrapped_response = JsonUtils._unwrap_response(parsed_response)

        logger.debug("Successfully parsed and unwrapped JSON response")
        return unwrapped_response

    @staticmethod
    def convert_entry(response: str) -> Dict:
        """
        Convert a JSON string response to a dictionary with robust parsing.

        Args:
            response (str): The JSON string response.

        Returns:
            Dict: The converted dictionary. Returns empty dict on parsing
                failure.

        Note:
            This method returns an empty dict on error instead of raising
            exceptions, unlike convert_keywords_labels.
        """
        logger.debug("Starting robust JSON parsing for entry")

        try:
            json_content = JsonUtils._extract_json_from_text(response)
            if not json_content:
                logger.error("No JSON content found in response")
                return {}

            parsed_response = JsonUtils._parse_json_safely(json_content)
            if not parsed_response:
                logger.error("Failed to parse JSON content")
                return {}

            if not isinstance(parsed_response, dict):
                logger.error("Parsed response is not a dictionary")
                return {}

            unwrapped_response = JsonUtils._unwrap_response(parsed_response)

            logger.debug("Successfully parsed and unwrapped JSON response")
            return unwrapped_response

        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing: {e}")
            return {}
