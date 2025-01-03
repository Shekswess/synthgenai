"""Utility functions for the SynthGenAI package."""

import json

import yaml
from loguru import logger


def convert_json_keywords(response) -> dict:
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


def convert_json_entry(response) -> dict:
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
        response = {}

    if not isinstance(response, dict):
        logger.error("LLM isn't returning a valid JSON object")
        response = {}

    return response


def convert_markdown(text: str) -> str:
    """
    Convert a markdown text to a string.

    Args:
        text (str): The markdown text to convert.

    Returns:
        str: The converted string.
    """
    if "```" in text:
        text = text.replace("```", "")
    return text


def extract_content(llm_generated_card: str) -> str:
    """
    Extract the content from a YAML string.

    Args:
        llm_generated_card (str): The YAML string generated by LLM.

    Returns:
        str: The extracted content.

    Raises:
        ValueError: If the YAML string does not contain the expected sections.
    """
    sections = llm_generated_card.split("---")
    if len(sections) < 3:
        logger.error("Invalid YAML string: missing sections")
        raise ValueError("Invalid YAML string: missing sections")

    return sections[2]


def merge_metadata(hf_generated_card: str, llm_generated_card: str) -> str:
    """
    Merge the metadata from two YAML strings.

    Args:
        hf_generated_card (str): The YAML string generated by Hugging Face.
        llm_generated_card (str): The YAML string generated by LLM.

    Returns:
        str: The merged metadata YAML string.

    Raises:
        ValueError: If the YAML strings do not contain the expected sections.
    """

    try:
        hf_generated_card_metadata = yaml.safe_load(hf_generated_card.split("---")[1])
        llm_generated_card_metadata = yaml.safe_load(llm_generated_card.split("---")[1])
    except IndexError:
        logger.error("Invalid YAML string: missing sections")
        raise ValueError("Invalid YAML string: missing sections")

    merged_metadata = {**hf_generated_card_metadata, **llm_generated_card_metadata}

    merged_yaml = yaml.dump(merged_metadata, default_flow_style=False)

    merged_output = f"---\n{merged_yaml}---\n"

    return merged_output


def save_markdown(text: str, file_path: str):
    """
    Save a markdown text to a file.

    Args:
        text (str): The markdown text to save.
        file_path (str): The file path to save the markdown file to.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        with open(file_path, "w") as f:
            f.write(text)
        logger.info(f"Markdown successfully saved to {file_path}")
    except IOError as e:
        logger.error(f"Failed to save markdown to {file_path}: {e}")
        raise
