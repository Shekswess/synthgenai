"""Prompt Utils for handling prompt templates with Jinja2."""

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger


class PromptUtils:
    """
    Manages prompt templates using Jinja2.
    """

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        """
        Initializes the PromptUtils with templates from the specified directory.

        Args:
            prompt_path (str | Path | None): Path to the directory containing
                prompt templates. If None, uses the default prompts directory.
        """
        if prompt_path is None:
            current_dir = Path(__file__).parent.parent
            prompt_path = current_dir / "prompts"

        self.env = Environment(loader=FileSystemLoader(prompt_path))
        self.prompts = {}
        for filename in os.listdir(prompt_path):
            self.prompts[filename] = self.env.get_template(filename)
        logger.info(
            "PromptUtils initialized with prompts from directory: {}",
            prompt_path,
        )

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Retrieves and renders a prompt template by name with the provided keyword arguments.

        Args:
            prompt_name (str): The name of the prompt template to retrieve.
            **kwargs: Keyword arguments to render the template with.

        Returns:
            str: The rendered prompt string.

        Raises:
            KeyError: If the specified prompt_name does not exist in the loaded prompts.
        """
        return self.prompts[prompt_name].render(**kwargs).strip()
