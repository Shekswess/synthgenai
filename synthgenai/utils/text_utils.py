"""Text utility functions for the SynthGenAI package."""


class TextUtils:
    """Utility class for text operations."""

    @staticmethod
    def convert_markdown(text: str) -> str:
        """
        Convert a markdown text to a string by removing markdown code blocks.

        Args:
            text (str): The markdown text to convert.

        Returns:
            str: The converted string.
        """
        if "```" in text:
            text = text.replace("```", "")
        return text
