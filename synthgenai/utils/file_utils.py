"""File utility functions for the SynthGenAI package."""

from loguru import logger


class FileUtils:
    """Utility class for file operations."""

    @staticmethod
    def save_markdown(text: str, file_path: str) -> None:
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
        except OSError as e:
            logger.error(f"Failed to save markdown to {file_path}: {e}")
            raise
