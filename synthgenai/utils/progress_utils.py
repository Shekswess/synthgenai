"""Progress utilities for dataset generation."""

import sys
from typing import Optional

from loguru import logger
from tqdm import tqdm


class TqdmLoguruHandler:
    """Custom tqdm class that works with loguru logger."""

    def __init__(
        self,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = "it",
        disable: bool = False,
        **kwargs
    ):
        """
        Initialize the progress bar with loguru compatibility.

        Args:
            total (Optional[int]): Total number of iterations
            desc (Optional[str]): Description for the progress bar
            unit (str): Unit of measurement for progress
            disable (bool): Whether to disable the progress bar
            **kwargs: Additional keyword arguments for tqdm
        """
        # Configure tqdm to work with loguru
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            disable=disable,
            file=sys.stdout,
            dynamic_ncols=True,
            **kwargs
        )
        self.desc = desc or "Processing"

    def update(self, n: int = 1):
        """Update the progress bar by n units."""
        self.pbar.update(n)

    def set_description(self, desc: str):
        """Set the description for the progress bar."""
        self.pbar.set_description(desc)
        self.desc = desc

    def set_postfix(self, **kwargs):
        """Set the postfix for the progress bar."""
        self.pbar.set_postfix(**kwargs)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()


class ProgressManager:
    """Manager class for handling progress bars in dataset generation."""

    @staticmethod
    def create_progress_bar(
        total: int,
        desc: str,
        unit: str = "it",
        disable: bool = False,
    ) -> TqdmLoguruHandler:
        """
        Create a progress bar compatible with loguru logging.

        Args:
            total (int): Total number of iterations
            desc (str): Description for the progress bar
            unit (str): Unit of measurement
            disable (bool): Whether to disable the progress bar

        Returns:
            TqdmLoguruHandler: Progress bar handler
        """
        return TqdmLoguruHandler(
            total=total,
            desc=desc,
            unit=unit,
            disable=disable,
            leave=True,
            ncols=100,
            bar_format=(
                '{l_bar}{bar}| {n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}]'
            )
        )

    @staticmethod
    def log_progress_start(desc: str, total: int):
        """Log the start of a progress operation."""
        logger.info(f"{desc} - Processing {total} items")

    @staticmethod
    def log_progress_complete(
        desc: str,
        completed: int,
        total: int,
        success_rate: Optional[float] = None,
    ):
        """Log the completion of a progress operation."""
        if success_rate is not None:
            logger.info(
                f"{desc} completed - {completed}/{total} items processed "
                f"(Success rate: {success_rate:.1%})"
            )
        else:
            logger.info(
                f"{desc} completed - {completed}/{total} items processed"
            )