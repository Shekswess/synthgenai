"""Text Classification Dataset Generator module."""

import random

from synthgenai.dataset_genetors.dataset_generator import DatasetGenerator


class TextClassificationDatasetGenerator(DatasetGenerator):
    """Text Classification Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Text Classification Dataset'."""
        self.dataset.set_dataset_type("Text Classification Dataset")

    def _generate_entry(self, keyword: str):
        """
        Generate a text classification dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return self._generate_entry_with_prompts(
            system_prompt="entry_classification_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
            label=random.choice(self.dataset.get_labels()),
        )

    async def _agenerate_entry(self, keyword: str):
        """
        Generate a text classification dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return await self._agenerate_entry_with_prompts(
            system_prompt="entry_classification_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
            label=random.choice(self.dataset.get_labels()),
        )
