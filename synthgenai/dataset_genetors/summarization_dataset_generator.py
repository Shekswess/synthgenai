"""Summarization Dataset Generator module."""

from synthgenai.dataset_genetors.dataset_generator import DatasetGenerator


class SummarizationDatasetGenerator(DatasetGenerator):
    """Summarization Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Summarization Dataset'."""
        self.dataset.set_dataset_type("Summarization Dataset")

    def _generate_entry(self, keyword: str):
        """
        Generate a summarization dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return self._generate_entry_with_prompts(
            system_prompt="entry_summarization_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
        )

    async def _agenerate_entry(self, keyword: str):
        """
        Generate a summarization dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return await self._agenerate_entry_with_prompts(
            system_prompt="entry_summarization_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
        )
