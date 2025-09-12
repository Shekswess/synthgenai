"""Preference Dataset Generator."""

from synthgenai.dataset_genetors.dataset_generator import DatasetGenerator


class PreferenceDatasetGenerator(DatasetGenerator):
    """Preference Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Preference Dataset'."""
        self.dataset.set_dataset_type("Preference Dataset")

    def _generate_entry(self, keyword: str):
        """
        Generate a preference dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return self._generate_entry_with_prompts(
            system_prompt="entry_preference_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
        )

    async def _agenerate_entry(self, keyword: str):
        """
        Generate a preference dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return await self._agenerate_entry_with_prompts(
            system_prompt="entry_preference_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
        )
