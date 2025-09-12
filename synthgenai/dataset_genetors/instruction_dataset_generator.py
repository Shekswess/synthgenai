"""Instruction Dataset Generator module."""

from synthgenai.dataset_genetors.dataset_generator import DatasetGenerator


class InstructionDatasetGenerator(DatasetGenerator):
    """Instruction Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Instruction Dataset'."""
        self.dataset.set_dataset_type("Instruction Dataset")

    def _generate_entry(self, keyword: str):
        """
        Generate an instruction dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return self._generate_entry_with_prompts(
            system_prompt="entry_instruction_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
        )

    async def _agenerate_entry(self, keyword: str):
        """
        Generate an instruction dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return await self._agenerate_entry_with_prompts(
            system_prompt="entry_instruction_system_prompt",
            user_prompt="entry_user_prompt",
            keyword=keyword,
        )
