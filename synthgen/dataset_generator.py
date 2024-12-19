"""Dataset Generator module."""

from loguru import logger
from pydantic import ValidationError

from .data_model import (
    DatasetGeneratorConfig,
    EntryInstructDataset,
    EntryPreferenceDataset,
    EntryRawDataset,
    EntryKeywords
)
from .dataset import Dataset
from .llm import LLM
from .prompts import (
    ENTRY_INSTRUCT_SYSTEM_PROMPT,
    ENTRY_INSTRUCT_USER_PROMPT,
    ENTRY_PREFERENCE_SYSTEM_PROMPT,
    ENTRY_PREFERENCE_USER_PROMPT,
    ENTRY_RAW_DATASET_SYSTEM_PROMPT,
    ENTRY_RAW_DATASET_USER_PROMPT,
    KEYWORD_SYSTEM_PROMPT,
    KEYWORD_USER_PROMPT,
    MARKDOWN_DESCRIPTION_SYSTEM_PROMPT,
    MARKDOWN_DESCRIPTION_USER_PROMPT,
)
from .utils import convert_json


class DatasetGenerator:
    """Dataset Generator class."""

    def __init__(self, dataset_generator_config: DatasetGeneratorConfig):
        """
        Initialize the DatasetGenerator with the provided configuration.

        Args:
            dataset_generator_config (DatasetGeneratorConfig): The configuration for the dataset generator.
        """
        self.dataset = Dataset(dataset_generator_config.dataset_config)
        self.llm = LLM(dataset_generator_config.llm_config)
        logger.info("Initialized Dataset Generator")

    def _create_messages(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> list[dict]:
        """
        Create messages for LLM interaction.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user prompt.
            **kwargs: Additional keyword arguments to be formatted into the prompts.

        Returns:
            list[dict]: The messages for LLM interaction.
        """
        logger.debug("Creating messages for LLM interaction")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(**kwargs)},
        ]

    def _generate_keywords(self):
        """Generate keywords for the dataset."""
        logger.info("Starting keyword generation")
        if self.llm.check_response_format():
            self.llm.set_response_format({"type": "json_object"})
        keywords = []
        num_keywords = self.dataset.get_num_keywords()
        while len(keywords) < num_keywords:
            remaining_keywords = num_keywords - len(keywords)
            messages = self._create_messages(
                KEYWORD_SYSTEM_PROMPT,
                KEYWORD_USER_PROMPT,
                topic=self.dataset.topic,
                domains=", ".join(self.dataset.domains),
                language=self.dataset.language,
                additional_description=self.dataset.additional_description,
                num_keywords=remaining_keywords,
            )
            response = self.llm.generate(messages)
            response = convert_json(response)
            new_keywords = response.get("keywords", [])
            if not new_keywords:
                logger.warning("No new keywords generated, breaking the loop")
                break
            keywords.extend(new_keywords)
            logger.info(
                f"Generated {len(new_keywords)} new keywords, {len(keywords)} total keywords"
            )
        try:
            keywords = EntryKeywords(keywords=keywords)
        except ValidationError as e:
            logger.error(f"Validation error for keywords: {e}")
            raise e
        self.dataset.set_keywords(keywords.keywords)

    def _generate_description(self):
        """Generate a description for the dataset."""
        logger.info("Generating dataset description")
        self.llm.set_response_format(None)
        messages = self._create_messages(
            MARKDOWN_DESCRIPTION_SYSTEM_PROMPT,
            MARKDOWN_DESCRIPTION_USER_PROMPT,
            topic=self.dataset.get_topic(),
            domains=", ".join(self.dataset.get_domains()),
            language=self.dataset.get_language(),
            additional_description=self.dataset.get_additional_description(),
            num_keywords=self.dataset.get_num_keywords(),
            dataset_type=self.dataset.get_dataset_type(),
            model=self.llm.get_model(),
        )
        response = self.llm.generate(messages)
        self.dataset.set_description(response)
        logger.info("Dataset description generated")

    def _generate_entry(self, keyword: str) -> str:
        """
        Generate an entry for the dataset. Must be implemented by subclasses.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        raise NotImplementedError("Subclasses must implement _generate_entry")

    def _set_dataset_type(self):
        """Set the dataset type. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _set_dataset_type")

    def _generate_entries(self):
        """Generate entries for the dataset."""
        logger.info("Generating entries for dataset")
        if self.llm.check_response_format():
            self.llm.set_response_format({"type": "json_object"})
        self._set_dataset_type()
        data = []
        keywords = self.dataset.get_keywords()
        for keyword in keywords:
            entry = self._generate_entry(keyword)
            if entry:
                data.append(entry)
                logger.info(f"Generated entry for keyword: {keyword}")
        self.dataset.set_data(data)

    def generate_dataset(self):
        """Generate the complete dataset."""
        self._generate_keywords()
        self._generate_description()
        self._generate_entries()
        return self.dataset


class RawDatasetGenerator(DatasetGenerator):
    """Raw Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Raw Dataset'."""
        self.dataset.set_dataset_type("Raw Dataset")

    def _generate_entry(self, keyword: str) -> str:
        """
        Generate a raw dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.

        Returns:
            str: The generated raw dataset entry.

        Raises:
            ValidationError: If the generated entry does not match the data model.
        """
        messages = self._create_messages(
            ENTRY_RAW_DATASET_SYSTEM_PROMPT,
            ENTRY_RAW_DATASET_USER_PROMPT,
            keyword=keyword,
            topic=self.dataset.get_topic(),
            language=self.dataset.get_language(),
        )
        response = self.llm.generate(messages)
        response = convert_json(response)
        try:
            entry = EntryRawDataset(**response)
            logger.debug(f"Raw dataset entry: {entry}")
        except ValidationError as e:
            logger.error(f"Validation error for keyword {keyword}: {e}")
            return None
        return entry.model_dump()


class InstructionDatasetGenerator(DatasetGenerator):
    """Instruction Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Instruction Dataset'."""
        self.dataset.set_dataset_type("Instruction Dataset")

    def _generate_entry(self, keyword: str) -> str:
        """
        Generate an instruction dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.

        Returns:
            str: The generated instruction dataset entry.

        Raises:
            ValidationError: If the generated entry does not match the data model.
        """
        messages = self._create_messages(
            ENTRY_INSTRUCT_SYSTEM_PROMPT,
            ENTRY_INSTRUCT_USER_PROMPT,
            keyword=keyword,
            topic=self.dataset.get_topic(),
            language=self.dataset.get_language(),
        )
        response = self.llm.generate(messages)
        response = convert_json(response)
        try:
            entry = EntryInstructDataset(**response)
            logger.debug(f"Instruction dataset entry: {entry}")
        except ValidationError as e:
            logger.error(f"Validation error for keyword {keyword}: {e}")
            return None
        return entry.model_dump()


class PreferenceDatasetGenerator(DatasetGenerator):
    """Preference Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Preference Dataset'."""
        self.dataset.set_dataset_type("Preference Dataset")

    def _generate_entry(self, keyword: str) -> str:
        """
        Generate a preference dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.

        Returns:
            str: The generated preference dataset entry.

        Raises:
            ValidationError: If the generated entry does not match the data model.
        """
        messages = self._create_messages(
            ENTRY_PREFERENCE_SYSTEM_PROMPT,
            ENTRY_PREFERENCE_USER_PROMPT,
            keyword=keyword,
            topic=self.dataset.get_topic(),
            language=self.dataset.get_language(),
        )
        response = self.llm.generate(messages)
        response = convert_json(response)
        try:
            entry = EntryPreferenceDataset(**response)
            logger.debug(f"Preference dataset entry: {entry}")
        except ValidationError as e:
            logger.error(f"Validation error for keyword {keyword}: {e}")
            return None
        return entry.model_dump()
