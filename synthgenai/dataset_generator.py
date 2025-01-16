"""Dataset Generator module."""

import asyncio
import random
import time

from loguru import logger
from pydantic import ValidationError

from .data_model import (
    DatasetGeneratorConfig,
    EntryDataset,
    EntryKeywords,
    EntryLabels,
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
    ENTRY_SENTIMENT_SYSTEM_PROMPT,
    ENTRY_SENTIMENT_USER_PROMPT,
    ENTRY_SUMMARIZATION_SYSTEM_PROMPT,
    ENTRY_SUMMARIZATION_USER_PROMPT,
    KEYWORD_SYSTEM_PROMPT,
    KEYWORD_USER_PROMPT,
    ENTRY_CLASSIFICATION_SYSTEM_PROMPT,
    ENTRY_CLASSIFICATION_USER_PROMPT,
    LABELS_USER_PROMPT,
    LABELS_SYSTEM_PROMPT,
    MARKDOWN_DESCRIPTION_SYSTEM_PROMPT,
    MARKDOWN_DESCRIPTION_USER_PROMPT,
)
from .utils import convert_json_entry, convert_json_keywords_labels

BATCH_SIZE = 5


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
            remaining_keywords = min(num_keywords - len(keywords), 30)
            additional_description = self.dataset.get_additional_description()
            if keywords:
                additional_description += f" Previously generated keywords: {', '.join(keywords)}. \nGenerate new keywords following the provided rules in the system prompt."
            messages = self._create_messages(
                KEYWORD_SYSTEM_PROMPT,
                KEYWORD_USER_PROMPT,
                topic=self.dataset.get_topic(),
                domains=", ".join(self.dataset.get_domains()),
                language=self.dataset.get_language(),
                additional_description=additional_description,
                num_keywords=remaining_keywords,
            )
            response = self.llm.generate(messages)
            logger.debug(f"Keyword generation response: {response}")
            try:
                response = convert_json_keywords_labels(response)
            except ValueError as e:
                logger.error(f"Invalid JSON response: {e}, retrying...")
                continue
            new_keywords = response.get("keywords", [])
            if not new_keywords:
                logger.warning("No new keywords generated, breaking the loop")
                break
            keywords.extend(new_keywords)
            logger.info(
                f"Generated {len(new_keywords)} new keywords, {len(keywords)} total keywords"
            )
        if len(keywords) > num_keywords:
            logger.warning("More keywords generated than required, truncating")
            keywords = keywords[:num_keywords]
        try:
            keywords = EntryKeywords(keywords=keywords)
        except ValidationError as e:
            logger.error(f"Validation error for keywords: {e}")
            raise e
        self.dataset.set_keywords(keywords.keywords)

    def _generate_labels(self):
        """Generate labels for the (text classification) dataset."""
        logger.info("Generating labels for the dataset")
        if self.llm.check_response_format():
            self.llm.set_response_format({"type": "json_object"})
        labels = []
        num_labels = random.randint(2, 5)
        while len(labels) < num_labels:
            remaining_labels = min(num_labels - len(labels), 30)
            additional_description = self.dataset.get_additional_description()
            if labels:
                additional_description += f" Previously generated labels: {', '.join(labels)}. \nGenerate new labels following the provided rules in the system prompt."
            messages = self._create_messages(
                LABELS_SYSTEM_PROMPT,
                LABELS_USER_PROMPT,
                topic=self.dataset.get_topic(),
                domains=", ".join(self.dataset.get_domains()),
                language=self.dataset.get_language(),
                additional_description=additional_description,
                num_labels=remaining_labels,
            )
            response = self.llm.generate(messages)
            logger.debug(f"Label generation response: {response}")
            try:
                response = convert_json_keywords_labels(response)
            except ValueError as e:
                logger.error(f"Invalid JSON response: {e}, retrying...")
                continue
            new_labels = response.get("labels", [])
            if not new_labels:
                logger.warning("No new labels generated, breaking the loop")
                break
            labels.extend(new_labels)
            logger.info(
                f"Generated {len(new_labels)} new labels, {len(labels)} total labels"
            )
        if len(labels) > num_labels:
            logger.warning("More labels generated than required, truncating")
            labels = labels[:num_labels]
        try:
            labels = EntryLabels(labels=labels)
        except ValidationError as e:
            logger.error(f"Validation error for labels: {e}")
            raise e
        self.dataset.set_labels(labels.labels)

    def _generate_description(self):
        """Generate a description for the dataset."""
        logger.info("Generating dataset description")
        tmp_llm_temperature = self.llm.get_temperature()
        self.llm.set_response_format(None)
        self.llm.set_temperature(None)
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
        self.llm.set_temperature(tmp_llm_temperature)
        logger.info("Dataset description generated")

    def _generate_entry(
        self, keyword: str, user_prompt: str, system_prompt: str, **kwargs
    ):
        """
        Generate an entry for the dataset. Specific to each dataset type.

        Args:
            keyword (str): The keyword for which to generate the entry.
            user_prompt (str): The user prompt for the entry.
            system_prompt (str): The system prompt for the entry.
            **kwargs: Additional keyword arguments to be formatted into the prompts (if any).

        Returns:
            dict: The generated raw dataset entry.

        Raises:
            ValidationError: If the generated entry does not match the data model.
        """
        messages = self._create_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            keyword=keyword,
            topic=self.dataset.get_topic(),
            language=self.dataset.get_language(),
            additional_description=self.dataset.get_additional_description(),
            **kwargs,
        )
        response = self.llm.generate(messages)
        try:
            response = convert_json_entry(response)
            entry = EntryDataset(**response)
            logger.debug(f"Raw dataset entry: {entry}")
        except ValidationError as e:
            logger.error(f"Validation error for keyword {keyword}: {e}")
            return None
        return entry.model_dump()

    async def _agenerate_entry(
        self, keyword: str, user_prompt: str, system_prompt: str, **kwargs
    ):
        """
        Generate an entry for the dataset asynchronously. Generate an entry for the dataset. Specific to each dataset type.

        Args:
            keyword (str): The keyword for which to generate the entry.
            user_prompt (str): The user prompt for the entry.
            system_prompt (str): The system prompt for the entry.
            **kwargs: Additional keyword arguments to be formatted into the prompts (if any).

        Returns:
            dict: The generated raw dataset entry.

        Raises:
            ValidationError: If the generated entry does not match the data model.
        """
        messages = self._create_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            keyword=keyword,
            topic=self.dataset.get_topic(),
            language=self.dataset.get_language(),
            additional_description=self.dataset.get_additional_description(),
            **kwargs,
        )
        response = await self.llm.agenerate(messages)
        try:
            response = convert_json_entry(response)
            entry = EntryDataset(**response)
            logger.debug(f"Raw dataset entry: {entry}")
        except ValidationError as e:
            logger.error(f"Validation error for keyword {keyword}: {e}")
            return None
        return entry.model_dump()

    def _set_dataset_type(self):
        """Set the dataset type. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _set_dataset_type")

    async def _agenerate_entries(self):
        """Generate entries for the dataset asynchronously."""
        logger.info("Generating entries for dataset asynchronously")
        if self.llm.check_response_format():
            self.llm.set_response_format({"type": "json_object"})
        self._set_dataset_type()
        logger.info(f"Dataset type: {self.dataset.get_dataset_type()}")
        if self.dataset.get_dataset_type() == "Text Classification Dataset":
            self._generate_labels()
        data = []
        keywords = self.dataset.get_keywords()
        for i in range(0, len(keywords), BATCH_SIZE):
            batch_keywords = keywords[i : i + BATCH_SIZE]
            tasks = [self._agenerate_entry(keyword) for keyword in batch_keywords]
            entries = await asyncio.gather(*tasks)
            time.sleep(10)
            for keyword, entry in zip(batch_keywords, entries):
                if entry:
                    data.append(entry)
                    logger.info(f"Generated entry for keyword: {keyword}")
                    logger.info(f"Number of entries generated: {len(data)}")
                else:
                    logger.warning(
                        f"Skipping entry for keyword: {keyword} due to validation error"
                    )
        self.dataset.set_data(data)

    def _generate_entries(self):
        """Generate entries for the dataset."""
        logger.info("Generating entries for dataset")
        if self.llm.check_response_format():
            self.llm.set_response_format({"type": "json_object"})
        self._set_dataset_type()
        logger.info(f"Dataset type: {self.dataset.get_dataset_type()}")
        if self.dataset.get_dataset_type() == "Text Classification Dataset":
            self._generate_labels()
        data = []
        keywords = self.dataset.get_keywords()
        for keyword in keywords:
            entry = self._generate_entry(keyword)
            if entry:
                data.append(entry)
                logger.info(f"Generated entry for keyword: {keyword}")
                logger.info(f"Number of entry generated: {len(data)}")
            else:
                logger.warning(
                    f"Skipping entry for keyword: {keyword} due to validation error"
                )
        self.dataset.set_data(data)

    def generate_dataset(self):
        """Generate the complete dataset."""
        start_time = time.time()
        self._generate_keywords()
        self._generate_entries()
        num_entries = len(self.dataset.get_data())
        if num_entries != self.dataset.get_num_keywords():
            logger.warning(
                f"Lower number of entries generated than required: {num_entries}, because of validation errors"
            )
            self.dataset.set_num_keywords(num_entries)
        self._generate_description()
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time taken to generate dataset: {total_time:.2f} seconds")
        return self.dataset

    async def agenerate_dataset(self):
        """Generate the complete dataset asynchronously."""
        start_time = time.time()
        self._generate_keywords()
        await self._agenerate_entries()
        num_entries = len(self.dataset.get_data())
        if num_entries != self.dataset.get_num_keywords():
            logger.warning(
                f"Lower number of entries generated than required: {num_entries}, because of validation errors"
            )
            self.dataset.set_num_keywords(num_entries)
        self._generate_description()
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time taken to generate dataset: {total_time:.2f} seconds")
        return self.dataset


class RawDatasetGenerator(DatasetGenerator):
    """Raw Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Raw Dataset'."""
        self.dataset.set_dataset_type("Raw Dataset")

    def _generate_entry(self, keyword: str):
        """
        Generate a raw dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._generate_entry(
            keyword, ENTRY_RAW_DATASET_USER_PROMPT, ENTRY_RAW_DATASET_SYSTEM_PROMPT
        )

    def _agenerate_entry(self, keyword: str):
        """
        Generate a raw dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._agenerate_entry(
            keyword, ENTRY_RAW_DATASET_USER_PROMPT, ENTRY_RAW_DATASET_SYSTEM_PROMPT
        )


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
        return super()._generate_entry(
            keyword, ENTRY_INSTRUCT_USER_PROMPT, ENTRY_INSTRUCT_SYSTEM_PROMPT
        )

    def _agenerate_entry(self, keyword: str):
        """
        Generate an instruction dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._agenerate_entry(
            keyword, ENTRY_INSTRUCT_USER_PROMPT, ENTRY_INSTRUCT_SYSTEM_PROMPT
        )


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
        return super()._generate_entry(
            keyword, ENTRY_PREFERENCE_USER_PROMPT, ENTRY_PREFERENCE_SYSTEM_PROMPT
        )

    def _agenerate_entry(self, keyword: str):
        """
        Generate a preference dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._agenerate_entry(
            keyword, ENTRY_PREFERENCE_USER_PROMPT, ENTRY_PREFERENCE_SYSTEM_PROMPT
        )


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
        return super()._generate_entry(
            keyword, ENTRY_SUMMARIZATION_USER_PROMPT, ENTRY_SUMMARIZATION_SYSTEM_PROMPT
        )

    def _agenerate_entry(self, keyword: str):
        """
        Generate a summarization dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._agenerate_entry(
            keyword, ENTRY_SUMMARIZATION_USER_PROMPT, ENTRY_SUMMARIZATION_SYSTEM_PROMPT
        )


class SentimentAnalysisDatasetGenerator(DatasetGenerator):
    """Sentiment Analysis Dataset Generator class."""

    def _set_dataset_type(self):
        """Set the dataset type to 'Sentiment Analysis Dataset'."""
        self.dataset.set_dataset_type("Sentiment Analysis Dataset")

    def _generate_entry(self, keyword: str):
        """
        Generate a sentiment analysis dataset entry for the given keyword.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._generate_entry(
            keyword,
            ENTRY_SENTIMENT_USER_PROMPT,
            ENTRY_SENTIMENT_SYSTEM_PROMPT,
            sentiment=random.choice(["positive", "negative", "neutral"]),
        )

    def _agenerate_entry(self, keyword: str):
        """
        Generate a sentiment analysis dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._agenerate_entry(
            keyword,
            ENTRY_SENTIMENT_USER_PROMPT,
            ENTRY_SENTIMENT_SYSTEM_PROMPT,
            sentiment=random.choice(["positive", "negative", "neutral"]),
        )


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
        return super()._generate_entry(
            keyword, ENTRY_CLASSIFICATION_USER_PROMPT, ENTRY_CLASSIFICATION_SYSTEM_PROMPT
        )

    def _agenerate_entry(self, keyword: str):
        """
        Generate a text classification dataset entry for the given keyword asynchronously.

        Args:
            keyword (str): The keyword for which to generate the entry.
        """
        return super()._agenerate_entry(
            keyword,
            ENTRY_CLASSIFICATION_USER_PROMPT,
            ENTRY_CLASSIFICATION_SYSTEM_PROMPT,
            label=random.choice(self.dataset.get_labels()),
        )
