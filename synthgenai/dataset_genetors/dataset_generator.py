"""Dataset Generator module."""

import asyncio
import os
import random
import time

from loguru import logger
from pydantic import ValidationError

from synthgenai.dataset.dataset import Dataset
from synthgenai.llm.llm import LLM
from synthgenai.schemas.config import DatasetGeneratorConfig
from synthgenai.schemas.datasets import (
    EntryInstructionDataset,
    EntryKeywords,
    EntryLabels,
    EntryPreferenceDataset,
    EntryRawDataset,
    EntrySentimentDataset,
    EntrySummaryDataset,
    EntryTextClassificationDataset,
)
from synthgenai.utils.json_utils import JsonUtils
from synthgenai.utils.progress_utils import ProgressManager
from synthgenai.utils.prompt_utils import PromptUtils

if os.environ.get("SYNTHGENAI_DETAILED_MODE", "true").lower() == "true":
    logger.remove()

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
        self.prompt_utils = PromptUtils()
        logger.info("Initialized Dataset Generator")

    def _create_messages(
        self, system_prompt_name: str, user_prompt_name: str, **kwargs
    ) -> list[dict]:
        """
        Create messages for LLM interaction.

        Args:
            system_prompt_name (str): The name of the system prompt template.
            user_prompt_name (str): The name of the user prompt template.
            **kwargs: Additional keyword arguments to be formatted into the prompts.

        Returns:
            list[dict]: The messages for LLM interaction.
        """
        logger.debug("Creating messages for LLM interaction")
        system_prompt = self.prompt_utils.get_prompt(system_prompt_name, **kwargs)
        user_prompt = self.prompt_utils.get_prompt(user_prompt_name, **kwargs)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _generate_keywords(self):
        """Generate keywords for the dataset."""
        logger.info("Starting keyword generation")
        try:
            if self.llm.check_response_format():
                self.llm.set_response_format(EntryKeywords)

            keywords = []
            num_keywords = self.dataset.get_num_keywords()
            max_retries = 3
            retry_count = 0

            with ProgressManager.create_progress_bar(
                total=num_keywords, desc="Generating keywords", unit="keywords"
            ) as pbar:
                while len(keywords) < num_keywords and retry_count < max_retries:
                    try:
                        remaining_keywords = min(num_keywords - len(keywords), 30)
                        additional_description = (
                            self.dataset.get_additional_description()
                        )
                        if keywords:
                            additional_description += (
                                f" Previously generated keywords: {', '.join(keywords)}. "
                                "\nGenerate new keywords following the provided rules in "
                                "the system prompt."
                            )
                        messages = self._create_messages(
                            system_prompt_name="keyword_system_prompt",
                            user_prompt_name="keyword_user_prompt",
                            topic=self.dataset.get_topic(),
                            domains=", ".join(self.dataset.get_domains()),
                            language=self.dataset.get_language(),
                            additional_description=additional_description,
                            num_keywords=remaining_keywords,
                        )
                        logger.info(f"Generating {remaining_keywords} keywords...")
                        response = self.llm.generate(messages)
                        logger.debug(f"Keyword generation response: {response}")

                        if not self.llm.check_response_format():
                            try:
                                response = JsonUtils.convert_keywords_labels(response)
                            except ValueError as e:
                                logger.error(f"Invalid JSON response: {e}, retrying...")
                                retry_count += 1
                                continue
                        else:
                            try:
                                response = JsonUtils.convert_keywords_labels(response)
                            except ValueError as e:
                                logger.error(f"Invalid JSON response: {e}, retrying...")
                                retry_count += 1
                                continue

                        new_keywords = response.get("keywords", [])
                        if not new_keywords:
                            logger.warning("No new keywords generated, retrying...")
                            retry_count += 1
                            continue

                        keywords.extend(new_keywords)
                        pbar.update(len(new_keywords))
                        pbar.set_postfix(total=len(keywords))
                        logger.debug(
                            f"Generated {len(new_keywords)} new keywords, "
                            f"{len(keywords)} total keywords"
                        )
                        retry_count = 0  # Reset retry count on success

                    except Exception as e:
                        logger.error(f"Error generating keywords: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Max retries reached for keyword generation")
                            raise

            if len(keywords) == 0:
                raise ValueError("Failed to generate any keywords")

            if len(keywords) > num_keywords:
                logger.warning("More keywords generated than required, truncating")
                keywords = keywords[:num_keywords]

            try:
                keywords = EntryKeywords(keywords=keywords)
            except ValidationError as e:
                logger.error(f"Validation error for keywords: {e}")
                raise e

            self.dataset.set_keywords(keywords.keywords)
            ProgressManager.log_progress_complete(
                "Keyword generation", len(keywords.keywords), num_keywords
            )

        except Exception as e:
            logger.error(f"Failed to generate keywords: {e}")
            raise

    def _generate_labels(self):
        """Generate labels for the (text classification) dataset."""
        logger.info("Generating labels for the dataset")
        if self.llm.check_response_format():
            self.llm.set_response_format(EntryLabels)

        labels = []
        num_labels = random.randint(2, 5)

        with ProgressManager.create_progress_bar(
            total=num_labels, desc="Generating labels", unit="labels"
        ) as pbar:
            while len(labels) < num_labels:
                remaining_labels = min(num_labels - len(labels), 30)
                additional_description = self.dataset.get_additional_description()
                if labels:
                    additional_description += (
                        f" Previously generated labels: {', '.join(labels)}. "
                        "\nGenerate new labels following the provided rules in "
                        "the system prompt."
                    )
                messages = self._create_messages(
                    system_prompt_name="labels_system_prompt",
                    user_prompt_name="labels_user_prompt",
                    topic=self.dataset.get_topic(),
                    domains=", ".join(self.dataset.get_domains()),
                    language=self.dataset.get_language(),
                    additional_description=additional_description,
                    num_labels=remaining_labels,
                )
                response = self.llm.generate(messages)
                logger.debug(f"Label generation response: {response}")

                if not self.llm.check_response_format():
                    try:
                        response = JsonUtils.convert_keywords_labels(response)
                    except ValueError as e:
                        logger.error(f"Invalid JSON response: {e}, retrying...")
                        continue
                else:
                    try:
                        response = JsonUtils.convert_keywords_labels(response)
                    except ValueError as e:
                        logger.error(f"Invalid JSON response: {e}, retrying...")
                        continue

                new_labels = response.get("labels", [])
                if not new_labels:
                    logger.warning("No new labels generated, breaking the loop")
                    break

                labels.extend(new_labels)
                pbar.update(len(new_labels))
                pbar.set_postfix(total=len(labels))
                logger.debug(
                    f"Generated {len(new_labels)} new labels, "
                    f"{len(labels)} total labels"
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
        ProgressManager.log_progress_complete(
            "Label generation", len(labels.labels), num_labels
        )

    def _generate_description(self):
        """Generate a description for the dataset."""
        logger.info("Generating dataset description")
        tmp_llm_temperature = self.llm.get_temperature()
        self.llm.set_response_format(None)
        self.llm.set_temperature(None)
        messages = self._create_messages(
            system_prompt_name="description_system_prompt",
            user_prompt_name="description_user_prompt",
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

    async def _agenerate_description(self):
        """Generate a description for the dataset asynchronously."""
        logger.info("Generating dataset description asynchronously")
        tmp_llm_temperature = self.llm.get_temperature()
        self.llm.set_response_format(None)
        self.llm.set_temperature(None)
        messages = self._create_messages(
            system_prompt_name="description_system_prompt",
            user_prompt_name="description_user_prompt",
            topic=self.dataset.get_topic(),
            domains=", ".join(self.dataset.get_domains()),
            language=self.dataset.get_language(),
            additional_description=self.dataset.get_additional_description(),
            num_keywords=self.dataset.get_num_keywords(),
            dataset_type=self.dataset.get_dataset_type(),
            model=self.llm.get_model(),
        )
        response = await self.llm.agenerate(messages)
        self.dataset.set_description(response)
        self.llm.set_temperature(tmp_llm_temperature)
        logger.info("Dataset description generated asynchronously")

    def _generate_entry(self, keyword: str):
        """
        Generate an entry for the dataset. Must be implemented by subclasses.

        Args:
            keyword (str): The keyword for which to generate the entry.

        Returns:
            dict: The generated dataset entry.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _generate_entry")

    async def _agenerate_entry(self, keyword: str):
        """
        Generate an entry for the dataset asynchronously. Must be implemented by subclasses.

        Args:
            keyword (str): The keyword for which to generate the entry.

        Returns:
            dict: The generated dataset entry.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _agenerate_entry")

    def _generate_entry_with_prompts(
        self, system_prompt: str, user_prompt: str, keyword: str, **kwargs
    ):
        """
        Generate an entry for the dataset using specific prompt templates.

        Args:
            system_prompt (str): The system prompt template name for the entry.
            user_prompt (str): The user prompt template name for the entry.
            keyword (str): The keyword for which to generate the entry.
            **kwargs: Additional keyword arguments to be formatted into the prompts.

        Returns:
            dict: The generated dataset entry.

        Raises:
            ValidationError: If the generated entry does not match the data model.
        """
        messages = self._create_messages(
            system_prompt_name=system_prompt,
            user_prompt_name=user_prompt,
            keyword=keyword,
            dataset_type=self.dataset.get_dataset_type(),
            topic=self.dataset.get_topic(),
            language=self.dataset.get_language(),
            additional_description=self.dataset.get_additional_description(),
            **kwargs,
        )
        response = self.llm.generate(messages)

        if not self.llm.check_response_format():
            try:
                response = JsonUtils.convert_entry(response)
                if not response:  # convert_entry returns empty dict on error
                    logger.error(f"Invalid JSON response for keyword {keyword}")
                    return None
            except Exception as e:
                logger.error(f"JSON parsing error for keyword {keyword}: {e}")
                return None

        try:
            if self.llm.check_response_format():
                response = JsonUtils.convert_entry(response)
            entry_format = self._get_entry_response_format()
            entry = entry_format(**response)
            logger.debug(f"Dataset entry: {entry}")
        except ValidationError as e:
            logger.error(f"Validation error for keyword {keyword}: {e}")
            return None
        return entry.model_dump()

    async def _agenerate_entry_with_prompts(
        self, system_prompt: str, user_prompt: str, keyword: str, **kwargs
    ):
        """
        Generate an entry for the dataset asynchronously using specific prompt templates.

        Args:
            system_prompt (str): The system prompt template name for the entry.
            user_prompt (str): The user prompt template name for the entry.
            keyword (str): The keyword for which to generate the entry.
            **kwargs: Additional keyword arguments to be formatted into the prompts.

        Returns:
            dict: The generated dataset entry.

        Raises:
            ValidationError: If the generated entry does not match the data model.
        """
        messages = self._create_messages(
            system_prompt_name=system_prompt,
            user_prompt_name=user_prompt,
            keyword=keyword,
            dataset_type=self.dataset.get_dataset_type(),
            topic=self.dataset.get_topic(),
            language=self.dataset.get_language(),
            additional_description=self.dataset.get_additional_description(),
            **kwargs,
        )
        response = await self.llm.agenerate(messages)

        if not self.llm.check_response_format():
            try:
                response = JsonUtils.convert_entry(response)
                if not response:  # convert_entry returns empty dict on error
                    logger.error(f"Invalid JSON response for keyword {keyword}")
                    return None
            except Exception as e:
                logger.error(f"JSON parsing error for keyword {keyword}: {e}")
                return None

        try:
            if self.llm.check_response_format():
                response = JsonUtils.convert_entry(response)
            entry_format = self._get_entry_response_format()
            entry = entry_format(**response)
            logger.debug(f"Dataset entry: {entry}")
        except ValidationError as e:
            logger.error(f"Validation error for keyword {keyword}: {e}")
            return None
        return entry.model_dump()

    def _set_dataset_type(self):
        """Set the dataset type. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _set_dataset_type")

    def _get_entry_response_format(self):
        """Get the appropriate response format for the current dataset type."""

        dataset_type = self.dataset.get_dataset_type()
        dataset_type_mapping = {
            "Raw Dataset": EntryRawDataset,
            "Instruction Dataset": EntryInstructionDataset,
            "Preference Dataset": EntryPreferenceDataset,
            "Summarization Dataset": EntrySummaryDataset,
            "Sentiment Analysis Dataset": EntrySentimentDataset,
            "Text Classification Dataset": EntryTextClassificationDataset,
            "Reasoning Dataset": EntryReasoningDataset,
        }

        return dataset_type_mapping.get(dataset_type)

    async def _agenerate_entries(self):
        """Generate entries for the dataset asynchronously."""
        logger.info("Generating entries for dataset asynchronously")
        self._set_dataset_type()
        logger.info(f"Dataset type: {self.dataset.get_dataset_type()}")

        if self.llm.check_response_format():
            entry_format = self._get_entry_response_format()
            self.llm.set_response_format(entry_format)

        if self.dataset.get_dataset_type() == "Text Classification Dataset":
            self._generate_labels()

        data = []
        keywords = self.dataset.get_keywords()
        total_batches = (len(keywords) + BATCH_SIZE - 1) // BATCH_SIZE

        with ProgressManager.create_progress_bar(
            total=len(keywords), desc="Generating entries (async)", unit="entries"
        ) as pbar:
            for i in range(0, len(keywords), BATCH_SIZE):
                batch_keywords = keywords[i : i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                pbar.set_description(f"Processing batch {batch_num}/{total_batches}")

                tasks = [self._agenerate_entry(keyword) for keyword in batch_keywords]
                entries = await asyncio.gather(*tasks)
                await asyncio.sleep(10)

                batch_success = 0
                for keyword, entry in zip(batch_keywords, entries):
                    if entry:
                        data.append(entry)
                        batch_success += 1
                        logger.debug(f"Generated entry for keyword: {keyword}")
                    else:
                        logger.warning(
                            f"Skipping entry for keyword: {keyword} due to validation error"
                        )
                    pbar.update(1)

                pbar.set_postfix(
                    batch=f"{batch_num}/{total_batches}",
                    success=len(data),
                    batch_success=batch_success,
                )

        self.dataset.set_data(data)
        success_rate = len(data) / len(keywords) if keywords else 0
        ProgressManager.log_progress_complete(
            "Async entry generation", len(data), len(keywords), success_rate
        )

    def _generate_entries(self):
        """Generate entries for the dataset."""
        logger.info("Generating entries for dataset")
        self._set_dataset_type()
        logger.info(f"Dataset type: {self.dataset.get_dataset_type()}")

        if self.llm.check_response_format():
            entry_format = self._get_entry_response_format()
            self.llm.set_response_format(entry_format)

        if self.dataset.get_dataset_type() == "Text Classification Dataset":
            self._generate_labels()

        data = []
        keywords = self.dataset.get_keywords()

        # Create progress bar for entry generation
        with ProgressManager.create_progress_bar(
            total=len(keywords), desc="Generating entries", unit="entries"
        ) as pbar:
            for keyword in keywords:
                entry = self._generate_entry(keyword)
                if entry:
                    data.append(entry)
                    pbar.set_postfix(
                        success=len(data), failed=pbar.pbar.n + 1 - len(data)
                    )
                    logger.debug(f"Generated entry for keyword: {keyword}")
                else:
                    logger.warning(
                        f"Skipping entry for keyword: {keyword} due to validation error"
                    )
                pbar.update(1)

        self.dataset.set_data(data)
        success_rate = len(data) / len(keywords) if keywords else 0
        ProgressManager.log_progress_complete(
            "Entry generation", len(data), len(keywords), success_rate
        )

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

        await self._agenerate_description()

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time taken to generate dataset: {total_time:.2f} seconds")
        return self.dataset
