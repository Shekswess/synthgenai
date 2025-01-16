"""Dataset Module"""

import os
import random
import re

from datasets import Dataset as HFDataset
from huggingface_hub import DatasetCard, create_repo, repo_exists, upload_file
from loguru import logger

from .data_model import DatasetConfig, DatasetType
from .utils import (
    convert_markdown,
    extract_content,
    merge_metadata,
    save_markdown,
)


class Dataset:
    """Dataset Class"""

    def __init__(self, dataset_config: DatasetConfig):
        """
        Initialize the Dataset class.

        Args:
            dataset_config (DatasetConfig): The configuration for the dataset.
        """
        self.topic = dataset_config.topic
        self.domains = dataset_config.domains
        self.language = dataset_config.language
        self.additional_description = dataset_config.additional_description
        self.num_keywords = dataset_config.num_entries
        self.type = None
        self.keywords = []
        self.labels = []
        self.data = []
        self.description = None

    def get_topic(self) -> str:
        """
        Get the topic of the dataset.

        Returns:
            str: The topic of the dataset.
        """
        return self.topic

    def get_domains(self) -> list[str]:
        """
        Get the domains of the dataset.

        Returns:
            list[str]: The domains of the dataset.
        """
        return self.domains

    def get_language(self) -> str:
        """
        Get the language of the dataset.

        Returns:
            str: The language of the dataset.
        """
        return self.language

    def get_additional_description(self) -> str:
        """
        Get the additional description of the dataset.

        Returns:
            str: The additional description of the dataset.
        """
        return self.additional_description

    def set_num_keywords(self, num_keywords: int):
        """
        Set the number of keywords for the dataset.

        Args:
            num_keywords (int): The number of keywords for the dataset.
        """
        self.num_keywords = num_keywords

    def get_num_keywords(self) -> int:
        """
        Get the number of keywords for the dataset.

        Returns:
            int: The number of keywords for the dataset.
        """
        return self.num_keywords

    def set_dataset_type(self, type: DatasetType):
        """
        Set the type of the dataset.

        Args:
            type (DatasetType): The type of the dataset.
        """
        self.type = type

    def get_dataset_type(self) -> DatasetType:
        """
        Get the type of the dataset.

        Returns:
            DatasetType: The type of the dataset.
        """
        return self.type

    def set_description(self, description: str):
        """
        Set the description of the dataset.

        Args:
            description (str): The description of the dataset.
        """
        self.description = description

    def get_description(self) -> str:
        """
        Get the description of the dataset.

        Returns:
            str: The description of the dataset.
        """
        return self.description

    def set_keywords(self, keywords: list[str]):
        """
        Set the keywords for the dataset.

        Args:
            keywords (list[str]): The keywords for the dataset.
        """
        self.keywords = keywords

    def get_keywords(self) -> list[str]:
        """
        Get the keywords for the dataset.

        Returns:
            list[str]: The keywords for the dataset.
        """
        return self.keywords

    def set_data(self, data: list[dict]) -> None:
        """
        Set the data for the dataset.

        Args:
            data (list[dict]): The data for the dataset.
        """
        self.data = data

    def get_data(self) -> list[dict]:
        """
        Get the data for the dataset.

        Returns:
            list[dict]: The data for the dataset.
        """
        return self.data

    def set_labels(self, labels: list[str]) -> None:
        """
        Set the labels for the dataset.

        Args:
            labels (list[str]): The labels for the dataset.
        """
        self.labels = labels

    def get_labels(self) -> list[str]:
        """
        Get the labels for the dataset.

        Returns:
            list[str]: The labels for the dataset.
        """
        return self.labels

    def _prepare_local_save(self, dataset_path: str) -> str:
        """
        Prepare the local save path.

        Args:
            dataset_path (str): The file path to save the dataset to.

        Returns:
            str: The local save path for the dataset.
        """
        if dataset_path is None:
            dataset_path = os.path.join(
                os.getcwd(), f"{self.topic.replace(' ', '_').lower()}_dataset"
            )
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "data"), exist_ok=True)
        logger.info(f"Prepared local save path at {dataset_path}")
        return dataset_path

    def _get_hf_token(self, hf_token: str) -> str:
        """
        Get the Hugging Face token.

        Args:
            hf_token (str): The Hugging Face token for authentication.

        Returns:
            str: The Hugging Face token.
        """
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token is None:
                logger.error("HF_TOKEN is not set")
                raise ValueError("HF_TOKEN is not set")
        else:
            os.environ["HF_TOKEN"] = hf_token
        logger.info("Hugging Face token retrieved")
        return hf_token

    def save_dataset(
        self, dataset_path: str = None, hf_repo_name: str = None, hf_token: str = None
    ):
        """
        Save the dataset to a local path and upload it to the Hugging Face Hub.

        Args:
            dataset_path (str): The file path to save the dataset to.
            hf_repo_name (str): The name of the Hugging Face repository to upload the dataset to. (format: 'organization_or_account/name_of_the_dataset')
            hf_token (str): The Hugging Face token for authentication.
        """
        try:
            dataset_path = self._prepare_local_save(dataset_path)
            random.shuffle(self.data)
            hf_dataset = HFDataset.from_list(mapping=self.data, split="train")
            hf_dataset.save_to_disk(os.path.join(dataset_path, "data"))
            logger.info(f"Dataset saved locally at {dataset_path}")

            markdown_description = convert_markdown(self.description)
            content = extract_content(markdown_description)
            save_markdown(content, os.path.join(dataset_path, "README.md"))
            logger.info("README.md file created")

            if hf_repo_name is not None:
                if not re.match(r"^[^/]+/[^/]+$", hf_repo_name):
                    logger.error(
                        "hf_repo_name must be in the format 'organization_or_account/name_of_the_dataset'"
                    )
                    raise ValueError(
                        "hf_repo_name must be in the format 'organization_or_account/name_of_the_dataset'"
                    )
                hf_token = self._get_hf_token(hf_token)
                if not repo_exists(repo_id=hf_repo_name):
                    create_repo(repo_id=hf_repo_name, token=hf_token, repo_type="dataset")
                    logger.info(f"Created new Hugging Face repo: {hf_repo_name}")

                hf_dataset.push_to_hub(
                    repo_id=hf_repo_name,
                    token=hf_token,
                    commit_message="Add dataset",
                )
                logger.info(f"Dataset pushed to Hugging Face Hub: {hf_repo_name}")

                card = DatasetCard.load(repo_id_or_path=hf_repo_name)
                card_metadata = merge_metadata(card.content, markdown_description)
                readme = card_metadata + "\n" + content
                save_markdown(readme, os.path.join(dataset_path, "README.md"))
                upload_file(
                    repo_id=hf_repo_name,
                    token=hf_token,
                    commit_message="Add README",
                    path_or_fileobj=os.path.join(dataset_path, "README.md"),
                    path_in_repo="README.md",
                    repo_type="dataset",
                )
                logger.info(f"README.md uploaded to Hugging Face Hub: {hf_repo_name}")

        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
