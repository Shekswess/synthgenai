"""Dataset Module"""

import os
import random
import re
from typing import Dict, List

from datasets import Dataset as HFDataset
from huggingface_hub import (
    DatasetCard,
    create_repo,
    repo_exists,
    upload_file,
)
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from loguru import logger

from synthgenai.dataset.base_dataset import BaseDataset
from synthgenai.schemas.config import DatasetConfig
from synthgenai.schemas.enums import DatasetType
from synthgenai.utils.file_utils import FileUtils
from synthgenai.utils.progress_utils import ProgressManager
from synthgenai.utils.text_utils import TextUtils
from synthgenai.utils.yaml_utils import YamlUtils


class Dataset(BaseDataset):
    """Dataset Class"""

    def __init__(self, dataset_config: DatasetConfig):
        """
        Initialize the Dataset class.

        Args:
            dataset_config (DatasetConfig): The configuration for the dataset.
        """
        super().__init__(dataset_config)

    def get_topic(self) -> str:
        """
        Get the topic of the dataset.

        Returns:
            str: The topic of the dataset.
        """
        return self.topic

    def get_domains(self) -> List[str]:
        """
        Get the domains of the dataset.

        Returns:
            List[str]: The domains of the dataset.
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

    def set_keywords(self, keywords: List[str]):
        """
        Set the keywords for the dataset.

        Args:
            keywords (List[str]): The keywords for the dataset.
        """
        self.keywords = keywords

    def get_keywords(self) -> List[str]:
        """
        Get the keywords for the dataset.

        Returns:
            List[str]: The keywords for the dataset.
        """
        return self.keywords

    def set_data(self, data: List[Dict]) -> None:
        """
        Set the data for the dataset.

        Args:
            data (List[Dict]): The data for the dataset.
        """
        self.data = data

    def get_data(self) -> List[Dict]:
        """
        Get the data for the dataset.

        Returns:
            List[Dict]: The data for the dataset.
        """
        return self.data

    def set_labels(self, labels: List[str]) -> None:
        """
        Set the labels for the dataset.

        Args:
            labels (List[str]): The labels for the dataset.
        """
        self.labels = labels

    def get_labels(self) -> List[str]:
        """
        Get the labels for the dataset.

        Returns:
            List[str]: The labels for the dataset.
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

    def _get_hf_token(self, hf_token: str | None) -> str:
        """
        Get the Hugging Face token.

        Args:
            hf_token (str | None): The Hugging Face token for authentication.

        Returns:
            str: The Hugging Face token.

        Raises:
            ValueError: If HF_TOKEN is not provided and not found in
                environment variables.
        """
        logger.info("Retrieving Hugging Face token...")
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token is None:
                error_msg = (
                    "HF_TOKEN is not set. Please provide it via --hf-token "
                    "parameter or set the HF_TOKEN environment variable. "
                    "Get your token at: https://huggingface.co/settings/tokens"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            os.environ["HF_TOKEN"] = hf_token
        logger.info("Hugging Face token retrieved successfully")
        return hf_token

    def _upload_to_huggingface(
        self,
        hf_repo_name: str,
        hf_token: str | None,
        hf_dataset: HFDataset,
        dataset_path: str,
        markdown_description: str,
        content: str,
    ):
        """
        Upload dataset to Hugging Face Hub with proper error handling.

        Args:
            hf_repo_name: Repository name in format 'org/repo'
            hf_token: HuggingFace token
            hf_dataset: The dataset to upload
            dataset_path: Local path where dataset is saved
            markdown_description: Markdown description
            content: README content
        """
        try:
            # Validate repository name format
            if not re.match(r"^[^/]+/[^/]+$", hf_repo_name):
                error_msg = (
                    "hf_repo_name must be in the format "
                    "'organization_or_account/name_of_the_dataset'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Get HF token
            hf_token = self._get_hf_token(hf_token)

            # Check if repository exists and create if needed
            logger.info(f"Checking if repository {hf_repo_name} exists...")
            try:
                exists = repo_exists(repo_id=hf_repo_name, token=hf_token)
                if not exists:
                    logger.info(
                        f"Repository {hf_repo_name} does not exist, creating..."
                    )
                    create_repo(
                        repo_id=hf_repo_name,
                        token=hf_token,
                        repo_type="dataset",
                        exist_ok=True,
                    )
                    logger.info(f"Created new Hugging Face repository: {hf_repo_name}")
                else:
                    logger.info(f"Repository {hf_repo_name} already exists")
            except Exception as e:
                logger.error(f"Error checking/creating repository: {str(e)}")
                raise

            # Upload dataset
            logger.info("Uploading dataset to Hugging Face Hub...")
            try:
                hf_dataset.push_to_hub(
                    repo_id=hf_repo_name,
                    token=hf_token,
                    commit_message="Add dataset",
                )
                logger.info(f"Dataset uploaded successfully to {hf_repo_name}")
            except Exception as e:
                logger.error(f"Failed to upload dataset: {str(e)}")
                raise

            # Update README with metadata
            logger.info("Updating README with dataset metadata...")
            try:
                # Load existing dataset card or create new one
                try:
                    card = DatasetCard.load(
                        repo_id_or_path=hf_repo_name, token=hf_token
                    )
                    logger.info("Loaded existing dataset card")
                except (RepositoryNotFoundError, HfHubHTTPError):
                    logger.info("No existing dataset card found, creating new one")
                    card = DatasetCard("")

                card_metadata = YamlUtils.merge_metadata(
                    card.content, markdown_description
                )
                readme = card_metadata + "\n" + content

                # Save updated README locally first
                FileUtils.save_markdown(readme, os.path.join(dataset_path, "README.md"))

                # Upload README
                upload_file(
                    repo_id=hf_repo_name,
                    token=hf_token,
                    commit_message="Update README with dataset metadata",
                    path_or_fileobj=os.path.join(dataset_path, "README.md"),
                    path_in_repo="README.md",
                    repo_type="dataset",
                )
                logger.info(f"README.md uploaded successfully to {hf_repo_name}")
            except Exception as e:
                logger.error(f"Failed to update README: {str(e)}")
                # Don't raise here - dataset upload was successful,
                # README update is less critical
                logger.warning("Dataset upload completed but README update failed")

        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face Hub: {str(e)}")
            raise

    def save_dataset(
        self,
        dataset_path: str | None = None,
        hf_repo_name: str | None = None,
        hf_token: str | None = None,
    ):
        """
        Save the dataset to a local path and upload it to the Hugging Face Hub.

        Args:
            dataset_path (str | None): The file path to save the dataset to.
            hf_repo_name (str | None): The name of the Hugging Face repository
                to upload the dataset to. Format: 'organization/dataset_name'
            hf_token (str | None): The Hugging Face token for authentication.
        """
        try:
            # Create progress bar for dataset saving process
            total_steps = 4 if hf_repo_name is None else 5
            with ProgressManager.create_progress_bar(
                total=total_steps, desc="Saving dataset", unit="steps"
            ) as pbar:
                # Step 1: Prepare local save path
                pbar.set_description("Preparing local save path")
                logger.info("Preparing local save path...")
                dataset_path = self._prepare_local_save(dataset_path)
                pbar.update(1)

                # Step 2: Shuffle and create HF dataset
                pbar.set_description("Creating HuggingFace dataset")
                logger.info(
                    "Shuffling dataset and creating Hugging Face dataset object..."
                )
                random.shuffle(self.data)
                hf_dataset = HFDataset.from_list(self.data)
                logger.info(
                    f"Created Hugging Face dataset with {len(self.data)} entries"
                )
                pbar.update(1)

                # Step 3: Save dataset locally
                pbar.set_description("Saving to local disk")
                logger.info("Saving dataset to local disk...")
                hf_dataset.save_to_disk(os.path.join(dataset_path, "data"))
                logger.info(f"Dataset saved locally at {dataset_path}")
                pbar.update(1)

                # Step 4: Create README
                pbar.set_description("Creating README.md")
                logger.info("Creating README.md file...")
                markdown_description = TextUtils.convert_markdown(self.description)
                content = YamlUtils.extract_content(markdown_description)
                FileUtils.save_markdown(
                    content, os.path.join(dataset_path, "README.md")
                )
                logger.info("README.md file created successfully")
                pbar.update(1)

                # Step 5: Upload to Hugging Face Hub (if requested)
                if hf_repo_name is not None:
                    pbar.set_description("Uploading to HuggingFace Hub")
                    logger.info(
                        f"Starting Hugging Face Hub upload to {hf_repo_name}..."
                    )
                    self._upload_to_huggingface(
                        hf_repo_name,
                        hf_token,
                        hf_dataset,
                        dataset_path,
                        markdown_description,
                        content,
                    )
                    pbar.update(1)
                else:
                    logger.info(
                        "Skipping Hugging Face Hub upload (no repository specified)"
                    )

            ProgressManager.log_progress_complete(
                "Dataset saving", total_steps, total_steps
            )

            logger.info("Dataset save completed successfully!")

        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            raise
