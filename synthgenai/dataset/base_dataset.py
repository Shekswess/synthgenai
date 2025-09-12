"""Base Dataset Module"""

from abc import ABC, abstractmethod
from typing import Dict, List

from synthgenai.schemas.config import DatasetConfig
from synthgenai.schemas.enums import DatasetType


class BaseDataset(ABC):
    """Abstract Base Dataset Class"""

    def __init__(self, dataset_config: DatasetConfig):
        """
        Initialize the BaseDataset class.

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

    @abstractmethod
    def get_topic(self) -> str:
        """
        Get the topic of the dataset.

        Returns:
            str: The topic of the dataset.
        """
        pass

    @abstractmethod
    def get_domains(self) -> List[str]:
        """
        Get the domains of the dataset.

        Returns:
            List[str]: The domains of the dataset.
        """
        pass

    @abstractmethod
    def get_language(self) -> str:
        """
        Get the language of the dataset.

        Returns:
            str: The language of the dataset.
        """
        pass

    @abstractmethod
    def get_additional_description(self) -> str:
        """
        Get the additional description of the dataset.

        Returns:
            str: The additional description of the dataset.
        """
        pass

    @abstractmethod
    def set_num_keywords(self, num_keywords: int):
        """
        Set the number of keywords for the dataset.

        Args:
            num_keywords (int): The number of keywords for the dataset.
        """
        pass

    @abstractmethod
    def get_num_keywords(self) -> int:
        """
        Get the number of keywords for the dataset.

        Returns:
            int: The number of keywords for the dataset.
        """
        pass

    @abstractmethod
    def set_dataset_type(self, type: DatasetType):
        """
        Set the type of the dataset.

        Args:
            type (DatasetType): The type of the dataset.
        """
        pass

    @abstractmethod
    def get_dataset_type(self) -> DatasetType:
        """
        Get the type of the dataset.

        Returns:
            DatasetType: The type of the dataset.
        """
        pass

    @abstractmethod
    def set_description(self, description: str):
        """
        Set the description of the dataset.

        Args:
            description (str): The description of the dataset.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get the description of the dataset.

        Returns:
            str: The description of the dataset.
        """
        pass

    @abstractmethod
    def set_keywords(self, keywords: List[str]):
        """
        Set the keywords for the dataset.

        Args:
            keywords (List[str]): The keywords for the dataset.
        """
        pass

    @abstractmethod
    def get_keywords(self) -> List[str]:
        """
        Get the keywords for the dataset.

        Returns:
            List[str]: The keywords for the dataset.
        """
        pass

    @abstractmethod
    def set_data(self, data: List[Dict]) -> None:
        """
        Set the data for the dataset.

        Args:
            data (List[dict]): The data for the dataset.
        """
        pass

    @abstractmethod
    def get_data(self) -> List[Dict]:
        """
        Get the data for the dataset.

        Returns:
            List[Dict]: The data for the dataset.
        """
        pass

    @abstractmethod
    def set_labels(self, labels: List[str]) -> None:
        """
        Set the labels for the dataset.

        Args:
            labels (List[str]): The labels for the dataset.
        """
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """
        Get the labels for the dataset.

        Returns:
            List[str]: The labels for the dataset.
        """
        pass

    @abstractmethod
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
        pass
