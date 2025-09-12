"""Unit tests for Dataset module."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from synthgenai.dataset.dataset import Dataset
from synthgenai.schemas.config import DatasetConfig
from synthgenai.schemas.enums import DatasetType


class TestDataset:
    """Test cases for Dataset class."""

    def test_initialization(self):
        """Test Dataset initialization."""
        config = DatasetConfig(
            topic="Test Topic",
            domains=["test", "domain"],
            language="English",
            additional_description="Test description",
            num_entries=100,
        )

        dataset = Dataset(config)

        assert dataset.get_topic() == "Test Topic"
        assert dataset.get_domains() == ["test", "domain"]
        assert dataset.get_language() == "English"
        assert dataset.get_additional_description() == "Test description"
        assert dataset.get_num_keywords() == 100

    def test_setters_and_getters(self):
        """Test all setter and getter methods."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)

        dataset = Dataset(config)

        # Test num_keywords
        dataset.set_num_keywords(200)
        assert dataset.get_num_keywords() == 200

        # Test dataset_type
        dataset.set_dataset_type(DatasetType.RAW)
        assert dataset.get_dataset_type() == DatasetType.RAW

        # Test description
        dataset.set_description("Test description")
        assert dataset.get_description() == "Test description"

        # Test keywords
        keywords = ["keyword1", "keyword2"]
        dataset.set_keywords(keywords)
        assert dataset.get_keywords() == keywords

        # Test data
        data = [{"text": "example1"}, {"text": "example2"}]
        dataset.set_data(data)
        assert dataset.get_data() == data

        # Test labels
        labels = ["label1", "label2"]
        dataset.set_labels(labels)
        assert dataset.get_labels() == labels

    def test_prepare_local_save_with_path(self):
        """Test local save path preparation with provided path."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_dataset")
            result_path = dataset._prepare_local_save(test_path)

            assert result_path == test_path
            assert os.path.exists(test_path)
            assert os.path.exists(os.path.join(test_path, "data"))

    def test_prepare_local_save_without_path(self):
        """Test local save path preparation without provided path."""
        config = DatasetConfig(topic="Test Topic", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        with (
            patch("os.getcwd", return_value="/tmp"),
            patch("os.makedirs") as mock_makedirs,
        ):
            result_path = dataset._prepare_local_save(None)

            expected_path = "/tmp/test_topic_dataset"
            assert result_path == expected_path

            # Check that directories were created
            assert mock_makedirs.call_count == 2

    def test_get_hf_token_from_parameter(self):
        """Test getting HF token from parameter."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        token = dataset._get_hf_token("test_token")
        assert token == "test_token"
        assert os.environ.get("HF_TOKEN") == "test_token"

    def test_get_hf_token_from_env(self):
        """Test getting HF token from environment."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            token = dataset._get_hf_token(None)
            assert token == "env_token"

    def test_get_hf_token_missing(self):
        """Test error when HF token is missing."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HF_TOKEN is not set"):
                dataset._get_hf_token(None)

    @patch("synthgenai.dataset.dataset.HFDataset")
    @patch("synthgenai.dataset.dataset.FileUtils")
    @patch("synthgenai.dataset.dataset.TextUtils")
    @patch("synthgenai.dataset.dataset.YamlUtils")
    def test_save_dataset_local_only(self, mock_yaml, mock_text, mock_file, mock_hf):
        """Test saving dataset locally only."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        # Set up test data
        dataset.set_data([{"text": "test1"}, {"text": "test2"}])
        dataset.set_description("Test dataset description")

        # Mock the HF dataset
        mock_dataset_instance = Mock()
        mock_hf.from_list.return_value = mock_dataset_instance

        # Mock text and yaml utils
        mock_text.convert_markdown.return_value = "markdown content"
        mock_yaml.extract_content.return_value = "extracted content"

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset.save_dataset(dataset_path=temp_dir)

            # Verify HF dataset was created (order may vary due to shuffle)
            mock_hf.from_list.assert_called_once()
            call_args = mock_hf.from_list.call_args[0][0]
            assert len(call_args) == 2
            assert {"text": "test1"} in call_args
            assert {"text": "test2"} in call_args
            mock_dataset_instance.save_to_disk.assert_called_once()

            # Verify markdown file was saved
            mock_file.save_markdown.assert_called_once()

    @patch("synthgenai.dataset.dataset.upload_file")
    @patch("synthgenai.dataset.dataset.create_repo")
    @patch("synthgenai.dataset.dataset.repo_exists")
    @patch("synthgenai.dataset.dataset.HFDataset")
    def test_upload_to_huggingface_new_repo(
        self, mock_hf, mock_repo_exists, mock_create_repo, mock_upload
    ):
        """Test uploading to HuggingFace with new repository."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        # Mock repository doesn't exist
        mock_repo_exists.return_value = False

        # Mock dataset
        mock_dataset_instance = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset._upload_to_huggingface(
                "user/test-repo",
                "test_token",
                mock_dataset_instance,
                temp_dir,
                "markdown",
                "content",
            )

            # Verify repo was created
            mock_create_repo.assert_called_once_with(
                repo_id="user/test-repo",
                token="test_token",
                repo_type="dataset",
                exist_ok=True,
            )

            # Verify dataset was uploaded
            mock_dataset_instance.push_to_hub.assert_called_once()

    def test_upload_to_huggingface_invalid_repo_name(self):
        """Test upload fails with invalid repository name."""
        config = DatasetConfig(topic="Test", domains=["test"], num_entries=50)
        dataset = Dataset(config)

        with pytest.raises(ValueError, match="hf_repo_name must be in the format"):
            dataset._upload_to_huggingface(
                "invalid-name",  # No slash
                "token",
                Mock(),
                "/tmp",
                "markdown",
                "content",
            )
