"""Unit tests for DatasetGenerator module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from synthgenai.dataset_genetors.dataset_generator import DatasetGenerator
from synthgenai.schemas.config import DatasetConfig, DatasetGeneratorConfig, LLMConfig


class TestDatasetGenerator:
    """Test cases for DatasetGenerator class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        dataset_config = DatasetConfig(
            topic="Test Topic",
            domains=["test", "domain"],
            language="English",
            num_entries=50,
        )
        llm_config = LLMConfig(model="gpt-4")

        return DatasetGeneratorConfig(
            dataset_config=dataset_config, llm_config=llm_config
        )

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    def test_initialization(self, mock_dataset, mock_llm, mock_config):
        """Test DatasetGenerator initialization."""
        generator = DatasetGenerator(mock_config)

        assert generator.dataset is not None
        assert generator.llm is not None
        assert generator.prompt_utils is not None

        # Verify Dataset and LLM were initialized with correct configs
        mock_dataset.assert_called_once_with(mock_config.dataset_config)
        mock_llm.assert_called_once_with(mock_config.llm_config)

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    def test_create_messages(self, mock_dataset, mock_llm, mock_config):
        """Test message creation for LLM interaction."""
        generator = DatasetGenerator(mock_config)

        # Mock prompt_utils
        generator.prompt_utils.get_prompt = Mock(
            side_effect=["System prompt content", "User prompt content"]
        )

        messages = generator._create_messages(
            "test_system_prompt", "test_user_prompt", keyword="test_keyword"
        )

        expected_messages = [
            {"role": "system", "content": "System prompt content"},
            {"role": "user", "content": "User prompt content"},
        ]

        assert messages == expected_messages
        assert generator.prompt_utils.get_prompt.call_count == 2

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    def test_set_dataset_type_not_implemented(
        self, mock_dataset, mock_llm, mock_config
    ):
        """Test that _set_dataset_type raises NotImplementedError."""
        generator = DatasetGenerator(mock_config)

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement _set_dataset_type"
        ):
            generator._set_dataset_type()

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    def test_generate_entry_not_implemented(self, mock_dataset, mock_llm, mock_config):
        """Test that _generate_entry raises NotImplementedError."""
        generator = DatasetGenerator(mock_config)

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement _generate_entry"
        ):
            generator._generate_entry("test_keyword")

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    @pytest.mark.asyncio
    async def test_agenerate_entry_not_implemented(
        self, mock_dataset, mock_llm, mock_config
    ):
        """Test that _agenerate_entry raises NotImplementedError."""
        generator = DatasetGenerator(mock_config)

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement _agenerate_entry"
        ):
            await generator._agenerate_entry("test_keyword")

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    def test_get_entry_response_format(self, mock_dataset, mock_llm, mock_config):
        """Test getting entry response format for different dataset types."""
        generator = DatasetGenerator(mock_config)
        generator.dataset.get_dataset_type = Mock()

        # Test Raw Dataset
        generator.dataset.get_dataset_type.return_value = "Raw Dataset"
        format_class = generator._get_entry_response_format()
        assert format_class.__name__ == "EntryRawDataset"

        # Test Instruction Dataset
        generator.dataset.get_dataset_type.return_value = "Instruction Dataset"
        format_class = generator._get_entry_response_format()
        assert format_class.__name__ == "EntryInstructionDataset"

        # Test unknown dataset type
        generator.dataset.get_dataset_type.return_value = "Unknown Dataset"
        format_class = generator._get_entry_response_format()
        assert format_class is None

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    @patch("synthgenai.dataset_genetors.dataset_generator.time")
    def test_generate_dataset(self, mock_time, mock_dataset, mock_llm, mock_config):
        """Test complete dataset generation."""
        generator = DatasetGenerator(mock_config)

        # Mock time
        mock_time.time.side_effect = [0, 10]  # Start and end times

        # Mock the required methods
        generator._generate_keywords = Mock()
        generator._generate_entries = Mock()
        generator._generate_description = Mock()
        generator.dataset.get_data = Mock(return_value=[{"text": "test"}])
        generator.dataset.get_num_keywords = Mock(return_value=1)

        result = generator.generate_dataset()

        # Verify all steps were called
        generator._generate_keywords.assert_called_once()
        generator._generate_entries.assert_called_once()
        generator._generate_description.assert_called_once()

        # Verify the dataset is returned
        assert result == generator.dataset

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    @patch("synthgenai.dataset_genetors.dataset_generator.time")
    @pytest.mark.asyncio
    async def test_agenerate_dataset(
        self, mock_time, mock_dataset, mock_llm, mock_config
    ):
        """Test complete async dataset generation."""
        generator = DatasetGenerator(mock_config)

        # Mock time
        mock_time.time.side_effect = [0, 10]  # Start and end times

        # Mock the required methods
        generator._generate_keywords = Mock()
        generator._agenerate_entries = AsyncMock()
        generator._agenerate_description = AsyncMock()
        generator.dataset.get_data = Mock(return_value=[{"text": "test"}])
        generator.dataset.get_num_keywords = Mock(return_value=1)

        result = await generator.agenerate_dataset()

        # Verify all steps were called
        generator._generate_keywords.assert_called_once()
        generator._agenerate_entries.assert_called_once()
        generator._agenerate_description.assert_called_once()

        # Verify the dataset is returned
        assert result == generator.dataset

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    def test_generate_description(self, mock_dataset, mock_llm, mock_config):
        """Test description generation."""
        generator = DatasetGenerator(mock_config)

        # Mock LLM methods
        generator.llm.get_temperature = Mock(return_value=0.7)
        generator.llm.set_response_format = Mock()
        generator.llm.set_temperature = Mock()
        generator.llm.generate = Mock(return_value="Generated description")
        generator.llm.get_model = Mock(return_value="gpt-4")

        # Mock dataset methods
        generator.dataset.get_topic = Mock(return_value="Test Topic")
        generator.dataset.get_domains = Mock(return_value=["domain1", "domain2"])
        generator.dataset.get_language = Mock(return_value="English")
        generator.dataset.get_additional_description = Mock(return_value="Additional")
        generator.dataset.get_num_keywords = Mock(return_value=10)
        generator.dataset.get_dataset_type = Mock(return_value="Raw Dataset")
        generator.dataset.set_description = Mock()

        # Mock prompt utils
        generator.prompt_utils.get_prompt = Mock(
            side_effect=["System prompt", "User prompt"]
        )

        generator._generate_description()

        # Verify description was set
        generator.dataset.set_description.assert_called_once_with(
            "Generated description"
        )

        # Verify temperature was restored
        generator.llm.set_temperature.assert_called_with(0.7)

    @patch("synthgenai.dataset_genetors.dataset_generator.LLM")
    @patch("synthgenai.dataset_genetors.dataset_generator.Dataset")
    @pytest.mark.asyncio
    async def test_agenerate_description(self, mock_dataset, mock_llm, mock_config):
        """Test async description generation."""
        generator = DatasetGenerator(mock_config)

        # Mock LLM methods
        generator.llm.get_temperature = Mock(return_value=0.7)
        generator.llm.set_response_format = Mock()
        generator.llm.set_temperature = Mock()
        generator.llm.agenerate = AsyncMock(return_value="Generated async description")
        generator.llm.get_model = Mock(return_value="gpt-4")

        # Mock dataset methods
        generator.dataset.get_topic = Mock(return_value="Test Topic")
        generator.dataset.get_domains = Mock(return_value=["domain1", "domain2"])
        generator.dataset.get_language = Mock(return_value="English")
        generator.dataset.get_additional_description = Mock(return_value="Additional")
        generator.dataset.get_num_keywords = Mock(return_value=10)
        generator.dataset.get_dataset_type = Mock(return_value="Raw Dataset")
        generator.dataset.set_description = Mock()

        # Mock prompt utils
        generator.prompt_utils.get_prompt = Mock(
            side_effect=["System prompt", "User prompt"]
        )

        await generator._agenerate_description()

        # Verify description was set
        generator.dataset.set_description.assert_called_once_with(
            "Generated async description"
        )

        # Verify temperature was restored
        generator.llm.set_temperature.assert_called_with(0.7)
