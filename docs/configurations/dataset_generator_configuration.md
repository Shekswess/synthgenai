# Dataset Generator Configuration 🏭

To configure the dataset generator, you need to create a `DatasetGeneratorConfig` object. This object contains the configuration for both the dataset and the LLM model.

## Example 📖

```python
from synthgenai import DatasetConfig, LLMConfig, DatasetGeneratorConfig

# Creating the DatasetConfig
dataset_config = DatasetConfig(
    topic="topic_name",
    domains=["domain1", "domain2"],
    language="English",
    additional_description="Additional description",
    num_entries=1000
)

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name",
    temperature=0.5,
    top_p=0.9,
    max_tokens=2048,
    api_base="https://api.example.com",
    api_key="your_api_key"
)

# Creating the DatasetGeneratorConfig
dataset_generator_config = DatasetGeneratorConfig(
    dataset_config=dataset_config,
    llm_config=llm_config
)
```

## Parameters 🎛

- `dataset_config` (DatasetConfig): The configuration for the dataset. (Required)
- `llm_config` (LLMConfig): The configuration for the LLM. (Required)

For more information on configuring the dataset generator, refer to the [SynthGenAI documentation](https://github.com/Shekswess/synthgenai).
