# Dataset Configuration 📚

To configure the datasets for generation, you need to create a `DatasetConfig` object. This object contains the configuration for the dataset, including the topic, domains, language, additional description, and the number of entries.

## Example 📖

```python
from synthgenai import DatasetConfig

# Creating the DatasetConfig
dataset_config = DatasetConfig(
    topic="topic_name",
    domains=["domain1", "domain2"],
    language="English",
    additional_description="Additional description",
    num_entries=1000
)
```

## Parameters 🎛

- `topic` (str): The topic of the dataset. (Required)
- `domains` (list[str]): A list of domains related to the dataset. Must contain at least one item. (Required)
- `language` (str): The language of the dataset. Default is "English". (Optional)
- `additional_description` (str): Any additional description for the dataset. Maximum length is 1000 characters. (Optional, default: "")
- `num_entries` (int): The number of entries to generate. Must be greater than 1. (Optional, default: 1000)

For more information on configuring datasets, refer to the [SynthGenAI documentation](https://github.com/Shekswess/synthgenai).
