# Preference Datasets 🌟

To generate a Preference dataset, you need to use the `PreferenceDatasetGenerator` class.

```python
from synthgenai import PreferenceDatasetGenerator
```

Example of generated entry for the preference dataset:

```json
{
  "keyword": "keyword",
  "topic": "topic",
  "language": "language",
  "generated_entry": {
    "prompt": [
      { "role": "system", "content": "generated system(instruction) prompt" },
      { "role": "user", "content": "generated user prompt" }
    ],
    "chosen": [
      { "role": "assistant", "content": "generated chosen assistant response" }
    ],
    "rejected": [
      {
        "role": "assistant",
        "content": "generated rejected assistant response"
      }
    ]
  }
}
```

## Synchronous Generation 🔁

```python
import os
from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    PreferenceDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name",
    temperature=0.5,
    top_p=0.9,
    max_tokens=2048,
)

# Creating the DatasetConfig
dataset_config = DatasetConfig(
    topic="topic_name",
    domains=["domain1", "domain2"],
    language="English",
    additional_description="Additional description",
    num_entries=1000
)

# Creating the DatasetGeneratorConfig
dataset_generator_config = DatasetGeneratorConfig(
    llm_config=llm_config,
    dataset_config=dataset_config,
)

# Creating the PreferenceDatasetGenerator
preference_dataset_generator = PreferenceDatasetGenerator(dataset_generator_config)

# Generating the dataset
preference_dataset = preference_dataset_generator.generate_dataset()
```

## Asynchronous Generation 🔀

```python
import os
import asyncio
from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    PreferenceDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name",
    temperature=0.5,
    top_p=0.9,
    max_tokens=2048,
)

# Creating the DatasetConfig
dataset_config = DatasetConfig(
    topic="topic_name",
    domains=["domain1", "domain2"],
    language="English",
    additional_description="Additional description",
    num_entries=1000
)

# Creating the DatasetGeneratorConfig
dataset_generator_config = DatasetGeneratorConfig(
    llm_config=llm_config,
    dataset_config=dataset_config,
)

# Creating the PreferenceDatasetGenerator
preference_dataset_generator = PreferenceDatasetGenerator(dataset_generator_config)

# Generating the dataset asynchronously
preference_dataset = asyncio.run(preference_dataset_generator.agenerate_dataset())
```
