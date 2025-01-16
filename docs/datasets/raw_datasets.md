# Raw Datasets 🥩

To generate a Raw dataset, you need to use the `RawDatasetGenerator` class.

```python
from synthgenai import RawDatasetGenerator
```

Example of generated entry for the raw dataset:

```json
{
  "keyword": "keyword",
  "topic": "topic",
  "language": "language",
  "generated_entry": {
    "text": "generated text"
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
    RawDatasetGenerator,
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

# Creating the RawDatasetGenerator
raw_dataset_generator = RawDatasetGenerator(dataset_generator_config)

# Generating the dataset
raw_dataset = raw_dataset_generator.generate_dataset()
```

## Asynchronous Generation 🔀

```python
import os
import asyncio
from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    RawDatasetGenerator,
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

# Creating the RawDatasetGenerator
raw_dataset_generator = RawDatasetGenerator(dataset_generator_config)

# Generating the dataset asynchronously
raw_dataset = asyncio.run(raw_dataset_generator.agenerate_dataset())
```
