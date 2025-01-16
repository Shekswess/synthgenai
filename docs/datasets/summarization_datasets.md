# Summarization Datasets 🧾

To generate a Summarization dataset, you need to use the `SummarizationDatasetGenerator` class.

```python
from synthgenai import SummarizationDatasetGenerator
```

Example of generated entry for the summarization dataset:

```json
{
  "keyword": "keyword",
  "topic": "topic",
  "language": "language",
  "generated_entry": {
    "text": "generated text",
    "summary": "generated summary"
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
    SummarizationDatasetGenerator,
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

# Creating the SummarizationDatasetGenerator
summarization_dataset_generator = SummarizationDatasetGenerator(dataset_generator_config)

# Generating the dataset
summarization_dataset = summarization_dataset_generator.generate_dataset()
```

## Asynchronous Generation 🔀

```python
import os
import asyncio
from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    SummarizationDatasetGenerator,
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

# Creating the SummarizationDatasetGenerator
summarization_dataset_generator = SummarizationDatasetGenerator(dataset_generator_config)

# Generating the dataset asynchronously
summarization_dataset = asyncio.run(summarization_dataset_generator.agenerate_dataset())
```
