# Text Classification Datasets 🔠

To generate a Text Classification dataset, you need to use the `TextClassificationDatasetGenerator` class.

```python
from synthgenai import TextClassificationDatasetGenerator
```

Example of generated entry for the text classification dataset:

```json
{
  "keyword": "keyword",
  "topic": "topic",
  "language": "language",
  "generated_entry": {
    "prompt": "generated text",
    "label": "generated sentiment (which will be from a list of labels, created from the model)"
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
    TextClassificationDatasetGenerator,
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

# Creating the TextClassificationDatasetGenerator
text_classification_dataset_generator = TextClassificationDatasetGenerator(dataset_generator_config)

# Generating the dataset
text_classification_dataset = text_classification_dataset_generator.generate_dataset()
```

## Asynchronous Generation 🔀

```python
import os
import asyncio
from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    TextClassificationDatasetGenerator,
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

# Creating the TextClassificationDatasetGenerator
text_classification_dataset_generator = TextClassificationDatasetGenerator(dataset_generator_config)

# Generating the dataset asynchronously
text_classification_dataset = asyncio.run(text_classification_dataset_generator.agenerate_dataset())
```
