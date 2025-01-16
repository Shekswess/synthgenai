# Instruction Datasets 💬

To generate an Instruction dataset, you need to use the `InstructionDatasetGenerator` class. 

```python
from synthgenai import InstructionDatasetGenerator
```

Example of generated entry for the instruction dataset:

```json
{
  "keyword": "keyword",
  "topic": "topic",
  "language": "language",
  "generated_entry": {
    "messages": [
      {
        "role": "system",
        "content": "generated system(instruction) prompt"
      },
      {
        "role": "user",
        "content": "generated user prompt"
      },
      {
        "role": "assistant",
        "content": "generated assistant prompt"
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
    InstructionDatasetGenerator,
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

# Creating the InstructionDatasetGenerator
instruction_dataset_generator = InstructionDatasetGenerator(dataset_generator_config)

# Generating the dataset
instruction_dataset = instruction_dataset_generator.generate_dataset()
```

## Asynchronous Generation 🔀

```python
import os
import asyncio
from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    InstructionDatasetGenerator,
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

# Creating the InstructionDatasetGenerator
instruction_dataset_generator = InstructionDatasetGenerator(dataset_generator_config)

# Generating the dataset asynchronously
instruction_dataset = asyncio.run(instruction_dataset_generator.agenerate_dataset())
```
