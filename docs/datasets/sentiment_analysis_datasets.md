# Sentiment Analysis Datasets 🎭

To generate a Sentiment Analysis dataset, you need to use the `SentimentAnalysisDatasetGenerator` class.

```python
from synthgenai import SentimentAnalysisDatasetGenerator
```

Example of generated entry for the sentiment analysis dataset:

```json
{
  "keyword": "keyword",
  "topic": "topic",
  "language": "language",
  "generated_entry": {
    "prompt": "generated text",
    "label": "generated sentiment (which can be positive, negative, neutral)"
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
    SentimentAnalysisDatasetGenerator,
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

# Creating the SentimentAnalysisDatasetGenerator
sentiment_analysis_dataset_generator = SentimentAnalysisDatasetGenerator(dataset_generator_config)

# Generating the dataset
sentiment_analysis_dataset = sentiment_analysis_dataset_generator.generate_dataset()
```

## Asynchronous Generation 🔀

```python
import os
import asyncio
from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    SentimentAnalysisDatasetGenerator,
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

# Creating the SentimentAnalysisDatasetGenerator
sentiment_analysis_dataset_generator = SentimentAnalysisDatasetGenerator(dataset_generator_config)

# Generating the dataset asynchronously
sentiment_analysis_dataset = asyncio.run(sentiment_analysis_dataset_generator.agenerate_dataset())
```
