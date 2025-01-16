# SynthGenAI-Package for Generating Synthetic Datasets using LLMs

![header_logo_image](./docs/assets/logo_header.png)

SynthGenAI is a package for generating Synthetic Datasets. The idea is to have a tool which is simple to use and can generate datasets on different topics by utilizing LLMs from different API providers. The package is designed to be modular and can be easily extended to include some different API providers for LLMs and new features.

> [!IMPORTANT]
> The package is still in the early stages of development and some features may not be fully implemented or tested. If you find any issues or have any suggestions, feel free to open an issue or create a pull request.

## Why SynthGenAI now? 🤔

Interest in synthetic data generation has surged recently, driven by the growing recognition of data as a critical asset in AI development. As Ilya Sutskever, one of the most important figures in AI, says: 'Data is the fossil fuel of AI.' The more quality data we have, the better our models can perform. However, access to data is often restricted due to privacy concerns, or it may be prohibitively expensive to collect. Additionally, the vast amount of high-quality data on the internet has already been extensively mined. Synthetic data generation addresses these challenges by allowing us to create diverse and useful datasets using current pre-trained Large Language Models (LLMs). Beyond LLMs, synthetic data also holds immense potential for training and fine-tuning Small Language Models (SLMs), which are gaining popularity due to their efficiency and suitability for specific, resource-constrained applications. By leveraging synthetic data for both LLMs and SLMs, we can enhance performance across a wide range of use cases while balancing resource efficiency and model effectiveness. This approach enables us to harness the strengths of both synthetic and authentic datasets to achieve optimal outcomes.

## Tools used for building SynthGenAI 🧰

The package is built using Python and the following libraries:

- [uv](https://docs.astral.sh/uv/), An extremely fast Python package and project manager, written in Rust.
- [litellm](https://docs.litellm.ai/docs/), A Python SDK for accessing LLMs from different API providers with standardized OpenAI Format.
- [langfuse](https://langfuse.com/), LLMOps platform for observability, tracebility and monitoring of LLMs.
- [pydantic](https://pydantic-docs.helpmanual.io/), Data validation and settings management using Python type annotations.
- [huggingface-hub](https://huggingface.co/) & [datasets](https://huggingface.co/docs/datasets/), A Python library for saving generated datasets on Hugging Face Hub.

## Installation 🛠️

To install the package, you can use the following command:

```bash
pip install synthgenai
```

or you can install the package directly from the source code using the following commands:

```bash
git clone https://github.com/Shekswess/synthgenai.git
uv build
pip install ./dist/synthgenai-{version}-py3-none-any.whl
```

### Requirements 📋

To use the package, you need to have the following requirements installed:

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) for building the package directly from the source code
- [Ollama](https://ollama.com/) running on your local machine if you want to use Ollama as an API provider (optional)
- [Langfuse](https://langfuse.com/) running on your local machine or in the cloud if you want to use Langfuse for tracebility (optional)
- [Hugging Face Hub](https://huggingface.co/) account if you want to save the generated datasets on Hugging Face Hub with generated token (optional)

## Usage 👨‍💻

The available API providers for LLMs are:

- **Groq**
- **Mistral AI**
- **Gemini**
- **Bedrock**
- **Anthropic**
- **OpenAI**
- **Hugging Face**
- **Ollama**
- **vLLM**
- **SageMaker**
- **Azure**
- **Vertex AI**

For observing the generated datasets, you can use **Langfuse** for tracebility and monitoring of the LLMs.

To use the LLMs from different API providers, to observe the generated datasets, and to save the generated datasets on Hugging Face Hub, you need to set the following environment variables:

```
# API keys for different LLM providers
GROQ_API_KEY=
MISTRAL_API_KEY=
GEMINI_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
AWS_PROFILE=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
AZURE_API_KEY
AZURE_API_BASE
AZURE_API_VERSION
AZURE_AD_TOKEN
AZURE_API_TYPE
GOOGLE_APPLICATION_CREDENTIALS
VERTEXAI_LOCATION
VERTEXAI_PROJECT
HUGGINGFACE_API_KEY=

# Langfuse API keys
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

# Huggingface token for uploading datasets on Huggingface
HF_TOKEN=
```

Currently there are three types of datasets that can be generated using SynthGenAI:

- **Raw Datasets**
- **Instruction Datasets**
- **Preference Datasets**
- **Sentiment Analysis Datasets**
- **Summarization Datasets**
- **Text Classification Datasets**

The datasets can be generated:

- **Synchronously** - each dataset entry is generated one by one
- **Asynchronously** - batch of dataset entries is generated at once

> [!NOTE]
> Asynchronous generation is faster than synchronous generation, but some of LLM providers can have limitations on the number of tokens that can be generated at once.

#### Raw Datasets 🥩

To generate a raw dataset, you can use the following code:

```python
# For asynchronous dataset generation
# import asyncio
import os

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    RawDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Optional for Langfuse Tracebility
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

# Optional for Hugging Face Hub upload
os.environ["HF_TOKEN"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name", # Check liteLLM docs for more info
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

# Generating the dataset asynchronously
# raw_dataset = asyncio.run(raw_dataset_generator.agenerate_dataset())

# Name of the Hugging Face repository where the dataset will be saved
hf_repo_name = "organization_or_user_name/dataset_name" # optional

# Saving the dataset to the locally and to the Hugging Face repository(optional)
raw_dataset.save_dataset(
    hf_repo_name=hf_repo_name,
)
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

#### Instruction Datasets 💬

To generate an instruction dataset, you can use the following code:

```python
# For asynchronous dataset generation
# import asyncio
import os

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    InstructionDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Optional for Langfuse Tracebility
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

# Optional for Hugging Face Hub upload
os.environ["HF_TOKEN"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name", # Check liteLLM docs for more info
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

# Generating the dataset asynchronously
# instruction_dataset = asyncio.run(instruction_dataset_generator.agenerate_dataset())

# Name of the Hugging Face repository where the dataset will be saved
hf_repo_name = "organization_or_user_name/dataset_name" # optional

# Saving the dataset to the locally and to the Hugging Face repository(optional)
instruction_dataset.save_dataset(
    hf_repo_name=hf_repo_name,
)
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

#### Preference Datasets 🌟

To generate a preference dataset, you can use the following code:

```python
# For asynchronous dataset generation
# import asyncio
import os

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    PreferenceDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Optional for Langfuse Tracebility
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

# Optional for Hugging Face Hub upload
os.environ["HF_TOKEN"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name", # Check liteLLM docs for more info
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

# Generating the dataset asynchronously
# preference_dataset = asyncio.run(preference_dataset_generator.agenerate_dataset())

# Name of the Hugging Face repository where the dataset will be saved
hf_repo_name = "organization_or_user_name/dataset_name" # optional

# Saving the dataset to the locally and to the Hugging Face repository(optional)
preference_dataset.save_dataset(
    hf_repo_name=hf_repo_name,
)
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

#### Sentiment Analysis Datasets 🎭

To generate a sentiment analysis dataset, you can use the following code:

```python
# For asynchronous dataset generation
# import asyncio
import os

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    SentimentAnalysisDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Optional for Langfuse Tracebility
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

# Optional for Hugging Face Hub upload
os.environ["HF_TOKEN"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name", # Check liteLLM docs for more info
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

# Generating the dataset asynchronously
# sentiment_analysis_dataset = asyncio.run(sentiment_analysis_dataset_generator.agenerate_dataset())

# Name of the Hugging Face repository where the dataset will be saved
hf_repo_name = "organization_or_user_name/dataset_name" # optional

# Saving the dataset to the locally and to the Hugging Face repository(optional)
sentiment_analysis_dataset.save_dataset(
    hf_repo_name=hf_repo_name,
)
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

#### Text Classification Datasets 🔠

To generate a text classification dataset, you can use the following code:

```python
# For asynchronous dataset generation
# import asyncio
import os

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    TextClassificationDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Optional for Langfuse Tracebility
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

# Optional for Hugging Face Hub upload
os.environ["HF_TOKEN"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name", # Check liteLLM docs for more info
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

# Generating the dataset asynchronously
# text_classification_dataset = asyncio.run(text_classification_dataset_generator.agenerate_dataset())

# Name of the Hugging Face repository where the dataset will be saved
hf_repo_name = "organization_or_user_name/dataset_name" # optional

# Saving the dataset to the locally and to the Hugging Face repository(optional)
text_classification_dataset.save_dataset(
    hf_repo_name=hf_repo_name,
)
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

#### Summarization Datasets 🧾

To generate a summarization dataset, you can use the following code:

```python
# For asynchronous dataset generation
# import asyncio
import os

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    SummarizationDatasetGenerator,
)

# Setting the API keys
os.environ["LLM_API_KEY"] = ""

# Optional for Langfuse Tracebility
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

# Optional for Hugging Face Hub upload
os.environ["HF_TOKEN"] = ""

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name", # Check liteLLM docs for more info
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

# Generating the dataset asynchronously
# summarization_dataset = asyncio.run(summarization_dataset_generator.agenerate_dataset())

# Name of the Hugging Face repository where the dataset will be saved
hf_repo_name = "organization_or_user_name/dataset_name" # optional

# Saving the dataset to the locally and to the Hugging Face repository(optional)
summarization_dataset.save_dataset(
    hf_repo_name=hf_repo_name,
)
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

#### More Examples 📖

More examples with different combinations of LLM API providers and dataset configurations can be found in the [examples](./examples) directory.

> [!IMPORTANT]
> Sometimes the generation of the keywords for the dataset and the dataset entries can fail due to the limitation of the LLM to generate JSON Object as output (this is handled by the package). That's why it is recommended to use models that are capable of generating JSON Objects (structured output). List of models that can generate JSON Objects can be found [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

## Generated Datasets 📚

Examples of generated synthetic datasets can be found on the [SynthGenAI Datasets Collection](https://huggingface.co/collections/Shekswess/synthgenai-datasets-6764ad878718b1e567653022) on Hugging Face Hub.

## Supported API Providers 💪

- [x] [Groq](https://groq.com/) - more info about Groq models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/groq)
- [x] [Mistral AI](https://mistral.ai/) - more info about Mistral AI models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/mistral-ai)
- [x] [Gemini](https://gemini.google.com/) - more info about Gemini models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/gemini)
- [x] [Bedrock](https://aws.amazon.com/bedrock) - more info about Bedrock models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/bedrock)
- [x] [Anthropic](https://www.anthropic.com/) - more info about Anthropic models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/anthropic)
- [x] [OpenAI](https://openai.com) - more info about OpenAI models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/openai)
- [x] [Hugging Face](https://huggingface.co/) - more info about Hugging Face models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/hugging-face)
- [x] [Ollama](https://ollama.com/) - more info about Ollama models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/ollama)
- [x] [vLLM](https://vllm.ai/) - more info about vLLM models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/vllm)
- [x] [SageMaker](https://aws.amazon.com/sagemaker/) - more info about SageMaker models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/aws_sagemaker)
- [x] [Azure](https://azure.microsoft.com/en-us/services/machine-learning/) - more info about Azure and Azure AI models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/azure) & [here](https://docs.litellm.ai/docs/providers/azure_ai)
- [x] [Vertex AI](https://cloud.google.com/vertex-ai) - more info about Vertex AI models that can be used, can be found [here](https://docs.litellm.ai/docs/providers/vertex)

## Next Steps 🚀

- [ ] Add CLI or TUI or UI for generating datasets

## Contributing 🤝

If you want to contribute to this project and make it better, your help is very welcome. Create a pull request with your changes and I will review it. If you have any questions, open an issue.

## License 📝

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Repo Structure 📂

```
.
├── .github
│   └── workflows
│       ├── build_n_release.yml
│       └── tests.yml
├── assets
│   └── logo_header.png
├── examples
│   ├── anthropic_instruction_dataset_example.py
│   ├── azure_ai_preference_dataset_example.py
│   ├── azure_summarization_dataset_example.py
│   ├── bedrock_raw_dataset_example.py
│   ├── gemini_langfuse_raw_dataset_example.py
│   ├── groq_preference_dataset_example.py
│   ├── huggingface_instruction_dataset_example.py
│   ├── mistral_preference_dataset_example.py
│   ├── ollama_preference_dataset_example.py
│   ├── openai_raw_dataset_example.py
│   ├── sagemaker_summarization_dataset_example.py
│   ├── vertex_ai_text_classification_dataset_example.py
│   └── vllm_sentiment_analysis_dataset_example.py
├── synthgenai
│   ├── data_model.py
│   ├── dataset_generator.py
│   ├── dataset.py
│   ├── __init__.py
│   ├── llm.py
│   ├── prompts.py
│   └── utils.py
├── tests
│   ├── test_dataset.py
│   └── test_llm.py
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```
