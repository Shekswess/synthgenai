"""Example of generating a sentiment analysis dataset using the VLLM API."""

# For asynchronous dataset generation
# import asyncio
import os

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    SentimentAnalysisDatasetGenerator,
)

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
# os.environ["AWS_PROFILE"] = ""  # Optional if you are using a profile and not keys

os.environ["HF_TOKEN"] = ""

if __name__ == "__main__":
    # Defining the LLM used for generating the dataset and the settings of the LLM
    llm_config = LLMConfig(
        model="hosted_vllm/facebook/opt-125m",
        api_base="https://hosted-vllm-api.co",  # change to the api base for your hosted vLLM model
        temperature=0.5,
    )

    # Defining the dataset configuration, the topic of the dataset, the domains, the language, the additional description, and the number of entries
    dataset_config = DatasetConfig(
        topic="Artificial Intelligence",
        domains=["Machine Learning", "Deep Learning"],
        language="English",
        additional_description="This dataset must be more focused on healthcare implementations of AI, Machine Learning, and Deep Learning.",
        num_entries=1000,
    )

    # Defining the dataset Hugging Face repository name
    hf_repo_name = "{organization_or_account_name}/artificial-intelligence-in-healthcare"

    # Defining the dataset generator configuration
    dataset_generator_config = DatasetGeneratorConfig(
        dataset_config=dataset_config, llm_config=llm_config
    )

    # Defining the sentiment analysis dataset generator based on the dataset generator configuration
    dataset_generator = SentimentAnalysisDatasetGenerator(dataset_generator_config)

    # Generating the dataset
    dataset = dataset_generator.generate_dataset()

    # Generating the dataset asynchronously
    # dataset = asyncio.run(dataset_generator.agenerate_dataset())

    # Saving the dataset to the locally and to the Hugging Face repository
    dataset.save_dataset(
        hf_repo_name=hf_repo_name,
    )
