"""Example of generating a raw dataset using the Gemini LLM API and the Langfuse for tracing"""

import os

from synthgen import (
    DatasetConfig,
    DatasetGeneratorConfig,
    LLMConfig,
    RawDatasetGenerator,
)

os.environ["GEMINI_API_KEY"] = ""

os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

os.environ["HF_TOKEN"] = ""

if __name__ == "__main__":
    # Defining the LLM used for generating the dataset and the settings of the LLM
    llm_config = LLMConfig(model="gemini/gemini-1.5-flash-8b", temperature=0.5)

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

    # Defining the raw dataset generator based on the dataset generator configuration
    dataset_generator = RawDatasetGenerator(dataset_generator_config)

    # Generating the dataset
    dataset = dataset_generator.generate_dataset()

    # Saving the dataset to the locally and to the Hugging Face repository
    dataset.save_dataset(
        hf_repo_name=hf_repo_name,
    )