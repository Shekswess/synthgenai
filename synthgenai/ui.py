"""The UI file for the SynthGenAI package."""

import os
import asyncio

import gradio as gr

from .data_model import DatasetConfig, DatasetGeneratorConfig, LLMConfig
from .dataset_generator import (
    InstructionDatasetGenerator,
    PreferenceDatasetGenerator,
    RawDatasetGenerator,
    SentimentAnalysisDatasetGenerator,
    SummarizationDatasetGenerator,
    TextClassificationDatasetGenerator,
)


def generate_synthetic_dataset(
    llm_model,
    temperature,
    top_p,
    max_tokens,
    api_base,
    api_key,
    dataset_type,
    topic,
    domains,
    language,
    additional_description,
    num_entries,
    hf_token,
    hf_repo_name,
    llm_env_vars,
):
    """
    Generate a dataset based on the provided parameters.

    Args:
        llm_model (str): The LLM model to use.
        temperature (float): The temperature for the LLM.
        top_p (float): The top_p value for the LLM.
        max_tokens (int): The maximum number of tokens for the LLM.
        api_base (str): The API base URL.
        api_key (str): The API key.
        dataset_type (str): The type of dataset to generate.
        topic (str): The topic of the dataset.
        domains (str): The domains for the dataset.
        language (str): The language of the dataset.
        additional_description (str): Additional description for the dataset.
        num_entries (int): The number of entries in the dataset.
        hf_token (str): The Hugging Face token.
        hf_repo_name (str): The Hugging Face repository name.
        llm_env_vars (str): Comma-separated environment variables for the LLM.

    Returns:
        str: A message indicating the result of the dataset generation.
    """
    os.environ["HF_TOKEN"] = hf_token

    for var in llm_env_vars.split(","):
        key, value = var.split("=")
        os.environ[key.strip()] = value.strip()

    if api_base and api_key:
        llm_config = LLMConfig(
            model=llm_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
        )
    else:
        llm_config = LLMConfig(
            model=llm_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    dataset_config = DatasetConfig(
        topic=topic,
        domains=domains.split(","),
        language=language,
        additional_description=additional_description,
        num_entries=num_entries,
    )

    dataset_generator_config = DatasetGeneratorConfig(
        llm_config=llm_config,
        dataset_config=dataset_config,
    )

    if dataset_type == "Raw":
        generator = RawDatasetGenerator(dataset_generator_config)
    elif dataset_type == "Instruction":
        generator = InstructionDatasetGenerator(dataset_generator_config)
    elif dataset_type == "Preference":
        generator = PreferenceDatasetGenerator(dataset_generator_config)
    elif dataset_type == "Sentiment Analysis":
        generator = SentimentAnalysisDatasetGenerator(dataset_generator_config)
    elif dataset_type == "Summarization":
        generator = SummarizationDatasetGenerator(dataset_generator_config)
    elif dataset_type == "Text Classification":
        generator = TextClassificationDatasetGenerator(dataset_generator_config)
    else:
        return "Invalid dataset type"

    dataset = asyncio.run(generator.agenerate_dataset())
    dataset.save_dataset(hf_repo_name=hf_repo_name)
    return "Dataset generated and saved successfully."


def ui_main():
    """
    Launch the Gradio UI for the SynthGenAI dataset generator.
    """
    with gr.Blocks(
        title="SynthGenAI Dataset Generator",
        css="footer {visibility: hidden}",
        theme="ParityError/Interstellar",
    ) as demo:
        gr.Markdown(
            """
            <div style="text-align: center;">
                <img src="https://raw.githubusercontent.com/Shekswess/synthgenai/refs/heads/main/docs/assets/logo_header.png" alt="Header Image" style="display: block; margin-left: auto; margin-right: auto; width: 50%;"/>
                <h1>SynthGenAI Dataset Generator</h1>
            </div>
            """
        )

        with gr.Row():
            llm_model = gr.Textbox(
                label="LLM Model", placeholder="model_provider/model_name"
            )
            temperature = gr.Slider(
                label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.5
            )
            top_p = gr.Slider(
                label="Top P", minimum=0.0, maximum=1.0, step=0.1, value=0.9
            )
            max_tokens = gr.Number(label="Max Tokens", value=2048)
            api_base = gr.Textbox(label="API Base", placeholder="API Base - Optional")
            api_key = gr.Textbox(
                label="API Key", placeholder="Your API Key - Optional", type="password"
            )

        with gr.Row():
            dataset_type = gr.Dropdown(
                label="Dataset Type",
                choices=[
                    "Raw",
                    "Instruction",
                    "Preference",
                    "Sentiment Analysis",
                    "Summarization",
                    "Text Classification",
                ],
            )
            topic = gr.Textbox(label="Topic", placeholder="Dataset topic")
            domains = gr.Textbox(label="Domains", placeholder="Comma-separated domains")
            language = gr.Textbox(
                label="Language", placeholder="Language", value="English"
            )
            additional_description = gr.Textbox(
                label="Additional Description",
                placeholder="Additional description",
                value="",
            )
            num_entries = gr.Number(label="Number of Entries", value=1000)

        with gr.Row():
            hf_token = gr.Textbox(
                label="Hugging Face Token",
                placeholder="Your HF Token",
                type="password",
                value=None,
            )
            hf_repo_name = gr.Textbox(
                label="Hugging Face Repo Name",
                placeholder="organization_or_user_name/dataset_name",
                value=None,
            )
            llm_env_vars = gr.Textbox(
                label="LLM Environment Variables",
                placeholder="Comma-separated environment variables (e.g., KEY1=VALUE1, KEY2=VALUE2)",
                value=None,
            )

        generate_button = gr.Button("Generate Dataset")
        output = gr.Textbox(label="Operation Result", value="")

        generate_button.click(
            generate_synthetic_dataset,
            inputs=[
                llm_model,
                temperature,
                top_p,
                max_tokens,
                api_base,
                api_key,
                dataset_type,
                topic,
                domains,
                language,
                additional_description,
                num_entries,
                hf_token,
                hf_repo_name,
                llm_env_vars,
            ],
            outputs=output,
        )

    demo.launch(inbrowser=True, favicon_path=None)
