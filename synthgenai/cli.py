"""SynthGenAI CLI for generating synthetic datasets."""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from synthgenai import (
    DatasetConfig,
    DatasetGeneratorConfig,
    InstructionDatasetGenerator,
    LLMConfig,
    PreferenceDatasetGenerator,
    RawDatasetGenerator,
    SentimentAnalysisDatasetGenerator,
    SummarizationDatasetGenerator,
    TextClassificationDatasetGenerator,
)

# Load environment variables from .env file if present
load_dotenv()

# Typer app
app = typer.Typer(
    name="synthgenai",
    help="SynthGenAI - CLI for generating Synthetic Datasets using LLMs.",
)

# Rich console
console = Console()

# Dataset type to generator mapping
DATASET_GENERATORS = {
    "raw": RawDatasetGenerator,
    "instruction": InstructionDatasetGenerator,
    "preference": PreferenceDatasetGenerator,
    "sentiment": SentimentAnalysisDatasetGenerator,
    "summarization": SummarizationDatasetGenerator,
    "classification": TextClassificationDatasetGenerator,
}

# Common LLM provider examples for help text
LLM_EXAMPLES = """OpenAI: openai/gpt-5"""


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    if verbose:
        logger.add(
            "synthgenai.log",
            rotation="10 MB",
            level="DEBUG",
            format="{time} | {level} | {message}",
        )


def validate_domains(domains: List[str]) -> List[str]:
    """Validate and clean domain list."""
    if not domains:
        raise typer.BadParameter("At least one domain must be provided")
    return [domain.strip() for domain in domains if domain.strip()]


def create_llm_config(
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMConfig:
    """Create LLM configuration."""
    config_dict = {"model": model}

    if temperature is not None:
        config_dict["temperature"] = temperature
    if top_p is not None:
        config_dict["top_p"] = top_p
    if max_tokens is not None:
        config_dict["max_tokens"] = max_tokens
    if api_base is not None:
        config_dict["api_base"] = api_base
    if api_key is not None:
        config_dict["api_key"] = api_key

    return LLMConfig(**config_dict)


def create_dataset_config(
    topic: str,
    domains: List[str],
    language: str,
    additional_description: str,
    num_entries: int,
) -> DatasetConfig:
    """Create dataset configuration."""
    return DatasetConfig(
        topic=topic,
        domains=validate_domains(domains),
        language=language,
        additional_description=additional_description,
        num_entries=num_entries,
    )


@app.command()
def generate(
    # Required arguments
    dataset_type: str = typer.Argument(
        help="Type of dataset to generate. Options: "
        + ", ".join(DATASET_GENERATORS.keys())
    ),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help=f"LLM model to use for generation, e.g: {LLM_EXAMPLES}",
    ),
    topic: str = typer.Option(
        ...,
        "--topic",
        "-t",
        help="Main topic/theme of the dataset",
    ),
    domains: List[str] = typer.Option(
        ...,
        "--domain",
        "-d",
        help="Domain(s) for the dataset. Can be specified multiple times.",
    ),
    num_entries: Annotated[
        int,
        typer.Option(
            "--entries",
            "-n",
            help="Number of entries to generate",
            min=10,
        ),
    ] = 1000,
    language: Annotated[
        str,
        typer.Option(
            "--language",
            "-l",
            help="Language for the dataset",
        ),
    ] = "English",
    additional_description: Annotated[
        str,
        typer.Option(
            "--description",
            "-desc",
            help="Additional description for the dataset",
        ),
    ] = "",
    # LLM Configuration
    temperature: Annotated[
        Optional[float],
        typer.Option(
            "--temperature",
            help="Temperature for LLM generation (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = None,
    top_p: Annotated[
        Optional[float],
        typer.Option(
            "--top-p",
            help="Top-p value for nucleus sampling (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = None,
    max_tokens: Annotated[
        Optional[int],
        typer.Option(
            "--max-tokens",
            help="Maximum tokens for completions (minimum 1000)",
            min=1000,
        ),
    ] = None,
    api_base: Annotated[
        Optional[str],
        typer.Option(
            "--api-base",
            help="Custom API base URL",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            help="API key (can also be set via environment variables)",
        ),
    ] = None,
    # Dataset saving options
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Local output directory for the dataset",
        ),
    ] = None,
    hf_repo: Annotated[
        Optional[str],
        typer.Option(
            "--hf-repo",
            help="Hugging Face repository name (format: org/repo-name)",
        ),
    ] = None,
    hf_token: Annotated[
        Optional[str],
        typer.Option(
            "--hf-token",
            help="Hugging Face token (via HF_TOKEN env var)",
        ),
    ] = None,
    # Generation options
    async_generation: Annotated[
        bool,
        typer.Option(
            "--async/--sync",
            help="Use asynchronous generation for faster processing",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
):
    """
    Generate a synthetic dataset using the specified configuration.
    """
    setup_logging(verbose)

    # Validate dataset type
    if dataset_type not in DATASET_GENERATORS:
        console.print(
            f"[bold red]❌ Invalid dataset type:[/] {dataset_type}\n"
            f"[yellow]Available types:[/] {', '.join(DATASET_GENERATORS.keys())}"
        )
        raise typer.Exit(1)

    # Early validation for HuggingFace requirements
    if hf_repo and not hf_token and not os.environ.get("HF_TOKEN"):
        console.print(
            Panel.fit(
                "[bold red]❌ HuggingFace token required when using --hf-repo!\n"
                "[white]Provide via:\n"
                "  • --hf-token YOUR_TOKEN\n"
                "  • HF_TOKEN environment variable\n"
                "Get your token at: https://huggingface.co/settings/tokens",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    if verbose:
        console.print(
            Panel.fit(
                f"🚀 Starting [cyan]{dataset_type.capitalize()}[/] dataset generation",
                border_style="cyan",
            )
        )
        console.print(f"[green]📊 Topic:[/] {topic}")
        console.print(f"[green]🏷️  Domains:[/] {', '.join(domains)}")
        console.print(f"[green]🔢 Entries:[/] {num_entries}")
        console.print(f"[green]🤖 Model:[/] {model}")
        console.print(f"[green]🌐 Language:[/] {language}")

    try:
        # Create configurations
        llm_config = create_llm_config(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
        )

        dataset_config = create_dataset_config(
            topic=topic,
            domains=domains,
            language=language,
            additional_description=additional_description,
            num_entries=num_entries,
        )

        generator_config = DatasetGeneratorConfig(
            dataset_config=dataset_config,
            llm_config=llm_config,
        )

        # Create dataset generator
        generator_class = DATASET_GENERATORS[dataset_type]
        generator = generator_class(generator_config)

        console.print("[yellow]⚡ Generating dataset...[/]")
        if async_generation:
            if verbose:
                console.print("[blue]🔄 Using asynchronous generation...[/]")
            dataset = asyncio.run(generator.agenerate_dataset())
        else:
            if verbose:
                console.print("[blue]🔄 Using synchronous generation...[/]")
            dataset = generator.generate_dataset()

        console.print("[yellow]💾 Saving dataset...[/]")
        try:
            output_path_str = str(output_path) if output_path else None
            dataset.save_dataset(
                dataset_path=output_path_str,
                hf_repo_name=hf_repo,
                hf_token=hf_token,
            )
        except Exception as e:
            console.print(f"[bold red]❌ Failed to save dataset:[/] {e}")
            raise typer.Exit(1)

        console.print(
            Panel.fit(
                "[bold green]✅ Dataset generated successfully![/]",
                border_style="green",
            )
        )
        if output_path:
            console.print(f"📁 [cyan]Local path:[/] {output_path}")
        if hf_repo:
            hf_url = f"https://huggingface.co/datasets/{hf_repo}"
            console.print(f"🤗 [magenta]Hugging Face:[/] {hf_url}")

    except Exception as e:
        console.print(f"[bold red]❌ Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def list_types():
    """List all available dataset types."""
    console.print(
        Panel.fit("📚 [bold cyan]Available dataset types[/]", border_style="cyan")
    )

    dataset_descriptions = {
        "raw": "Generate unstructured text data around topics and keywords",
        "instruction": "Generate instruction-following conversations",
        "preference": "Generate preference data with chosen/rejected",
        "sentiment": "Generate text with sentiment labels",
        "summarization": "Generate text summarization pairs (text -> summary)",
        "classification": "Generate text classification data with labels",
    }

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Dataset Type", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")

    for dtype, description in dataset_descriptions.items():
        table.add_row(dtype, description)

    console.print(table)


@app.command()
def examples():
    """Show example commands for different use cases."""
    console.print(
        Panel.fit("💡 [bold yellow]Example Commands[/]", border_style="yellow")
    )

    examples = [
        {
            "title": "Basic instruction dataset with OpenAI",
            "command": (
                'synthgenai generate instruction --model "openai/gpt-5" '
                '--topic "Programming" --domain "Python" --entries 100'
            ),
        },
        {
            "title": "Preference dataset with Anthropic Claude",
            "command": (
                "synthgenai generate preference "
                '--model "anthropic/claude-sonnet-4-20250514" '
                '--topic "Creative Writing" --domain "Fiction" '
                "--temperature 0.8 --entries 100"
            ),
        },
    ]

    table = Table(
        show_header=True, header_style="bold blue", box=box.MINIMAL_DOUBLE_HEAD
    )
    table.add_column("Use Case", style="magenta", no_wrap=True)
    table.add_column("Command", style="cyan")

    for ex in examples:
        table.add_row(ex["title"], ex["command"])

    console.print(table)


@app.command()
def providers():
    """Show supported LLM providers and example model names."""
    console.print(
        Panel.fit("🤖 [bold cyan]Supported LLM Providers[/]", border_style="cyan")
    )

    providers = [
        "OpenAI",
        "Anthropic",
        "Gemini",
        "Vertex AI",
        "Groq",
        "Mistral",
        "Bedrock",
        "Ollama",
        "Hugging Face",
        "xAI",
        "DeepSeek",
        "vLLM",
        "SageMaker",
        "Azure (OpenAI)",
        "OpenRouter",
    ]

    table = Table(show_header=False, box=box.SIMPLE)
    for provider in providers:
        table.add_row(f"📡 [green]{provider}[/]")
    console.print(table)


@app.command()
def version():
    """Show version information."""
    console.print(
        Panel.fit(
            "[bold blue]SynthGenAI v2.0.0[/]", border_style="blue", title="🚀 Version"
        )
    )


@app.command()
def env_setup():
    """Show required environment variables and API keys for providers."""
    console.print(
        Panel.fit(
            "🔑 [bold yellow]Environment Variables Setup Guide[/]",
            border_style="yellow",
        )
    )

    env_vars = [
        ("OPENAI_API_KEY", "OpenAI GPT models", "https://platform.openai.com/api-keys"),
        (
            "ANTHROPIC_API_KEY",
            "Anthropic Claude models",
            "https://console.anthropic.com/",
        ),
        (
            "GEMINI_API_KEY",
            "Google Gemini models",
            "https://aistudio.google.com/app/apikey",
        ),
        ("GROQ_API_KEY", "Groq models", "https://console.groq.com/keys"),
        (
            "MISTRAL_API_KEY",
            "Mistral AI models",
            "https://console.mistral.ai/api-keys/",
        ),
    ]

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Variable", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Get Key", style="yellow")

    for var, desc, url in env_vars:
        table.add_row(var, desc, url)

    console.print(table)


@app.callback()
def main():
    """
    SynthGenAI - Generate synthetic datasets using various LLM providers.

    Quick Start:
      1. Run 'synthgenai env-setup' to see required environment variables
      2. Set up API keys for your chosen LLM provider(s)
      3. Use 'synthgenai generate' to create your dataset
    """
    pass


if __name__ == "__main__":
    app()
