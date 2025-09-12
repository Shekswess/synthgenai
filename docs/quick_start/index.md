# Quick Start 🚀

Get up and running with SynthGenAI CLI in minutes! This guide covers installation, setup, and your first dataset generation.

## Installation 📦

To install the package, you can use the following command:

```bash
pip install synthgenai
```

or if you want to use uv package manager, you can use the following command:

```bash
uv add synthgenai
```

or you can install the package directly from the source code using the following commands:

```bash
git clone https://github.com/Shekswess/synthgenai.git
uv build
pip install ./dist/synthgenai-{version}-py3-none-any.whl
```

## Setup Environment Variables 🔑

Before generating datasets, you'll need API keys for your chosen LLM provider. See all required environment variables:

```bash
synthgenai env-setup
```

### Quick Setup Example

For OpenAI (most common):

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Or create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Optional: Enable Detailed Logging

For debugging or development, you can enable detailed logging:

```bash
# Enable detailed logging (shows all debug information)
export SYNTHGENAI_DETAILED_MODE="false"

# Default NO logging (recommended for production)
export SYNTHGENAI_DETAILED_MODE="true"
```

## Your First Dataset 📊

### 1. Generate an Instruction Dataset

```bash
synthgenai generate instruction \
  --model "openai/gpt-4o" \
  --topic "Machine Learning" \
  --domain "Python Programming" \
  --entries 100
```

### 2. Generate a Preference Dataset

```bash
synthgenai generate preference \
  --model "anthropic/claude-3-5-sonnet-20241022" \
  --topic "Creative Writing" \
  --domain "Science Fiction" \
  --temperature 0.8 \
  --entries 50
```

### 3. Generate Raw Text Data

```bash
synthgenai generate raw \
  --model "gemini/gemini-1.5-flash" \
  --topic "Technology News" \
  --domain "Artificial Intelligence" \
  --language "English" \
  --entries 200
```

## CLI Commands Overview 📋

### Essential Commands

| Command                 | Description                         |
| ----------------------- | ----------------------------------- |
| `synthgenai generate`   | Generate a synthetic dataset        |
| `synthgenai list-types` | List available dataset types        |
| `synthgenai providers`  | Show supported LLM providers        |
| `synthgenai env-setup`  | Show required environment variables |
| `synthgenai examples`   | Show example commands               |

### Dataset Types

SynthGenAI supports 6 dataset types:

- **raw** - Unstructured text data
- **instruction** - Instruction-following conversations
- **preference** - Preference data with chosen/rejected responses
- **sentiment** - Text with sentiment labels
- **summarization** - Text-to-summary pairs
- **classification** - Text classification with labels

## Advanced Usage ⚡

### Asynchronous Generation (Faster)

```bash
synthgenai generate instruction \
  --model "groq/llama-3.1-70b-versatile" \
  --topic "Data Science" \
  --domain "Statistics" \
  --async \
  --entries 500
```

### Save to Hugging Face Hub

```bash
# Set your HF token first
export HF_TOKEN="your_hf_token_here"

synthgenai generate summarization \
  --model "openai/gpt-4o" \
  --topic "News Articles" \
  --domain "Technology" \
  --hf-repo "myorg/tech-summaries" \
  --entries 100
```

### Fine-Tuned Generation Parameters

```bash
synthgenai generate preference \
  --model "anthropic/claude-3-5-sonnet-20241022" \
  --topic "Code Review" \
  --domain "Python" \
  --temperature 0.3 \
  --top-p 0.9 \
  --max-tokens 2048 \
  --entries 100
```

## Troubleshooting 🔧

### Common Issues

**API Key Not Found:**

```bash
# Check your environment variables
echo $OPENAI_API_KEY

# Or use the env-setup command
synthgenai env-setup
```

**Model Not Found:**

```bash
# Check available providers and models
synthgenai providers

# Use correct model format: provider/model-name
# Example: "openai/gpt-4o" not "gpt-4o"
```

**Generation Errors:**

```bash
# Use verbose mode for debugging
synthgenai generate instruction \
  --model "openai/gpt-4o" \
  --topic "Test" \
  --domain "General" \
  --verbose \
  --entries 10

# Enable detailed logging for more debug information
export SYNTHGENAI_DETAILED_MODE="false"
synthgenai generate instruction \
  --model "openai/gpt-4o" \
  --topic "Test" \
  --domain "General" \
  --entries 10
```

## Next Steps 📚

- 📖 Explore [Dataset Types](../datasets/index.md) for detailed information
- ⚙️ Learn about [Configuration Options](../configurations/index.md)
- 🤖 Check [LLM Providers](../llm_providers/index.md) for provider-specific setup
- 💻 Browse [Examples](../examples/index.md) for code samples
- 🛠️ See [Contributing](../contributing/index.md) to contribute to the project

## Need Help? 💬

- 📋 Run `synthgenai --help` for command help
- 💡 Run `synthgenai examples` for more examples
- 🐛 Report issues on [GitHub](https://github.com/Shekswess/synthgenai/issues)
- 📚 Read the full documentation at [synthgenai.docs](https://shekswess.github.io/synthgenai/)
