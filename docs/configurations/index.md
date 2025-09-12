# Configuration Types ⚙️

SynthGenAI uses three main configuration types to generate synthetic datasets. These configurations work together to define the dataset parameters, LLM settings, and overall generation process:

- [**Dataset Configuration**](./dataset_configuration.md) - Configure dataset parameters like topic, domains, language, and number of entries
- [**LLM Configuration**](./llm_configuration.md) - Configure the language model settings including model selection, temperature, and API credentials
- [**Dataset Generator Configuration**](./dataset_generator_configuration.md) - Combine dataset and LLM configurations for the complete generation setup

## Configuration Overview 🔧

### Dataset Configuration
The `DatasetConfig` defines what kind of dataset you want to generate, including:

- Topic and domains
- Target language
- Number of entries
- Additional descriptions

### LLM Configuration
The `LLMConfig` specifies which language model to use and how, including:

- Model provider and name
- Generation parameters (temperature, top_p, max_tokens)
- API credentials and endpoints

### Dataset Generator Configuration
The `DatasetGeneratorConfig` combines both configurations to create a complete setup for dataset generation across all supported dataset types.

## Environment Variables 🔐

SynthGenAI uses several environment variables to control behavior and configuration:

### Logging Configuration

- **`SYNTHGENAI_DETAILED_MODE`** - Controls logging verbosity
  - `"true"` (default): Minimal logging output, recommended for production
  - `"false"`: Detailed debug logging, useful for development and troubleshooting

```bash
# Enable detailed logging for debugging
export SYNTHGENAI_DETAILED_MODE="false"

# No logging (default)
export SYNTHGENAI_DETAILED_MODE="true"
```

### API Configuration

Environment variables for different LLM providers are documented in the [LLM Configuration](./llm_configuration.md) section.

## Usage Pattern 📋

All dataset generators follow the same configuration pattern:

1. Create a `DatasetConfig` with your dataset requirements
2. Create an `LLMConfig` with your preferred language model settings
3. Combine them into a `DatasetGeneratorConfig`
4. Use this configuration with any dataset generator type

This unified approach ensures consistency across all dataset types while providing the flexibility to customize both the dataset characteristics and the underlying language model behavior.
