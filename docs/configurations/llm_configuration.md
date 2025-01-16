# LLM Configuration 🤖

To configure the LLMs for generating datasets, you need to create an `LLMConfig` object. This object contains the configuration for the LLM model, including the model name, temperature, top_p, and max_tokens.

## Example 📖

```python
from synthgenai import LLMConfig

# Creating the LLMConfig
llm_config = LLMConfig(
    model="model_provider/model_name", # Check LiteLLM docs for more info
    temperature=0.5,
    top_p=0.9,
    max_tokens=2048,
    api_base="https://api.example.com",
    api_key="your_api_key"
)
```

## Parameters 🎛

- `model` (str): The name of the model to use. This should be in the format `model_provider/model_name`. (Required)
- `temperature` (float): The temperature to use for the model. This controls the randomness of the generated text. Must be between 0.0 and 1.0. (Optional, default: None)
- `top_p` (float): The top_p value to use for the model. This controls the nucleus sampling. Must be between 0.0 and 1.0. (Optional, default: None)
- `max_tokens` (int): The maximum number of tokens to generate. Must be greater than 1000. (Optional, default: None)
- `api_base` (AnyUrl): The API base URL for the LLM service. (Optional, default: None)
- `api_key` (str): The API key for authenticating with the LLM service. (Optional, default: None)

For more information on the available models and their configurations, refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/).
