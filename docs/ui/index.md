# Quick Start 🚀

To quickly start using the SynthGenAI, you need to have the package installed. You can install it using the following command:

```bash
pip install synthgenai
```

After installation, simply run the following command in your terminal:

```bash
synthgenai
```

This will launch the Gradio UI for generating synthetic datasets. You can also try the Gradio UI deployed on HuggingFace spaces [here](https://huggingface.co/spaces/Shekswess/SynthGenAI-UI).

<center>
    <img src="../assets/ui.png" alt="UI look"/>
    <br />
    *Gradio UI for generating synthetic datasets*
</center>

To create datasets, you need to set up the following fields in the UI:

- **LLM Model**: The LLM model to use (e.g., model_provider/model_name).
- **Temperature**: The temperature for the LLM.
- **Top P**: The top_p value for the LLM.
- **Max Tokens**: The maximum number of tokens for the LLM.
- **API Base**: The API base URL (optional).
- **API Key**: The API key (optional).
- **Dataset Type**: The type of dataset to generate (e.g., Raw, Instruction, Preference, Sentiment Analysis, Summarization, Text Classification).
- **Topic**: The topic of the dataset.
- **Domains**: The domains for the dataset (comma-separated).
- **Language**: The language of the dataset.
- **Additional Description**: Additional description for the dataset (optional).
- **Number of Entries**: The number of entries in the dataset.
- **Hugging Face Token**: The Hugging Face token.
- **Hugging Face Repo Name**: The Hugging Face repository name.
- **LLM Environment Variables**: Comma-separated environment variables for the LLM (e.g., KEY1=VALUE1, KEY2=VALUE2).
