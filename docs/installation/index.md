# Installation 🛠️

To install the package, you can use the following command:

```bash
pip install synthgenai
```

or you can install the package directly from the source code using the following command:

```bash
git clone https://github.com/Shekswess/synthgenai.git
uv build
pip install ./dist/synthgenai-{version}-py3-none-any.whl
```

## Requirements 📋

To use the package, you need to have the following requirements installed:

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) for building the package directly from the source code
- [Ollama](https://ollama.com/) running on your local machine if you want to use Ollama as an API provider (optional)
- [Langfuse](https://langfuse.com/) running on your local machine or in the cloud if you want to use Langfuse for tracebility (optional)
- [Hugging Face Hub](https://huggingface.co/) account if you want to save the generated datasets on Hugging Face Hub with generated token (optional)
